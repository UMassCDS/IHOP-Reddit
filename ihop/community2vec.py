"""Trains community2vec models (word2vec where users are context and words are subreddits)

Inputs:
1) CSV of subreddits (other columns optional) to use as vocabulary

2) Multiple CSVs with user,words used as document contexts

.. TODO Code cleanup: introduce an additional class/mixin for embedding operations post-training that don't need to be tied to Gensim and training (e.g. tsne, distance operations, etc...)
"""
import argparse
import csv
import functools
import importlib.resources
import itertools
import json
import logging
import operator
import os
import pathlib
import shutil

import gensim
import pandas as pd
import pyspark.sql.functions as fn
from pyspark.sql.types import StringType, StructField, StructType

import ihop.utils

logger = logging.getLogger(__name__)

# Documents don't really need to be Reddit users, could be other text
INPUT_CSV_SCHEMA = StructType([StructField("subreddit_list", StringType(), False)])

# The filename for gensim vectors stored for community2vec models
VECTORS_FILE_NAME = "keyedVectors"

# Metrics json output keys
# These are the only ones that need to be used outside this class for displaying
# metrics in the app
MODEL_ID_KEY = "model_id"
CONTEXTS_PATH_KEY = "contexts_path"
ANALOGY_ACC_KEY = "analogy_accuracy"
DETAILED_ANALOGY_KEY = "detailed_analogy_results"
NUM_USERS_KEY = "num_users"
MAX_COMMENTS_KEY = "max_comments"


def get_vocabulary(vocabulary_csv, has_header=True, token_index=0, count_index=1):
    """Return vocabulary as dictionary str->int of frequency counts

    :param vocabulary_csv: path to csv vocabulary
    """
    vocab = {}
    with open(vocabulary_csv) as vocab_in:
        vocab_reader = csv.reader(vocab_in)
        if has_header:
            next(vocab_reader)
        for row in vocab_reader:
            vocab[row[token_index]] = int(row[count_index])

    return vocab


def get_w2v_params_from_spark_df(spark, contexts_path):
    """Returns number of contexts and longest context size from a spark dataframe.
    In the community2vec setting this corresponds to the number of users and largest number of comments for a single user

    :param spark: Spark context
    :param contexts_path: str, path to a csv dataframe matching the input schema
    """
    context_df = spark.read.csv(contexts_path, header=False, schema=INPUT_CSV_SCHEMA)

    num_users = context_df.count()

    max_comments = (
        context_df.select(fn.split("subreddit_list", " ").alias("subreddit_list"))
        .select(fn.size("subreddit_list").alias("num_comments"))
        .agg(fn.max("num_comments"))
        .head()[0]
    )

    return num_users, max_comments


def analogy_sections_to_str(detailed_accs):
    """Parses the sectional analogy results from Gensim to a string for logging, displays, etc...
    :param detailed_accs: list of dict with 'correct', 'incorrect' and 'section' keys
    """
    section_strings = []
    for dr in detailed_accs:
        section_correct = len(dr["correct"])
        total_section_examples = section_correct + len(dr["incorrect"])
        section_strings.append(
            f"{dr['section']}:{section_correct}/{total_section_examples}"
        )

    return ",".join(section_strings)


class EpochLossCallback(gensim.models.callbacks.CallbackAny2Vec):
    """Callback to print loss after each epoch.
    See https://stackoverflow.com/questions/54888490/gensim-word2vec-print-log-loss
    """

    def __init__(self):
        self.epoch = 1
        self.loss_to_be_subed = 0

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        loss_now = loss - self.loss_to_be_subed
        self.loss_to_be_subed = loss
        logger.info(f"Loss after epoch {self.epoch}: {loss_now}")
        self.epoch += 1


class SaveVectorsCallback(gensim.models.callbacks.CallbackAny2Vec):
    """Callback to save embeddings for a model after each epoch"""

    def __init__(self, save_vector_prefix):
        """
        :param save_vector_prefix: str, directory and basename for the path for saving vectors, epoch will be appended
        """
        self.save_vector_prefix = save_vector_prefix
        self.epoch = 1

    def on_epoch_end(self, w2v_model):
        vector_out = self.save_vector_prefix + f"_epoch_{self.epoch}"
        logger.info(f"Saving vectors for epoch {self.epoch} to {vector_out}")
        w2v_model.wv.save(vector_out)
        self.epoch += 1


class AnalogyAccuracyCallback(gensim.models.callbacks.CallbackAny2Vec):
    """Callback for reporting analogy accuracy after each epoch"""

    def __init__(self, analogies_path, case_insensitive=False):
        """
        :param analogies_path: str, path to the analogies file for Gensim's KeyedVectors
        :param case_insensitive: boolean, set to True to deal with case mismatch in analogy pairs. For Reddit, this should typically be False.
        """
        self.analogies_path = analogies_path
        self.epoch = 1
        self.case_insensitive = case_insensitive

    def on_epoch_end(self, w2v_model):
        max_vocab = len(w2v_model.wv.index_to_key) + 1
        score, _ = w2v_model.wv.evaluate_word_analogies(
            self.analogies_path,
            restrict_vocab=max_vocab,
            case_insensitive=self.case_insensitive,
        )
        logger.info(f"Analogy score after epoch {self.epoch}: {score}")
        self.epoch += 1


class GensimCommunity2Vec:
    """Implements Community2Vec Skip-gram with negative sampling (SGNS) using the gensim Word2Vec model.
    Determines the appropriate window-size and sets vocabulary according to the
    filtered subreddits.
    """

    # When models are saved, what are the files named
    MODEL_SAVE_NAME = "word2vec.pickle"
    PARAM_SAVE_NAME = "parameters.json"

    def __init__(
        self,
        vocab_dict,
        contexts_path,
        max_comments=0,
        num_users=0,
        vector_size=150,
        negative=20,
        sample=0,
        alpha=0.025,
        min_alpha=0.0001,
        seed=1,
        epochs=5,
        batch_words=10000,
        workers=3,
    ):
        """
        Instantiates a gensim Word2Vec model for Community2Vec
        :param vocab_dict: dict, str->int storing frequency counts of the vocab elements.
        :param contexts_path: Path to a text file storing the subreddits a user commented on, one user per line. Can be compressed as a bzip2 or gzip.
        :param max_comments: int, maximum window for skip grams (for c2v this should be 'infinity' or the largest number of comments for a single user in the data)
        :param num_users: int, number of contexts/users
        :param vector_size: int, embedding size, passed to gensim Word2Vec
        :param negative: int, how many 'noise words' should be drawn, passed to gensim Word2Vec
        :param sample: float, the threshold for configuring which higher-frequency words are randomly downsampled, useful range is (0, 1e-5), passed to gensim Word2Vec, defaults to 0
        :param alpha: float, initial learning rate, passed to gensim Word2Vec
        :param min_alpha: float, minimum for learning rate decay
        :param seed: int, randomSeed for initializing embeddings, passed to gensim Word2Vec
        :param epochs: int, Number of iterations over the corpus, passed to gensim Word2Vec
        :param batch_words: int, Target size (in words) for batches of examples passed to worker threads
        :param workers: int, number of worker threads for training
        """
        self.contexts_path = contexts_path
        self.max_comments = max_comments
        self.num_users = num_users
        self.epochs = epochs
        self.w2v_model = gensim.models.word2vec.Word2Vec(
            vector_size=vector_size,
            min_count=0,
            window=self.max_comments,
            sg=1,
            hs=0,
            negative=negative,
            sample=sample,
            seed=seed,
            alpha=alpha,
            epochs=epochs,
            batch_words=batch_words,
            min_alpha=min_alpha,
            workers=workers,
        )
        self.w2v_model.build_vocab_from_freq(vocab_dict)

    def get_params_as_dict(self):
        """Returns dictionary of parameters for tracking experiments."""
        return {
            NUM_USERS_KEY: self.num_users,
            MAX_COMMENTS_KEY: self.max_comments,
            CONTEXTS_PATH_KEY: self.contexts_path,
            "epochs": self.epochs,
            "vector_size": self.w2v_model.vector_size,
            "skip_gram": self.w2v_model.sg,
            "hierarchical_softmax": self.w2v_model.hs,
            "negative": self.w2v_model.negative,
            "ns_exponent": self.w2v_model.ns_exponent,
            "alpha": self.w2v_model.alpha,
            "min_alpha": self.w2v_model.min_alpha,
            "seed": self.w2v_model.seed,
            "batch_words": self.w2v_model.batch_words,
            "sample": self.w2v_model.sample,
        }

    def train(
        self,
        save_vectors_prefix=None,
        analogies_path=None,
        epoch_analogies=True,
        case_insensitive=False,
        **kwargs,
    ):
        """Trains the word2vec model. Returns the result from gensim.
        :param save_vectors_prefix: str or None, use set this to save vectors after each epoch. The epoch will be appended to the filename.
        :param analogies_path: str, optional. If specified use this file to report analogy performance after each epoch
        :param epoch_analogies: boolean, True if you want report performance on the default subreddit analogies after each epoch
        :param case_insensitive: boolean, set to True to deal with case mismatch in analogy pairs. For Reddit, this should typically be False.
        :param **kwargs: passed to gensim Word2Vec.train()
        """
        callbacks = [EpochLossCallback()]
        if save_vectors_prefix:
            callbacks.append(SaveVectorsCallback(save_vectors_prefix))

        # Solve given analogies after each epoch or use default
        if analogies_path:
            callbacks.append(AnalogyAccuracyCallback(analogies_path, case_insensitive))
        elif epoch_analogies:
            with importlib.resources.path(
                "ihop.resources.analogies", "subreddit_analogies.txt"
            ) as default_analogies:
                callbacks.append(
                    AnalogyAccuracyCallback(str(default_analogies), case_insensitive)
                )

        train_result = self.w2v_model.train(
            gensim.models.word2vec.PathLineSentences(self.contexts_path),
            total_examples=self.num_users,
            epochs=self.epochs,
            callbacks=callbacks,
            **kwargs,
        )
        return train_result

    def save(self, save_dir):
        """Save the current model object with parameters in json and the word2vec model saved using the gensim save() method

        :param save_dir: str, path of directory to save model and parameters
        """
        os.makedirs(save_dir, exist_ok=True)
        w2v_path = os.path.join(save_dir, self.MODEL_SAVE_NAME)
        self.w2v_model.save(w2v_path)
        parameters_path = os.path.join(save_dir, self.PARAM_SAVE_NAME)

        with open(parameters_path, "w") as f:
            json.dump(self.get_params_as_dict(), f)

    def save_vectors(self, save_path):
        """Save only the embeddings from this model as gensim KeyedVectors. These can't be used for further training of the Community2Vec model, but have smaller RAM footprint and are more efficient"""
        self.w2v_model.wv.save(save_path)

    def get_normed_vectors(self):
        """Returns the normed embedding weights for the Gensim Keyed Vectors"""
        return self.w2v_model.wv.get_normed_vectors()

    def get_index_to_key(self):
        """Returns the vocab of the Word2Vec embeddings as an indexed list of strings."""
        return self.w2v_model.wv.index_to_key

    def get_index_as_dict(self):
        """Returns the index of the Word2Vec embeddings as a dictionary mapping int -> string"""
        return dict(enumerate(self.w2v_model.wv.index_to_key))

    def score_analogies(self, analogies_path=None, case_insensitive=False):
        """ "Returns the trained embedding's accuracy for solving subreddit algebra analogies and detailed section results. If not file path is specified, return results on the default sports and university-city analogies from ihop.resources.analogies.

        :param analogies_path: str, optional. Define to use a particular analogies file where lines are whitespace separated 4-tuples and split into sections by ': SECTION NAME' lines
        :param case_insensitive: boolean, set to True to deal with case mismatch in analogy pairs. For Reddit, this should typically be False.
        """
        max_vocab = len(self.get_index_to_key()) + 1
        if analogies_path:
            return self.w2v_model.wv.evaluate_word_analogies(
                analogies_path,
                restrict_vocab=max_vocab,
                case_insensitive=case_insensitive,
            )
        else:
            with importlib.resources.path(
                "ihop.resources.analogies", "subreddit_analogies.txt"
            ) as default_analogies:
                return self.w2v_model.wv.evaluate_word_analogies(
                    default_analogies,
                    restrict_vocab=max_vocab,
                    case_insensitive=case_insensitive,
                )

    def get_nearest_neighbors(self, term, topn):
        """Returns the list of topn nearest neighbors to the given term in the community2vec model. If the term isn't in the model's vocab, an empty list is returned.

        :param term: str, the subreddit term you'd like to get neighbors for
        :param topn: int, the number of top nearest neighbors to return
        """
        if term in self.w2v_model.wv:
            term_score_tuples = self.w2v_model.wv.most_similar(term, topn=topn)
            return [p[0] for p in term_score_tuples]

        return []

    @classmethod
    def init_with_spark(
        cls,
        spark,
        vocab_dict,
        contexts_path,
        vector_size=150,
        negative=20,
        sample=0,
        alpha=0.025,
        min_alpha=0.0001,
        seed=1,
        epochs=5,
        batch_words=10000,
        workers=3,
    ):
        """Instantiates a community2vec model using max_comments and num_users determined the contexts_path file using Spark.
        :param spark: Spark context
        :param vocab_dict: dict, str->int storing frequency counts of the vocab elements.
        :param contexts_path: Path to a text file storing the subreddits a user commented on, one user per line. Can be compressed as a bzip2 or gzip.
        :param vector_size: int, embedding size, passed to gensim Word2Vec
        :param negative: int, how many 'noise words' should be drawn, passed to gensim Word2Vec
        :param sample: float, the threshold for configuring which higher-frequency words are randomly downsampled, useful range is (0, 1e-5), passed to gensim Word2Vec, defaults to 0
        :param alpha: float, initial learning rate, passed to gensim Word2Vec
        :param min_alpha: float, minimum for learning rate decay
        :param seed: int, randomSeed for initializing embeddings, passed to gensim Word2Vec
        :param epochs: int, Number of iterations over the corpus, passed to gensim Word2Vec
        :param batch_words: int, Target size (in words) for batches of examples passed to worker threads
        :param workers: int, number of worker threads for training
        """
        num_users, max_comments = get_w2v_params_from_spark_df(spark, contexts_path)

        return cls(
            vocab_dict,
            contexts_path,
            max_comments,
            num_users,
            vector_size,
            negative,
            sample,
            alpha,
            min_alpha,
            seed,
            epochs,
            batch_words,
            workers,
        )

    @classmethod
    def load(cls, load_dir):
        """Returns a GensimCommunity2Vec object that was pickled in the file.
        :param load_dir: str, directory to load the objects from
        """
        json_file = os.path.join(load_dir, GensimCommunity2Vec.PARAM_SAVE_NAME)
        w2v_file = os.path.join(load_dir, GensimCommunity2Vec.MODEL_SAVE_NAME)
        with open(json_file, "r") as j:
            json_params = json.load(j)
        model = GensimCommunity2Vec(
            {},
            json_params[CONTEXTS_PATH_KEY],
            json_params[MAX_COMMENTS_KEY],
            json_params[NUM_USERS_KEY],
            epochs=json_params["epochs"],
        )
        model.w2v_model = gensim.models.Word2Vec.load(w2v_file)
        return model


class GridSearchTrainer:
    """Trains multiple community2vec models, storing model results and vectors as
    it goes. There is no held out test set, performance is determined by accuracy on solving analogies.
    """

    # If the user doesn't give a param grid, just train a single default model
    DEFAULT_PARAM_GRID = {
        "vector_size": [150],
        "negative": [20],
        "sample": [0],
        "alpha": [0.025],
        "min_alpha": [0.0001],
    }

    PERFORMANCE_CSV_NAME = "analogy_accuracy_results.csv"
    BEST_MODEL_DIR_NAME = "best_model"
    METRICS_JSON_NAME = "metrics.json"

    def __init__(
        self,
        vocab_csv,
        contexts_path,
        num_contexts,
        max_context_window,
        model_output_dir,
        param_grid=None,
        analogies_path=None,
        case_insensitive=False,
        keep_all=False,
    ):
        """
        :param vocab_csv: Path to csv storing vocab with counts in the corpus
        :param contexts_path: Path to a text file storing the subreddits a user commented on, one user per line. Can be compressed as a bzip2 or gzip.
        :param num_contexts: int, the number of contexts/users/documents
        :param max_context_window: int, the largest context window/number of comments by a user
        :param model_output_dir: str/Path to directory, where to save model results during training
        :param param_grid: dict, keys match paramters to GensimCommunity2Vec models, values are lists over which to iterate for grid search
        :param analogies_path: str, optional. Define to use a particular analogies file where lines are whitespace separated 4-tuples and split into sections by ': SECTION NAME' lines
        :param case_insensitive: boolean, set to True to deal with case mismatch in analogy pairs. For Reddit c2v, this should typically be False.
        :param keep_all: boolean, set to True to write every trained model to disk, rather than keeping the best model. If this flag is true, then a directory will be created in model_output_dir for each model in the grid search, rather than just the best one
        """
        self.vocab_csv = vocab_csv
        self.vocab_dict = get_vocabulary(vocab_csv)
        self.contexts_path = contexts_path
        self.num_contexts = num_contexts
        self.max_context_window = max_context_window
        self.model_output_dir = model_output_dir
        self.keep_all = keep_all

        self.best_acc = 0.0
        self.best_model_id = None
        self.best_model_path = os.path.join(
            self.model_output_dir, self.BEST_MODEL_DIR_NAME
        )
        self.best_vectors_path = os.path.join(self.best_model_path, VECTORS_FILE_NAME)

        if param_grid is None or len(param_grid) == 0:
            self.param_grid = GridSearchTrainer.DEFAULT_PARAM_GRID
            self.num_models = 0
        else:
            self.param_grid = param_grid

        self.num_models = functools.reduce(
            operator.mul, [len(x) for x in self.param_grid.values()]
        )

        self.analogy_results = list()
        self.analogies_path = analogies_path
        self.case_insensitive = case_insensitive

    def train(self, epochs=5, workers=3, **kwargs):
        """Train models according to the param grid defined for this object. Saves each model and analogy results after training, updating the best analogy accuracy and best model parameters
        as needed.

        Returns the best_acc and the unique identifier for the best model upon completion.

        :param epochs: int, number of epochs to train each model
        :param workers: int, number of threads used for training each individual model, passed to Gensim
        :param **kwargs: any additional parameters that need to be passed to GensimCommunity2Vec that aren't defined in the param grid

        """
        if os.path.exists(self.model_output_dir):
            logger.warning(
                "Specified model directory %s already exists", self.model_output_dir
            )

        for i, param_dict in enumerate(self.expand_param_grid_to_list()):
            model_id = self.get_model_id(param_dict)
            save_vectors_prefix = None
            if self.keep_all:
                curr_model_path, save_vectors_prefix = self.prep_model_output_dir(
                    model_id
                )
                logger.info("Training model %s of %s: %s", i, self.num_models, model_id)

            c2v_model = GensimCommunity2Vec(
                self.vocab_dict,
                self.contexts_path,
                self.max_context_window,
                self.num_contexts,
                epochs=epochs,
                workers=workers,
                **param_dict,
            )
            c2v_model.train(
                analogies_path=self.analogies_path,
                case_insensitive=self.case_insensitive,
                save_vectors_prefix=save_vectors_prefix,
                **kwargs,
            )

            if self.keep_all:
                logger.debug("Saving trained model to: %s", curr_model_path)
                c2v_model.save(curr_model_path)
                c2v_model.save_vectors(save_vectors_prefix)

            acc, detailed_accs = c2v_model.score_analogies(
                self.analogies_path,
            )
            logger.info(
                "Model id %s achieved %s accuracy on analogy task", model_id, acc
            )
            results_dict = self.get_single_model_full_results(
                model_id, c2v_model, acc, detailed_accs
            )
            self.analogy_results.append(results_dict)

            if self.keep_all:
                self.write_single_model_metrics_json(curr_model_path, results_dict)

            if acc >= self.best_acc:
                logger.info("New best model %s with analogy accuracy %s", model_id, acc)
                if os.path.exists(self.best_model_path):
                    logger.debug(
                        "Removing old best model path: %s", self.best_model_path
                    )
                    shutil.rmtree(self.best_model_path)
                self.best_acc = acc
                self.best_model_id = model_id
                logger.info("Saving new best model to %s", self.best_model_path)
                os.mkdir(self.best_model_path)
                c2v_model.save(self.best_model_path)
                c2v_model.save_vectors(self.best_vectors_path)
                self.write_single_model_metrics_json(self.best_model_path, results_dict)

        return self.best_acc, self.best_model_id

    def get_model_id(self, grid_param_dict):
        """Returns a string that uniquely names the model within this
        grid search setting.
        """
        model_id_elems = list()
        # Sort keys alphabetically, remove any underscores from keys and camel case
        for k, v in sorted(grid_param_dict.items()):
            k_split = k.split("_")
            updated_key = "".join([k_split[0], *[x.title() for x in k_split[1:]]])
            model_id_elems.append(f"{updated_key}{v}")

        return "_".join(model_id_elems)

    def expand_param_grid_to_list(self):
        """Returns the parameter grid to a list of dicts to iterate over when training."""
        result = list()
        for v in itertools.product(*self.param_grid.values()):
            result.append(dict(zip(self.param_grid.keys(), v)))
        return result

    def model_analogy_results_as_dataframe(self):
        """Returns the training results as rows in a pandas DataFrame with columns
        for model id, analogy accuracy and parameters defined for grid search
        """
        return pd.DataFrame.from_records(self.analogy_results)

    def write_performance_results(self):
        """Writes analogy accuracy results for all models to a csv file in the model_otuput_directory"""
        self.model_analogy_results_as_dataframe().to_csv(
            os.path.join(self.model_output_dir, self.PERFORMANCE_CSV_NAME), index=False
        )

    def get_single_model_full_results(self, model_id, c2v_model, acc, detailed_accs):
        """Returns all the paramenters and metrics for experimental results tracking for a single model in a dictionary.

        :param model_id: str, unique identifier for this model
        :param c2v_model: GensimCommunity2Vec object
        :param acc: float, accuracy on the analogy task
        :param detailed_accs: str, the detailed accuracy results broken down by category as returned by Gensim
        """
        results_dict = {
            MODEL_ID_KEY: model_id,
            CONTEXTS_PATH_KEY: self.contexts_path,
            ANALOGY_ACC_KEY: acc,
            DETAILED_ANALOGY_KEY: analogy_sections_to_str(detailed_accs),
        }
        results_dict.update(c2v_model.get_params_as_dict())
        return results_dict

    def write_single_model_metrics_json(self, dir_path, metrics_dict):
        """Writes the given dictionary of metrics to a json file in the specified path

        :param dir_path: str, path to folder where json file will be saved
        :param metrics_dict: dict, contents to store in json
        """
        metrics_json_path = os.path.join(dir_path, self.METRICS_JSON_NAME)
        logger.debug("Writing model metrics and params to %s", metrics_json_path)
        with open(metrics_json_path, "w") as metrics_json:
            json.dump(metrics_dict, metrics_json)

    def prep_model_output_dir(self, model_name):
        """Returns the full absolute path to folder where model artifacts will be stored and path for saving vectors given a model name. Ensures the model path folder exists

        :param model_name: str, the folder name where all model artifacts will be stored
        """
        model_path = os.path.join(self.model_output_dir, model_name)
        os.makedirs(model_path, exist_ok=True)
        vectors_path = os.path.join(model_path, VECTORS_FILE_NAME)

        return model_path, vectors_path


def train_with_hyperparam_tuning(
    vocab_csv,
    contexts,
    param_grid,
    num_users,
    max_comments,
    model_output_dir,
    workers,
    epochs,
    analogies,
    case_insensitive=False,
    keep_all=False,
    **kwargs,
):
    """
    Trains models using grid search, then saves results to the specified
    output directory.
    :param vocab_csv: Path to csv storing vocab with counts in the corpus
    :param contexts_path: Path to a text file storing the subreddits a user commented on, one user per line. Can be compressed as a bzip2 or gzip.
    :param num_users: int, the number of contexts/users/documents
    :param max_comments: int, the largest context window/number of comments by a user
    :param model_output_dir: str/Path to directory, where to save model results during training
    :param param_grid: dict, keys match paramters to GensimCommunity2Vec models, values are lists over which to iterate for grid search
    :param analogies: str, optional. Define to use a particular analogies file where lines are whitespace separated 4-tuples and split into sections by ': SECTION NAME' lines
    :param case_insensitive: boolean, whether analogies should be done case insensitive or not, for Reddit typically False.
    :param keep_all: boolean, set to True to write every trained model to disk, rather than keeping the best model. If this flag is true, then a directory will be created in model_output_dir for each model in the grid search, rather than just the best one
    :param kwargs: Passed to the Gensim Model at training time
    """
    logger.info("Param grid: %s", param_grid)
    grid_trainer = GridSearchTrainer(
        vocab_csv,
        contexts,
        num_users,
        max_comments,
        model_output_dir,
        param_grid,
        analogies_path=analogies,
        case_insensitive=case_insensitive,
        keep_all=keep_all,
    )
    grid_trainer.train(epochs, workers, **kwargs)
    grid_trainer.write_performance_results()


parser = argparse.ArgumentParser(
    description="Training community2vec models from pre-processed data using a grid search to optimize performance on analogies."
)

parser.add_argument(
    "--config",
    type=pathlib.Path,
    help="JSON file used to override default logging and spark configurations",
)

parser.add_argument(
    "--contexts",
    "-c",
    help="Path to context training data. Can be raw text or a directory of raw text, optionally compressed.",
    required=True,
)
parser.add_argument(
    "--vocab_csv",
    "-v",
    help="CSV file with vocabulary items and their counts in the corpus",
    required=True,
)
parser.add_argument(
    "--output_dir",
    "-o",
    help="Directory to store model files and peformance results",
    required=True,
)
parser.add_argument(
    "--param_grid",
    "-p",
    nargs="?",
    type=json.loads,
    default="{}",
    help="JSON defining the parameter grid. Keys are strings corresponding to the GensimCommunity2Vec params, values are lists of values to try. Defaults to a single model using the GensimCommunity2Vec default params.",
)
parser.add_argument(
    "--workers",
    "-w",
    type=int,
    default=3,
    help="Number of workers for Gensim training. Defaults to 3.",
)
parser.add_argument(
    "--epochs",
    "-e",
    type=int,
    default=5,
    help="Number of epochs to train each model. Defaults to 5.",
)
parser.add_argument(
    "--analogies",
    "-a",
    nargs="?",
    help="Path to an anlogies file for evaluating performance. Optional, if unspecified a default consisting of sports and university towns will be used.",
)
parser.add_argument(
    "--keep-all",
    action="store_true",
    help="Use this flag to keep all models trained during hyperparameter tuning with the vectors after each epoch. Otherwise only the best model will be kept. Note that using this will create LOTS of model folders and vector files.",
)


if __name__ == "__main__":
    try:
        args = parser.parse_args()
        config = ihop.utils.parse_config_file(args.config)
        ihop.utils.configure_logging(config[1])
        logger.debug("Script arguments: %s", args)
        spark = ihop.utils.get_spark_session("IHOP Community2Vec", config[0])

        num_users, max_comments = get_w2v_params_from_spark_df(spark, args.contexts)
        spark.stop()
        train_with_hyperparam_tuning(
            args.vocab_csv,
            args.contexts,
            args.param_grid,
            num_users,
            max_comments,
            args.output_dir,
            args.workers,
            args.epochs,
            args.analogies,
            args.keep_all,
        )
    except Exception:
        logger.error("Fatal error while training community2vec", exc_info=True)
