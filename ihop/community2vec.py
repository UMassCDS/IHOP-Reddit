"""Trains community2vec models (word2vec where users are context and words are subreddits)

Inputs:
1) CSV of subreddits (other columns optional) to use as vocabulary

2) Multiple CSVs with user,words used as document contexts

.. TODO Implement community2vec as described in https://www.tensorflow.org/tutorials/text/word2vec with negative sampling or using gensim

.. TODO Account for downsampling of skipgrams as described in https://github.com/BIU-NLP/word2vecf/blob/master/word2vecf.c#L421 (I think this can be done in tensorflow with the sampling_table) or with the sample kw option in skipgrams

# TODO Considerations for cross validation
"""
import csv
import json
import logging
import os

import gensim
import pandas as pd
import pyspark.sql.functions as fn
from sklearn.manifold import TSNE

# TODO Logging should be configurable, but for now just turn it on for Gensim
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


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
        logging.info(f'Loss after epoch {self.epoch}: {loss_now}')
        self.epoch += 1


class SaveVectorsCallback(gensim.models.callbacks.CallbackAny2Vec):
    """Callback to save embeddings for a model after each epoch
    """
    def __init__(self, save_vector_prefix):
        self.save_vector_prefix = save_vector_prefix
        self.epoch = 1

    def on_epoch_end(self, w2v_model):
        vector_out = self.save_vector_prefix + f"_epoch_{self.epoch}"
        logging.info(f'Saving vectors for epoch {self.epoch} to {vector_out}')
        w2v_model.wv.save(vector_out)
        self.epoch += 1


class GensimCommunity2Vec:
    """Implements Community2Vec Skip-gram with negative sampling (SGNS) using the gensim Word2Vec model.
    Determines the appropriate window-size and sets vocabulary according to the
    filtered subreddits.
    """
    # When models are saved, what are the files named
    MODEL_SAVE_NAME = "word2vec.pickle"
    PARAM_SAVE_NAME = "parameters.json"

    def __init__(self, vocab_dict, contexts_path, max_comments=0, num_users=0, vector_size=150, negative=20, sample=0, alpha=0.025, min_alpha=0.0001, seed=1, epochs=5, batch_words=10000, workers=3):
        """
        Instantiates a gensim Word2Vec model for Community2Vec
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
        self.contexts_path = contexts_path
        self.max_comments = max_comments
        self.num_users = num_users
        self.epochs = epochs
        self.w2v_model = gensim.models.word2vec.Word2Vec(vector_size=vector_size, min_count=0, window=self.max_comments, sg=1, hs=0, negative=negative, sample=sample, seed=seed, alpha=alpha, epochs=epochs, batch_words=batch_words, min_alpha=min_alpha, workers=workers)
        self.w2v_model.build_vocab_from_freq(vocab_dict)

    def get_params_as_dict(self):
        """Returns dictionary of parameters that aren't stored in gensim's Word2Vec save()
        """
        return {"num_users": self.num_users,
                "max_comments": self.max_comments,
                "contexts_path": self.contexts_path,
                "epochs": self.epochs
                }

    def train(self, save_vectors_prefix=None, **kwargs):
        """Trains the word2vec model. Returns the result from gensim.
        :param save_vectors_prefix: str or None, use set this to save vectors after each epoch. The epoch will be appended to the filename.
        :param **kwargs: passed to gensim Word2Vec.train()
        """
        callbacks = [EpochLossCallback()]
        if save_vectors_prefix:
            callbacks.append(SaveVectorsCallback(save_vectors_prefix))
        train_result = self.w2v_model.train(gensim.models.word2vec.PathLineSentences(self.contexts_path), total_examples=self.num_users, epochs=self.epochs, callbacks=callbacks, **kwargs)
        return train_result

    def save(self, save_dir):
        """Save the current model object with parameters in json and the word2vec model saved using the gensim save() method

        :param save_dir: str, path of directory to save model and parameters
        """
        if not os.path.exists(save_dir) and os.path.isdir(save_dir):
            os.mkdir(save_dir)
        w2v_path = os.path.join(save_dir, self.MODEL_SAVE_NAME)
        self.w2v_model.save(w2v_path)
        parameters_path = os.path.join(save_dir, self.PARAM_SAVE_NAME)

        with open(parameters_path, 'w') as f:
            json.dump(self.get_params_as_dict(), f)

    def save_vectors(self, save_path):
        """Save only the embeddings from this model as gensim KeyedVectors. These can't be used for further training of the Community2Vec model, but have smaller RAM footprint and are more efficient
        """
        self.w2v_model.wv.save(save_path)

    def get_normed_vectors(self):
        """Returns the normed embedding weights for the Gensim Keyed Vectors
        """
        return self.w2v_model.wv.get_normed_vectors()

    def get_tsne_dataframe(self, key_col="subreddit", **kwargs):
        """Fits a TSNE representation of the dataframe.
        Returns the results as both a pandas dataframe and the resulting TSNE projection as a numpy array

        :param kwargs: dict params passed to sklearn's TNSE model
        """
        tsne_fitter = TSNE(**kwargs, init="pca", metric="cosine", learning_rate="auto")
        tsne_projection = tsne_fitter.fit_transform(self.get_normed_vectors())
        dataframe_elements = list()
        for i, vocab_elem in enumerate(self.w2v_model.wv.index_to_key):
            elem_proj = tsne_projection[i]
            dataframe_elements.append((vocab_elem, elem_proj[0], elem_proj[1]))

        dataframe = pd.DataFrame.from_records(dataframe_elements, columns=[key_col, "tsne_x", "tsne_y"])
        return dataframe, tsne_projection

    def get_index_to_key(self):
        """Returns the vocab of the Word2Vec embeddings as an indexed list of strings.
        """
        return self.w2v_model.wv.index_to_key


    @classmethod
    def init_with_spark(cls, spark, vocab_dict, contexts_path, vector_size=150, negative=20, sample=0, alpha=0.025, min_alpha=0.0001, seed=1, epochs=5, batch_words=10000, workers=3):
        """Instantiates a community2vec model using max_comments and num_users determined the contexts_path file using Spark.
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
        context_df = spark.read.csv(contexts_path)
        max_comments = context_df.select(fn.split("_c0", " ").alias("subreddit_list")).select("subreddit_list", fn.size("subreddit_list").alias("num_comments")).agg(fn.max("num_comments")).head()[0]
        num_users = context_df.count()

        return cls(vocab_dict, contexts_path, max_comments, num_users,
            vector_size, negative, sample, alpha, min_alpha,
            seed, epochs, batch_words, workers)

    @classmethod
    def load(cls, load_dir):
        """Returns a GensimCommunity2Vec object that was pickled in the file.
        :param load_dir: str, directory to load the objects from
        """
        json_file = os.path.join(load_dir, GensimCommunity2Vec.PARAM_SAVE_NAME)
        w2v_file = os.path.join(load_dir, GensimCommunity2Vec.MODEL_SAVE_NAME)
        with open(json_file, 'r') as j:
            json_params = json.load(j)
        model = GensimCommunity2Vec({}, json_params["contexts_path"],
                        json_params["max_comments"], json_params["num_users"],
                        epochs=json_params["epochs"])
        model.w2v_model = gensim.models.Word2Vec.load(w2v_file)
        return model








