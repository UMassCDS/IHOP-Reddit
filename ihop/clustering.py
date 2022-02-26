"""Train clusters on community2vec embeddings and other embedding data.

.. TODO: Support clustering of documents based on TF-IDF, not just c2v embeddings
.. TODO: Implement training of topic models on text: tf-idf-> KMeans, Hierarchical Dirichlet Processes
.. TODO: Lift document level clusters to subreddit level (will we need spark again or will pandas be sufficient?)
.. TODO: AuthorTopic models with subreddits as the metadata field (instead of author)
.. TODO: Data also needs to be serialized with the cluster models, preferably as a pandas dataframe and not a spark dataframe (for LDA)
"""
import argparse
import copy
import json
import logging
import os

import gensim.models as gm
import joblib
import numpy as np
import pandas as pd
import pytimeparse
from sklearn.cluster import KMeans, AffinityPropagation, AgglomerativeClustering
from sklearn import metrics

import ihop

logger = logging.getLogger(__name__)
# TODO Logging should be configurable, but for now just turn it on for Gensim
logging.basicConfig(
    format='%(name)s : %(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class ClusteringModelFactory:
    """Return appropriate class given input params
    """
    AFFINITY_PROP = "affinity_propagation"
    AGGLOMERATIVE = "agglomerative_clustering"
    KMEANS = "kmeans"
    LDA = "lda"

    # Default parameters used when instantiating the model
    DEFAULT_MODEL_PARAMS = {
        AFFINITY_PROP: {'affinity': 'precomputed', 'max_iter': 1000, 'convergence_iter': 50, 'random_state': 100},
        AGGLOMERATIVE: {'n_clusters': 250, 'linkage': 'average', 'affinity': 'cosine', 'compute_distances': True},
        KMEANS: {'n_clusters': 250, 'random_state': 100},
        LDA: {'num_topics': 250, 'alpha': 'asymmetric',
              'eta': 'symmetric', 'iterations': 1000}
    }

    @classmethod
    def init_clustering_model(cls, model_choice, data, index, model_name=None, **kwargs):
        """Returns a ClusteringModel instance instantiated with the appropriate parameters and ready to train on the given data.
        :param model_choice: str, type of model to instantiate
        :param data: data used to train the model, type is dependent on the model choice
        :param index: dict, int -> str, how to name each data point, important for exporting data for users and visualizations
        :param model_name: str, used to identify the model in output and string representation. If left as None, then choice value will be used as model name
        :param kwargs: parameters to pass to the sklearn or Gensim model
        """
        if model_name is None:
            model_id = model_choice
        else:
            model_id = model_name

        if model_choice not in cls.DEFAULT_MODEL_PARAMS:
            raise ValueError(f"Model choice {model_choice} is not supported")

        parameters = {}
        parameters.update(cls.DEFAULT_MODEL_PARAMS[model_choice])
        parameters.update(kwargs)

        if model_choice == cls.KMEANS:
            return ClusteringModel(data, KMeans(**parameters), model_id, index)
        elif model_choice == cls.AFFINITY_PROP:
            if isinstance(data, gm.keyedvectors.KeyedVectors) and parameters['affinity'] == "precomputed":
                precomputed_distances = np.zeros((len(index), len(index)))
                for i, v in index.items():
                    precomputed_distances[i] = np.array(data.distances(v))

                return ClusteringModel(precomputed_distances, AffinityPropagation(**parameters), model_choice, index)
            else:
                return ClusteringModel(data, AffinityPropagation(**parameters), model_id, index)
        elif model_choice == cls.AGGLOMERATIVE:
            return ClusteringModel(data, AgglomerativeClustering(**parameters), model_id, index)
        elif model_choice == cls.LDA:
            return GensimLDAModel(data, model_id, index, **parameters)
        else:
            raise ValueError(f"Model type '{model_choice}' is not supported")


class ClusteringModel:
    """Wrapper around sklearn clustering models
    """
    MODEL_NAME_KEY = "model_name"
    PARAMETERS_JSON = "parameters.json"
    MODEL_FILE = "sklearn_cluster_model.joblib"

    def __init__(self, data, clustering_model, model_name, index_to_key):
        """
        :param data: array-like of data points, e.g. numpy array or gensim.KeyedVectors
        :param clustering_model: sklearn.base.ClusterMixin object
        :param model_name: str, human readable identifier for the model
        :param index_to_key: dict, int -> str, how to name each data point, important for exporting data for users and visualizations
        """
        self.data = data
        self.index_to_key = index_to_key
        self.clustering_model = clustering_model
        self.model_name = model_name
        self.clusters = None

    def train(self):
        """Fits the model to data and predicts the cluster labels for each data point.
        Returns the predicted clusters for each data point in training data
        """
        self.clusters = self.clustering_model.fit_predict(self.data)
        return self.clusters

    def predict(self, new_data):
        """Returns cluster assignments for the given data as
        :param new_data: numpy array, data to predict clusters for
        """
        return self.clustering_model.predict(new_data)

    def get_cluster_results_as_df(self, datapoint_col_name="subreddit", join_df=None):
        """Returns the cluster results as a Pandas DataFrame that can be used to easily display or plot metrics.

        :param datapoint_col_name: str, name of column that serves as key for data points
        :param join_df: Pandas DataFrame, optionally inner join this dataframe on the datapoint_col_name in the returned results
        """
        cluster_df = pd.DataFrame({datapoint_col_name: self.index_to_key,
                                   self.model_name: self.clusters})
        cluster_df[self.model_name] = cluster_df[self.model_name].astype(
            'category')
        if join_df is not None:
            cluster_df = pd.merge(cluster_df, join_df,
                                  how='inner', on=datapoint_col_name, sort=False)
        return cluster_df

    def get_metrics(self):
        """Returns Silhouette Coefficient, Caliniski-Harbasz Index and Davis-Bouldin Index for the trained clustering model on the given data as a dictionary.
        Returns an empty dictionary if the model learned only one cluster.
        """
        labels = self.clustering_model.labels_
        if len(set(labels)) > 1:
            silhouette = metrics.silhouette_score(
                self.data, labels, metric="cosine")
            ch_index = metrics.calinski_harabasz_score(
                self.data, labels)
            db_index = metrics.davies_bouldin_score(self.data, labels)
            return {'Silhouette': silhouette,
                    'Calinski-Harabasz': ch_index,
                    'Davies-Bouldin': db_index}
        else:
            return {}

    def get_parameters(self):
        """Returns the model name and salient parameters as a dictionary
        """
        param_dict = {}
        param_dict.update(self.clustering_model.get_params())
        param_dict[self.MODEL_NAME_KEY] = self.model_name
        return param_dict

    def save(self, output_dir):
        """Persists model and json parameters in the given directory
        :param output_dir: str, path to desired directory
        """
        logger.debug("Saving ClusterModel to directory: %s", output_dir)
        os.makedirs(output_dir, exist_ok=True)
        self.save_model(os.path.join(output_dir, self.MODEL_FILE))
        self.save_parameters(os.path.join(output_dir, self.PARAMETERS_JSON))
        logger.debug("All ClusterModel components saved")

    def save_model(self, model_path):
        """Writes the model to the given path
        :param model_path, str, file type, path to write Sklearn model to
        """
        joblib.dump(self.clustering_model, model_path)

    def save_parameters(self, parameters_path):
        """Saves the parameters of this model as json
        :param path: str, file type, path to write json to
        """
        with open(parameters_path, 'w') as f:
            json.dump(self.get_parameters(), f)

    def load_model(self, model_path):
        self.clustering_model = joblib.load(model_path)

    @classmethod
    def load(cls, directory, data, index_to_key):
        """Loads a ClusterModel object from a given directory, assuming the filenames are the defaults.
        Creates cluster predictions for the given data without re-fitting the clusters.

        :param directory: Path to the model directory
        :param data: array-like of data points, e.g. numpy array or gensim.KeyedVectors
        :param index_to_key: dict, int -> str, how to name each data point, important for exporting data for users and visualizations
        """
        clustermodel = cls(None, None, None, None)
        clustermodel.load_model(os.path.join(directory, cls.MODEL_FILE))
        clustermodel.index_to_key = index_to_key
        clustermodel.data = data
        clustermodel.clusters = clustermodel.clustering_model.predict(data)

        with open(os.path.join(directory, cls.PARAMETERS_JSON)) as js:
            params = json.load(js)
            clustermodel.model_name = params['model_name']
        return clustermodel


class DocumentClusteringModel(ClusteringModel):
    # TODO
    pass


class TfIdfDocumentClusters(DocumentClusteringModel):
    # TODO
    pass


class GensimLDAModel(DocumentClusteringModel):
    """Wrapper around the gensim LdaMulticore model to train on an iterable corpus or SparkRedditCorpus object
    See http://dirichlet.net/pdf/wallach09rethinking.pdf for notes on alpha and eta priors, where it was found an asymmetric prior on doc-topic dist and symmetric prior on topic-word dist performs best.
    """
    MODEL_FILE = "gensim_lda.gz"

    def __init__(self, corpus_iter, model_name, id2word, num_topics=250, alpha='asymmetric', eta='symmetric', iterations=1000, **kwargs):
        """Initializes an LDA model in gensim
        :param corpus_iter: SparkCorpusIterator, returns (int, float) for document BOW representations
        :param id2word: dict, {int -> str}, indexes the words in the vocabulary
        :param model_name: str, how to identify the model
        :param num_topics: int, number of topics to use for this model
        :param alpha: str, opinionated choice about doc-topic prior passed to Gensim LDA model
        :param eta: str, opinionated choice about topic-word prior passed to Gensim LDA model
        :param iterations: int, maximum number of iterations when infering the model
        :param kwargs: Any other LDA params that should be set, especially consider setting and workers
        """
        self.corpus_iter = corpus_iter
        self.word2id = {v: k for k, v in id2word.items()}
        self.lda_model = gm.ldamulticore.LdaMulticore(
            num_topics=num_topics, id2word=id2word,
            alpha=alpha, eta=eta, iterations=iterations,
            **kwargs)
        self.model_name = model_name
        self.coherence_model = gm.coherencemodel.CoherenceModel(
            self.lda_model, corpus=self.corpus_iter, coherence='u_mass')

    def train(self):
        """Trains LDA topic model on the corpus
        Returns topic assignments for each document in the training data
        """
        logger.debug("Staring LDA training")
        self.lda_model.update(self.corpus_iter)
        logger.debug("Finished LDA training")
        return self.get_topic_assignments()

    def predict(self, bow_docs):
        """Returns topic assignments as a numpy array of shape (len(bow_docs), num_topics) where each cell represents the probability of a document being associated with a topic

        :param bow_docs: iterable of lists of (int, float) representing docs in bag-of-words format
        """
        result = np.zeros((len(bow_docs), self.lda_model.num_topics))
        for i, bow in enumerate(bow_docs):
            indices, topic_probs = zip(
                *self.lda_model.get_document_topics(bow))
            result[i, indices] = topic_probs

        return result

    def get_topic_assignments(self, corpus_iter=None):
        """Returns {str-> list((int,float))}, the topic assignments for each document id as a list of list of (int, float) sorted in order of decreasing probability.
        :param corpus_iter: SparkCorpusIterator with is_return_id as true, if not specified, uses the training corpus
        """
        if corpus_iter is None:
            current_iterator = copy.copy(self.corpus_iter)
            current_iterator.is_return_id = True
        else:
            current_iterator = corpus_iter
        results = dict()
        for doc_id, bow_doc in current_iterator:
            results[doc_id] = sorted(self.lda_model.get_document_topics(bow_doc),
                                     key=lambda t: t[1])

        return results

    # TODO: What we actually want is average coherence, like Mallet gives
    # def get_topic_scores(self, corpus, **kwargs):
    #    """Returns a dataframe of coherence scores and other scoring metrics for the model. Rows are documents, columns are topics with coherence
    #    :param corpus: iterable of list of (int, float)
    #    """
    #    return self.lda_model.top_topics(corpus=corpus, **kwargs)

    def get_top_words(self, num_words=20):
        """Returns the top words for each learned topic as list of [(topic_id, [(word, probability)...]),...]
        :param num_words: int, How many of the top words to return for each topic
        """
        return self.lda_model.show_topics(num_topics=-1, num_words=num_words, formatted=False)

    def get_top_words_as_dataframe(self, num_words=20):
        """Returns the top words for each learned topic as a pandas dataframe
        """
        topic_ids, word_probs = zip(*self.get_top_words(num_words=num_words))
        word_strings = [" ".join([w[0] for w in topic_words])
                        for topic_words in word_probs]

        return pd.DataFrame({'topic_id': topic_ids, 'top_terms': word_strings})

    def get_cluster_results_as_df(self, corpus_iter=None, doc_col_name="id", join_df=None):
        """Returns the topic probabilities for each document in the training corpus as a pandas DataFrame

        :param corpus_iter: SparkCorpusIterator with is_return_id as true, if not specified, uses the training corpus
        :param doc_col_name: str, column name that identifies for documents
        :param join_df: Pandas DataFrame, optionally inner join this dataframe on doc_col_name the in the returned results
        """
        topic_probabilities = list()
        for doc_id, topics in self.get_topic_assignments(corpus_iter).items():
            topic_probabilities.extend([(doc_id, t[0], t[1]) for t in topics])

        topics_df = pd.DataFrame(topic_probabilities, columns=[
                                 doc_col_name, self.model_name, "probability"])

        topics_df[self.model_name] = topics_df[self.model_name].astype(
            'category')
        if join_df is not None:
            topics_df = pd.merge(topics_df, join_df,
                                 how='inner', on=doc_col_name, sort=False)

        return topics_df

    def get_metrics(self):
        """Returns LDA coherence in dictionary
        .. TODO Add exclusivity and other metrics
        """
        return {'Coherence': self.coherence_model.get_coherence()}

    def get_term_topics(self, word):
        """Returns the most relevant topics to the word as a list of (int, float) representing topic id and probability (relevence to the given word) worded by decreasing probability

        :param word: str, word of interest
        """
        if word in self.word2id:
            return sorted(self.lda_model.get_term_topics(self.word2id[word]), key=lambda x: x[1], reverse=True)
        else:
            return []

    def get_parameters(self):
        """Returns the model's paramters as a dictionary
        """
        params = {}
        params[self.MODEL_NAME_KEY] = self.model_name
        params["num_topics"] = self.lda_model.num_topics
        params["decay"] = self.lda_model.decay
        params["offset"] = self.lda_model.offset
        params["iterations"] = self.lda_model.iterations
        return params

    def save_model(self, path):
        """Save the LDA model to the path
        :param path: str or open file-like object, Path to save model to file
        """
        logger.debug("Saving Gensim LDA model to %s", path)
        self.lda_model.save(path)
        logger.debug("Gensim LDA model successfully saved")

    @classmethod
    def load(cls, load_path, corpus_iter):
        """
        :param load_path: str, directory to load LDA model files from
        :param corpus_iter: iterable over bag-of-words format documents
        """
        # Gensim
        loaded_model = cls([[(0, 1.0)]], None, {0: 'dummy'})
        loaded_model.lda_model = gm.ldamulticore.LdaMulticore.load(
            os.path.join(load_path, cls.MODEL_FILE))
        loaded_model.word2id = {v: k for k,
                                v in loaded_model.lda_model.id2word.items()}
        loaded_model.corpus_iter = corpus_iter
        loaded_model.coherence_model = gm.coherencemodel.CoherenceModel(
            loaded_model.lda_model, corpus=corpus_iter, coherence='u_mass')

        with open(os.path.join(load_path, cls.PARAMETERS_JSON)) as js:
            params = json.load(js)
            loaded_model.model_name = params['model_name']
        return loaded_model


def main(model_choice, data, index, experiment_dir, cluster_params,
         clusters_csv_filename="clusters.csv", words_csv_filename="words.csv",
         is_quiet=False):
    """Main method to train a clustering model, then save model and cluster outputs. Returns the trained model.

    :param model_choice: str, type of model to instantiate
    :param data: data used to train the model, type is dependent on the model choice
    :param index: dict, int -> str, how to name each data point, important for exporting data for users and visualizations
    :param experiment_dir: str, path to model output directory
    :param cluster_params: dict, any keyword arguments to pass along to sklearn or Gensim
    :param clusters_csv_filename: str, how to name the CSV file storing cluster output
    :param words_csv_filename: str, if the model is LDA, how to name csv filename storing topic keywords
    :param is_quiet: boolean, set to true to silence print statements for metrics
    """
    model = ClusteringModelFactory.init_clustering_model(
        model_choice, data, index, **cluster_params)
    logger.info("Training model %s", model.model_name)
    model.train()
    logger.info("Finished training model %s", model.model_name)
    logger.info("Saving model %s to %s", model.model_name, experiment_dir)
    model.save(experiment_dir)

    if not is_quiet:
        metrics = model.get_metrics()
        logger.info("Model performance metrics: %s", model.get_metrics())
        print("Performance metrics:", metrics)

    if clusters_csv_filename is not None:
        cluster_csv = os.path.join(experiment_dir, clusters_csv_filename)
        logger.info("Saving clusters to CSV %s", cluster_csv)
        model.get_cluster_results_as_df().to_csv(cluster_csv, index=False)

    if model_choice == ClusteringModelFactory.LDA and words_csv_filename is not None:
        words_csv = os.path.join(experiment_dir, words_csv_filename)
        logger.info("Saving topic keywords to CSV %s", words_csv)
        model.get_top_words_as_dataframe().to_csv(words_csv, index=False)

    return model


parser = argparse.ArgumentParser(
    description="Produce clusterings of the input data")
parser.add_argument("-q", "--quiet", action='store_true',
                    help="Use to turn off verbose info statements.")

parser.add_argument("input", nargs='+',
                    help="Path to the file containing input data in the specified format")

parser.add_argument("--data_type", '-d',
                    choices=["keyedvectors", "sparkcorpus"], default="keyedvectors")
parser.add_argument("--cluster_type", "-c",
                    choices=ClusteringModelFactory.DEFAULT_MODEL_PARAMS.keys(),
                    default=ClusteringModelFactory.KMEANS)
parser.add_argument("--cluster_params", "-p", nargs='?', type=json.loads, default="{}",
                    help="JSON defining overriding or additional parameters to pass to the sklearn or Gensim model")

parser.add_argument("--output_dir", '-o', required=True,
                    help="Directory to save the trained model and any additional data and parameters")
parser.add_argument("--min_doc_frequency", default=0.05,
                    type=float, help="Minimum document frequency")
parser.add_argument("--max_doc_frequency", type=float,
                    default=0.95, help="Maximum document frequency")
parser.add_argument("--max_time_delta", "-x", type=pytimeparse.parse,
                    help="Specify a maximum allowed time between the creation time of a submission creation and when a comment is added. Can be formatted like '1d2h30m2s' or '26:30:02'. If this is not used, all comments are kept for every submission.",
                    default="72h")
parser.add_argument("--min_time_delta", "-m", type=pytimeparse.parse,
                    help="Optionally specify a minimum allowed time between the creation time of a submission creation and when a comment is added. Can be formatted like '1d2h30m2s' or '26:30:02'. If this is not used, all comments are kept for every submission.",
                    default="3s")


if __name__ == "__main__":
    # TODO Clean this up a bit
    args = parser.parse_args()
    if args.data_type == 'keyedvectors' and args.cluster_type == ClusteringModelFactory.LDA:
        raise ValueError("LDA models do not support KeyedVector data type")
    if args.data_type == 'sparkcorpus' and args.cluster_type != ClusteringModelFactory.LDA:
        raise ValueError(
            "Document clustering with sklearn models not implemented yet")

    if args.data_type == 'keyedvectors':
        data = gm.KeyedVectors.load(args.input)
        index = data.index
    elif args.data_type == 'sparkcorpus':
        spark = ihop.utils.get_spark_session("LDA Clustering prep", args.quiet)

        vectorized_corpus, pipeline = ihop.text_preprocessing.prep_spark_corpus(
            spark, args.input,
            min_time_delta=args.min_time_delta, max_time_delta=args.max_time_delta,
            min_doc_frequency=args.min_doc_frequency, max_doc_frequency=args.max_doc_frequency,
            output_dir=args.output_dir)

        if not args.quiet:
            # TODO Compute and print document length statistics after tokenization
            pass

        data = vectorized_corpus.get_vectorized_column_iterator()
        index = pipeline.get_id_to_word()

    main(args.cluster_type, data, index, args.output_dir,
         args.cluster_params, args.quiet)
