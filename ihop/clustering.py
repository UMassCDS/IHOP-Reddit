"""Train clusters on community2vec embeddings and other embedding data.

.. TODO: Implement main method and argparse for training document clustering models/topics models from a joined dataframe or clusters from embeddings using a script
.. TODO: Base topic model interface/abstract class defining necessary behaviours
.. TODO: Support clustering of documents based on TF-IDF, not just c2v embeddings
.. TODO: Implement training of topic models on text: tf-idf-> KMeans, Hierarchical Dirichlet Processes
.. TODO: Lift document level clusters to subreddit level (will we need spark again or will pandas be sufficient?)
.. TODO: AuthorTopic models with subreddits as the metadata field (instead of author)
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


logger = logging.getLogger(__name__)
# TODO Logging should be configurable, but for now just turn it on for Gensim
logging.basicConfig(
    format='%(name)s : %(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class ClusteringModelFactory:
    """Return appropriate class given input params
    """
    AFFINITY_PROP = "Affinity Propagation"
    AGGLOMERATIVE = "Agglomerative Clustering"
    KMEANS = "Kmeans"
    LDA = "LDA"

    # Default parameters used when instantiating the model
    DEFAULT_MODEL_PARAMS = {
        AFFINITY_PROP: {'affinity': 'precomputed', 'max_iter': 1000, 'convergence_iter': 50, 'random_state': 100},
        AGGLOMERATIVE: {'n_clusters': 250, 'linkage': 'average', 'affinity': 'cosine', 'compute_distances': True},
        KMEANS: {'n_clusters': 250, 'random_state': 100},
        LDA: {'num_topics': 250, 'alpha': 'asymmetric',
              'eta': 'symmetric', 'iterations': 1000}
    }

    @classmethod
    def init_clustering_model(cls, choice, data, index, model_name=None, **kwargs):
        """Returns a ClusteringModel instance instantiated with the appropirate parameters and ready to train on the given data.
        """
        if model_name is None:
            model_id = choice
        else:
            model_id = model_name

        if choice not in cls.DEFAULT_MODEL_PARAMS:
            raise ValueError(f"Model choice {choice} is not supported")

        parameters = {}
        parameters.update(cls.DEFAULT_MODEL_PARAMS[choice])
        parameters.update(kwargs)

        if choice == cls.KMEANS:
            return ClusteringModel(data, KMeans(**parameters), model_id, index)
        elif choice == cls.AFFINITY_PROP:
            if isinstance(data, gm.keyedvectors.KeyedVectors) and parameters['affinity'] == "precomputed":
                precomputed_distances = np.zeros((len(index), len(index)))
                for i, v in index.items():
                    precomputed_distances[i] = np.array(data.distances(v))

                return ClusteringModel(precomputed_distances, AffinityPropagation(**parameters), choice, index)
            else:
                return ClusteringModel(data, AffinityPropagation(**parameters), model_id, index)
        elif choice == cls.AGGLOMERATIVE:
            return ClusteringModel(data, AgglomerativeClustering(**parameters), model_id, index)
        elif choice == cls.LDA:
            return GensimLDAModel(data, model_id, index, **parameters)
        else:
            raise ValueError(f"Model type '{choice}' is not supported")


class ClusteringModel:
    """Wrapper around sklearn clustering models

    """
    MODEL_NAME_KEY = "model_name"
    PARAMETERS_JSON = "parameters.json"
    INDEX_JSON = "index.json"
    MODEL_FILE = "sklearn_cluster_model.joblib"

    def __init__(self, data, clustering_model, model_name, index_to_key):
        """
        :param data: array-like of data points, e.g. numpy array or gensim.KeyedVectors
        :param clustering_model: sklearn.base.ClusterMixin object
        :param model_name:
        :param index_to_key: dict, int -> str, how to name each data point, important for exporting data for users and visualizations
        """
        self.data = data
        self.index_to_key = index_to_key
        self.clustering_model = clustering_model
        self.model_name = model_name
        self.clusters = None

    def train(self):
        """Fits the model to data and predicts the cluster labels for each data point.
        Returns the predicted clusters for each data point.
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

        :param datapoint_col_name: How to identify the data points column
        :param join_df: Pandas DataFrame, optionally inner join this dataframe in the returned results
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
        os.makedirs(output_dir, exist_ok=True)
        self.save_model(os.path.join(output_dir, self.MODEL_FILE))
        self.save_parameters(os.path.join(output_dir, self.PARAMETERS_JSON))
        self.save_index(os.path.join(output_dir, self.INDEX_JSON))

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

    def save_index(self, index_path):
        """Saves the index to file as json dictionary
        """
        with open(index_path, 'w') as f:
            json.dump(self.index_to_key, f)

    def load_model(self, model_path):
        self.clustering_model = joblib.load(model_path)

    def load_index(self, index_path):
        with open(index_path, 'r') as f:
            index = json.load(f)
            self.index_to_key = {}
            for k, v in index.items():
                self.index_to_key[int(k)] = v

    @classmethod
    def load(cls, directory):
        """Loads a ClusterModel object from a given directory, assuming the filenames are the defaults
        """
        clustermodel = cls(None, None, None, None)
        clustermodel.load_model(os.path.join(directory, cls.MODEL_FILE))
        clustermodel.load_index(os.path.join(directory, cls.INDEX_JSON))

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
        self.index = id2word
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
        """
        self.lda_model.update(self.corpus_iter)

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
        return self.lda_model.show_topics(num_topics=self.lda_model.num_topics, num_words=num_words, formatted=False)

    def get_top_words_as_dataframe(self, num_words=20):
        """Returns the top words for each learned topic as a pandas dataframe
        """
        topic_ids, word_probs = zip(*self.get_top_words())
        word_strings = [" ".join([w[0] for w in topic_words])
                        for topic_words in word_probs]

        return pd.DataFrame({'topic_id': topic_ids, 'top_terms': word_strings})

    def get_topic_assigments(self, corpus):
        """Returns the topic assignments for each document as a list of list of (int, float) sorted in order of decreasing probability
        :param corpus: SparkCorpusIterator with is_return_id as true
        """
        results = dict()
        for doc_id, bow_doc in corpus:
            results[doc_id] = sorted(self.lda_model.get_document_topics(bow_doc),
                                     key=lambda t: t[1])

        return results

    def get_cluster_results_as_df(self, vocab_col_name="document_id", join_df=None):
        """Returns the most likely topic for each document in the corpus as a pandas DataFrame
        """
        corpus_iterator = copy.copy(self.corpus_iter)
        corpus_iterator.is_return_id = True
        topic_assigments = [(doc_id, topics[0][0]) for doc_id, topics in self.get_topic_assigments(
            corpus_iterator).items()]

        topics_df = pd.DataFrame(topic_assigments, columns=[
                                 vocab_col_name, self.model_name])

        topics_df[self.model_name] = topics_df[self.model_name].astype(
            'category')
        if join_df is not None:
            topics_df = pd.merge(topics_df, join_df,
                                 how='inner', on=vocab_col_name, sort=False)

        return topics_df

    def get_metrics(self):
        """Returns LDA coherence in dictionary
        .. TODO Add exclusivity and other metrics
        """
        return {'Coherence': self.coherence_model.get_coherence()}

    def get_term_topics(self, word):
        """Returns the most relevant topics to the word as a list of (int, float) representing topic id and probability (relevence to the given word)

        :param word: str, word of interest
        """
        if word in self.word2id:
            return self.lda_model.get_term_topics(self.word2id[word])
        else:
            return []

    def get_parameters(self):
        """Returns the model's paramters as a dictionary
        """
        params = {}
        params[self.MODEL_NAME_KEY] = self.model_name
        params["num_topics"] = self.lda_model.num_topics
        params["alpha"] = list(self.lda_model.alpha)
        params["eta"] = list(self.lda_model.eta)
        params["decay"] = self.lda_model.decay
        params["offset"] = self.lda_model.offset
        params["iterations"] = self.lda_model.iterations
        params["random_state_seed"] = self.lda_model.random_state.seed
        return params

    def save_model(self, path):
        """Save the LDA model to the path
        :param path: str or open file-like object, Path to save model to file
        """
        self.lda_model.save(path)

    @ classmethod
    def load(cls, load_path):
        """TODO
        """
        loaded_model = cls({})
        loaded_model.lda_model = gm.ldamulticore.LdaMulticore.load(load_path)
        loaded_model.index = loaded_model.lda_model.id2word
        return loaded_model


# TODO Main method
# TODO Finish all argparse options for scripts
parser = argparse.ArgumentParser(
    description="Pre-process text and train document-based topic or cluster models from Reddit threads")
parser.add_argument("input", nargs='+',
                    help="Path to the dataset output by 'ihop.import_data bow'")
parser.add_argument("--model_dir", required=True,
                    help="Path to serialize the trained model to")
parser.add_argument("--min_term_frequency", default=0,
                    help="Minimum term frequency for terms in each document")
parser.add_argument("--min_doc_frequency", default=0.05,
                    type=float, help="Minimum document frequency")
parser.add_argument("--max_doc_frequency", type=float,
                    default=0.90, help="Maximum document frequency")
parser.add_argument("--max_time_delta", "-x", type=pytimeparse.parse,
                    help="Specify a maximum allowed time between the creation time of a submission creation and when a comment is added. Can be formatted like '1d2h30m2s' or '26:30:02'. If this is not used, all comments are kept for every submission.")
parser.add_argument("--min_time_delta", "-m", type=pytimeparse.parse,
                    help="Optionally specify a minimum allowed time between the creation time of a submission creation and when a comment is added. Can be formatted like '1d2h30m2s' or '26:30:02'. If this is not used, all comments are kept for every submission.")
