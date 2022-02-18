"""Train clusters on community2vec embeddings and other embedding data.


.. TODO: Determine best design pattern for cluster model wrapper and update params accordingly
.. TODO: Base topic model interface/abstract class defining necessary behaviours
.. TODO: Support clustering of documents based on TF-IDF, not just c2v embeddings
.. TODO: Figure out how to get AffinityPropagation to use appropriately precomputed distances
.. TODO: Implement training of topic models on text: tf-idf-> KMeans, LDA, Hierarchical Dirichlet Processes
.. TODO: Lift document level clusters to subreddit level (will we need spark again or will pandas be sufficient?)
"""
import argparse
import logging

import gensim.models as gm
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AffinityPropagation, AgglomerativeClustering
from sklearn import metrics


logger = logging.getLogger(__name__)


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
    def train_clustering_model(cls, choice, data, index, model_name=None, **kwargs):
        """

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
            raise ValueError("Model type")


class ClusteringModel:
    """Wrapper around sklearn clustering models
    # TODO generify - allow to pass in tf-idf documents as well as
    """

    def __init__(self, data, clustering_model, model_name, index_to_key):
        """
        :param data: array-like of data points, e.g. numpy array or gensim.KeyedVectors
        :param clustering_model: sklearn.base.ClusterMixin object
        :param model_name:
        :param index_to_key: dict, int -> str, how to name each data point, important for exporting data for users and visualizations
        """
        # TODO
        self.data = data
        self.index_to_key = index_to_key
        self.clustering_model = clustering_model
        self.model_name = model_name
        self.clusters = self.clustering_model.fit_predict(data)

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
                self.embeddings, labels, metric="cosine")
            ch_index = metrics.calinski_harabasz_scores(
                self.embeddings, labels)
            db_index = metrics.davies_bouldin_score(self.embeddings, labels)
            return {'Silhouette': silhouette,
                    'Calinski Harabasz': ch_index,
                    'Davies Bouldin': db_index}
        else:
            return {}


class DocumentClusteringModel(ClusteringModel):
    # TODO
    pass


class TfIdfDocumentClusters(DocumentClusteringModel):
    # TODO
    pass


class GensimLDAModel(DocumentClusteringModel):
    """Wrapper around the gensim LdaMulticore model to train on SparkRedditCorpus object
    See http://dirichlet.net/pdf/wallach09rethinking.pdf for notes on alpha and eta priors, where it was found an asymmetric prior on doc-topic dist and symmetric prior on topic-word dist performs best.
    """

    def __init__(self, corpus, model_name, id2word, num_topics=250, alpha='asymmetric', eta='symmetric', iterations=1000, **kwargs):
        """Initializes an LDA model in gensim
        :param corpus: iterable of list of (int, float)
        :param id2word: dict, {int -> str}, indexes the words in the vocabulary
        :param model_name: str, how to identify the model
        :param num_topics: int, number of topics to use for this model
        :param alpha: str, opinionated choice about doc-topic prior passed to Gensim LDA model
        :param eta: str, opinionated choice about topic-word prior passed to Gensim LDA model
        :param iterations: int, maximum number of iterations when infering the model
        :param kwargs: Any other LDA params that should be set, especially consider setting and workers
        """
        self.lda_model = gm.ldamulticore.LdaMulticore(
            num_topics=num_topics, id2word=id2word,
            alpha=alpha, eta=eta, iterations=iterations,
            **kwargs)
        self.lda_model.update(corpus)

    def get_topic_scores(self, corpus, **kwargs):
        """Returns a dataframe of coherence scores and other scoring metrics for the model. Rows are documents, columns are topics with coherece
        """
        return self.top_topics(corpus=corpus, **kwargs)

    def get_top_words(self, num_words=20):
        """Returns the top words for each learned topic
        """
        return self.lda_model.show_topics(num_topics=self.lda_model.num_topics, num_words=num_words)

    def get_top_words_as_dataframe(self):
        """Returns the top words for each learned topic as a pandas dataframe
        """
        return pd.DataFrame.from_records(self.get_top_words(), columns=["topic_id", "top_terms"])

    def get_cluster_results_as_df(self, vocab_col_name="subreddit", join_df=None):
        """
        """
        # TODO
        pass

    def get_metrics(self):
        """TODO: Override superclass with coherence & exclusivity scores
        """
        pass

    def save(self, savepath):
        """Save the LDA model to the path
        :param savepath: str or open file-like object, Path to save model to file
        """
        self.lda_model(savepath)

    @ classmethod
    def load(cls, load_path):
        """TODO
        """
        loaded_model = cls({})
        loaded_model.lda_model = gm.ldamulticore.LdaMulticore.load(load_path)
        return loaded_model
