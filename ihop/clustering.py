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

from gensim.models.ldamulticore import LdaMulticore
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans, AffinityPropagation, AgglomerativeClustering
from sklearn import metrics


logger = logging.getLogger(__name__)


class ClusteringModelFactory:
    """Return appropriate class given input params

    # TODO
    """
    KMEANS = "kmeans"
    AFFINITY_PROP = "affinity propagation"
    AGGLOMERATIVE = "agglomerative clustering"
    VALID_CLUSTERING_ALGORITHMS = [KMEANS, AFFINITY_PROP, AGGLOMERATIVE]


class ClusteringModel:
    """Wrapper around sklearn clustering models
    # TODO generify - allow to pass in tf-idf documents as well as
    """

    def __init__(self, embeddings, clustering_model, model_name):
        """
        :param embeddings: Gensim KeyedVectors object
        """
        # TODO
        self.embeddings = embeddings
        self.clustering_model = clustering_model
        self.model_name = model_name
        self.clusters = self.clustering_model.fit_predict(embeddings)

    def get_cluster_results_as_df(self, vocab_col_name="subreddit", join_df=None):
        """Returns the cluster results as a Pandas DataFrame that can be used to easily display or plot metrics.
        :param vocab_col_name: How to identify the data points
        :param join_df: Pandas DataFrame, optionally inner join this dataframe in the returned results
        """
        # TODO Choose how to assign cluster column name
        vocab_list = self.embeddings.index_to_key
        cluster_df = pd.DataFrame({vocab_col_name: vocab_list,
                                  self.model_name: self.clusters})
        cluster_df[self.model_name] = cluster_df[self.model_name].astype(
            'category')
        if join_df is not None:
            cluster_df = pd.merge(cluster_df, join_df,
                                  how='inner', on=vocab_col_name, sort=False)
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

    def __init__(self, id2word, num_topics=100, alpha='asymmetric', eta='symmetric', iterations=1000, **kwargs):
        """Initializes an LDA model in gensim
        :param id2word: dict, {int -> str}, indexes the words in the vocabulary
        :param num_topics: int, number of topics to use for this model
        :param alpha: str, opinionated choice about doc-topic prior passed to Gensim LDA model
        :param eta: str, opinionated choice about topic-word prior passed to Gensim LDA model
        :param iterations: int, maximum number of iterations when infering the model
        :param kwargs: Any other LDA params that should be set, especially consider setting and workers
        """
        self.num_topics = num_topics
        self.iterations = iterations
        self.lda_model = LdaMulticore(
            num_topics=num_topics, id2word=id2word, alpha=alpha, eta=eta, **kwargs)

    def train(self, corpus):
        """
        :param corpus: iterable of list of (int, float)
        """
        self.lda_model.update(corpus, iterations=self.iterations)

    def get_topic_scores(self, corpus):
        """Returns a dataframe of coherence scores and other scoring metrics for the model
        """
        # TODO
        pass

    def get_top_words(self):
        """Returns the top words for each learned topic
        """
        # TODO
        pass

    def get_top_words_as_dataframe(self):
        """Returns the top words for each learned topic as a pandas dataframe
        """

    def save(self, path):
        # TODO
        pass

    @ classmethod
    def load(cls, load_path):
        """TODO
        """
        pass
