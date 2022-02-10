"""Train clusters on community2vec embeddings and other embedding data.


.. TODO: Determine best design pattern for cluster model wrapper and update params accordingly
.. TODO: Support clustering of documents based on TF-IDF, not just c2v embeddings
.. TODO: Figure out how to get AffinityPropagation to use appropriately precomputed distances
"""
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans, AffinityPropagation, AgglomerativeClustering
from sklearn import metrics

class ClusteringModelFactory:
    """Return appropriate
    """
    KMEANS = "kmeans"
    AFFINITY_PROP = "affinity propagation"
    AGGLOMERATIVE = "agglomerative clustering"
    VALID_CLUSTERING_ALGORITHMS = [KMEANS, AFFINITY_PROP,
                                   ]

class ClusteringModel:
    """Wrapper around sklearn clustering models
    """

    def __init__(self, embeddings, clustering_model):
        """
        :param embeddings: Gensim KeyedVectors object
        """
        # TODO
        self.embeddings = embeddings
        self.clustering_model = clustering_model
        self.clusters = self.clustering_model.fit_predict(embeddings)


    def get_cluster_results_as_df(self, vocab_col_name="subreddit", join_df=None):
        """Returns the cluster results as a Pandas DataFrame that can be used to easily display or plot metrics.
        :param vocab_col_name: How to identify the data points
        :param join_df: Pandas DataFrame, optionally inner join this dataframe in the returned results
        """
        # TODO Choose how to assign cluster column name
        vocab_list = self.embeddings.index_to_key
        cluster_df = pd.DataFrame(
            {vocab_col_name:vocab_list,
             cluster_col_name:self.clusters}
            )
        cluster_df[cluster_col_name] = cluster_df[cluster_col_name].astype('category')
        if join_df is not None:
            cluster_df = pd.merge(cluster_df, join_df, how='inner', on = vocab_col_name, sort=False)
        return cluster_df


    def get_metrics(self):
        """Returns Silhouette Coefficient, Caliniski-Harbasz Index and Davis-Bouldin Index for the trained clustering model on the given embeddings.
        Returns None for all metrics if the model learned only one cluster.
        """
        labels = self.clustering_model.labels_
        if len(set(labels)) > 1:
            silhouette = metrics.silhouette_score(self.embeddings, labels, metric="cosine")
            ch_index = metrics.calinski_harabasz_scores(self.embeddings, labels)
            db_index = metrics.davies_bouldin_score(self.embeddings, labels)
            return silhouette, ch_index, db_index
        else:
            return None, None, None
