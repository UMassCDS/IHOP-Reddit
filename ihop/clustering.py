"""Train clusters on community2vec embeddings and other embedding data.

A list of ideas for clustering based on text (not users):
.. TODO: Support clustering of documents based on TF-IDF, not just c2v embeddings
.. TODO: Implement training of topic models on text: tf-idf-> KMeans, Hierarchical Dirichlet Processes
.. TODO: Gensim Coherence model supported with Spark LDA implementation
.. TODO: Lift document level clusters to subreddit level (will we need spark again or will pandas be sufficient?)
.. TODO: AuthorTopic models with subreddits as the metadata field (instead of author)
"""
import argparse
import json
import logging
import os
import pathlib
import pickle

import gensim.models as gm
import gensim.corpora as gc
import joblib
import numpy as np
import pandas as pd
import pyspark.ml.clustering as sparkmc
import pyspark.sql.functions as fn
import pyspark.sql.types as sparktypes
import pytimeparse
from scipy.stats import entropy
from sklearn.cluster import KMeans, AffinityPropagation, AgglomerativeClustering
from sklearn import metrics

import ihop.utils
import ihop.text_processing

logger = logging.getLogger(__name__)

# Constants for supported data types
KEYED_VECTORS = "KeyedVectors"
SPARK_DOCS = "SparkDocuments"
SPARK_VEC = "SparkVectorized"

# Constant to use for an additional cluster assignment for when
# a datapoint is missing from one clustering
MISSING_CLUSTER_ASSIGNMENT = -1

# Use to idenfity different was of comparing clusterings
UNION_UNIFORM = "union_uniform_probability"
INTERSECT_UNIFORM = "intersection_uniform_probability"
INTERSECT_COMMENT_PROB = "intersection_comment_probability"

VOI = "variation_of_information"
COMPLETENESS = "completeness"
HOMOGENEITY = "homogeneity"
V_MEASURE = "v_measure"
ADJUSTED_RAND_INDEX = "adjusted_rand_index"
RAND_INDEX = "rand_index"
NORM_MUTUAL_INFO = "normalized_mutual_info"


def get_probabilities(counts_dict, keys_to_keep, default_count_value=0):
    """Returns the probabilities for a list of datapoints keys given
    from their counts in a dictionary, accounting for keys that should
    be left out or are missing.

    :param counts_dict: dict, maps key to integer
    :param keys_to_keep: list of keys
    :param default_count_value: int, value to use for missing keys, defaults to 0
    :return: array of floats corresponding to keys_to_keep datapoints
    """
    all_counts = np.full(len(keys_to_keep), default_count_value)
    for i, k in enumerate(keys_to_keep):
        if k in counts_dict:
            all_counts[i] = counts_dict[k]
    return all_counts / np.sum(all_counts)


def get_cluster_probabilities(cluster_assignments, datapoint_counts, cluster_indexes):
    """Return the probability for each cluster based on the probabilities for each data point assigned to the clusters

    :param cluster_assignments: the cluster assignment for each datapoint
    :param datapoint_counts: array, store frequency counts for each datapoint
    :param cluster_indexes: list or array, used to track which cluster is stored at the index in the array
    """
    total_counts = np.sum(datapoint_counts)
    cluster_probs = np.zeros((len(cluster_indexes)))
    for i, c in enumerate(cluster_indexes):
        cluster_probs[i] = np.sum(datapoint_counts[np.where(cluster_assignments == c)])

    return cluster_probs / total_counts


def get_contingency_table(
    cluster_1_assignments,
    cluster_2_assignments,
    cluster_1_counts,
    cluster_2_counts,
    cluster_1_indices,
    cluster_2_indices,
):
    """Returns the frequency distributions of datapoints between two clusterings as numpy matrix
    Clustering 1 is the first axis, clustering 2 is the second axis.

    :param cluster_1_assignments: list or array storing cluster assignment for each datapoint in clustering 1
    :param cluster_2_assignments: list or array storing cluster assignment for each datapoint in clustering 2
    :param cluster_1_counts: list or array of int, same length as cluster_1_assignments, frequency counts of each datapoint in cluster assignment 1
    :param cluster_2_counts:  list or array of int, same length as cluster_2_assignments, frequency counts of each datapoint in cluster assignment 2
    :param cluster_1_indices: list or array index pointer that identifies the position index of each cluster in clustering 1
    :param cluster_2_indices: list or array index pointer that identifies the position index of each cluster in clustering 2
    """
    contingency_table = np.zeros((len(cluster_1_indices), len(cluster_2_indices)))
    for i, c1 in enumerate(cluster_1_assignments):
        c2 = cluster_2_assignments[i]
        c1_index = cluster_1_indices.index(c1)
        c2_index = cluster_2_indices.index(c2)
        contingency_table[c1_index, c2_index] += (
            cluster_1_counts[i] + cluster_2_counts[i]
        )

    return contingency_table


def get_mutual_information(contingency_table, cluster_1_probs, cluster_2_probs):
    """Returns the mutual information between clusterings 1 and 2 as a float
    based on the contingency table used to calculate joint distribution and the probability distribution of individual clusterings

    :param contingency_table: Frequency counts of cluster assignments comparison between both clustering 1 on first axis and clustering 2 on second axis
    :param cluster_1_probs: np array, probability of cluster assignments in clustering 1
    :param cluster_2_probs: np array, probability of cluster assignments in clustering 2
    """
    probs_products = np.outer(cluster_1_probs, cluster_2_probs)
    total_freqs = np.sum(contingency_table)
    joint_probs = contingency_table / total_freqs
    # Can safely ignore divide by zero in log2 warnings, they aren't included in the final sum
    with np.errstate(divide="ignore", invalid="ignore"):
        mi_components = joint_probs * (np.log2(joint_probs / probs_products))
    mi = np.sum(mi_components[np.where(mi_components > 0)])
    return mi


def remap_clusters(
    cluster_mapping_1,
    cluster_mapping_2,
    use_union=False,
    missing_cluster_value=MISSING_CLUSTER_ASSIGNMENT,
):
    """Remaps clusterings so that they are partitions of the same data, returning cluster assignments as two arrays.
    Also returns the data point keys as a third array for indexing.
    Uses the intersection of data points by default.

    :param cluster_mapping_1: dict, maps a data point to its cluster assignment for the first clustering
    :param cluster_mapping_2: dict, maps a data point to its cluster assignment for the second clustering
    :param use_union: boolean, set to True to use union of data points by having an additional cluster that consists of those values in only one cluster, defaults to False using the intersection of datapoints
    """
    if use_union:
        all_datapoints = cluster_mapping_1.keys() | cluster_mapping_2.keys()
        logger.info("Computed cluster partitions using union.")
    else:
        all_datapoints = cluster_mapping_1.keys() & cluster_mapping_2.keys()
        logger.info("Computed cluster partitions using intersection.")
    all_datapoints = np.array(sorted(all_datapoints))
    logger.info("Number of datapoints: %s", len(all_datapoints))
    cluster_assignments_1 = list()
    cluster_assignments_2 = list()
    for d in all_datapoints:
        cluster_assignments_1.append(cluster_mapping_1.get(d, missing_cluster_value))
        cluster_assignments_2.append(cluster_mapping_2.get(d, missing_cluster_value))

    return (
        np.array(cluster_assignments_1),
        np.array(cluster_assignments_2),
        all_datapoints,
    )


def compare_cluterings(
    cluster_mapping_1,
    cluster_mapping_2,
    use_union=False,
    cluster_1_counts=None,
    cluster_2_counts=None,
    missing_cluster_assignment=MISSING_CLUSTER_ASSIGNMENT,
):
    """Returns comparison metrics between two clusterings. Results formated as nested dictionary where the outer key indicates the comparison style:
    1) whether intersection or union is used
    2) if uniform probabilities for each data point to cluster (subreddit) or probability counts of data point to cluster (subreddit, probability determined by number of comments over the time period)
    Returned dictionary like {comparison style: {metric name: metric value}}

    :param cluster_mapping_1: dict, maps a data point to its cluster assignment for the first clustering
    :param cluster_mapping_2: dict, maps a data point to its cluster assignment for the second clustering
    :param use_union: boolean, set to True to use union of data points by having an additional cluster that consists of those values in only one cluster, defaults to False using the intersection of datapoints
    :param cluster_1_counts: dict, maps a datapoint to an integer value, used to compute probabilities
    :param cluster_2_counts: dict, maps a datapoint to an integer, used to compute probabilities
    :param missing_cluster_assignment: constant value to assign clusters when using the union of the partition. User is responsible for ensuring this value doesn't conflict with any actual cluster ids.
    """
    cluster_assignment_1, cluster_assignment_2, datapoint_keys = remap_clusters(
        cluster_mapping_1,
        cluster_mapping_2,
        use_union=use_union,
        missing_cluster_value=missing_cluster_assignment,
    )
    results_key = INTERSECT_UNIFORM
    results_dict = {}
    if use_union:
        results_key = UNION_UNIFORM

    # probabilities can only be used with intersection
    if not use_union and cluster_1_counts is not None and cluster_2_counts is not None:
        results_key = INTERSECT_COMMENT_PROB
        # Order the counts in the same way as cluster assignments
        cluster_1_counts_reordered = []
        cluster_2_counts_reordered = []
        for d in datapoint_keys:
            cluster_1_counts_reordered.append(cluster_1_counts[d])
            cluster_2_counts_reordered.append(cluster_2_counts[d])

        results_dict[VOI] = variation_of_information(
            cluster_assignment_1,
            cluster_assignment_2,
            np.array(cluster_1_counts_reordered),
            np.array(cluster_2_counts_reordered),
        )
    else:

        results_dict[ADJUSTED_RAND_INDEX] = metrics.adjusted_rand_score(
            cluster_assignment_1, cluster_assignment_2
        )
        results_dict[RAND_INDEX] = metrics.rand_score(
            cluster_assignment_1, cluster_assignment_2
        )
        results_dict[NORM_MUTUAL_INFO] = metrics.normalized_mutual_info_score(
            cluster_assignment_1, cluster_assignment_2
        )

        h, c, v = metrics.homogeneity_completeness_v_measure(
            cluster_assignment_1, cluster_assignment_2
        )
        results_dict[HOMOGENEITY] = h
        results_dict[COMPLETENESS] = c
        results_dict[V_MEASURE] = v

        results_dict[VOI] = variation_of_information(
            cluster_assignment_1, cluster_assignment_2
        )

    return {f"{results_key}_{m}": v for m, v in results_dict.items()}


def variation_of_information(
    cluster_assignment_1,
    cluster_assignment_2,
    cluster_1_counts=None,
    cluster_2_counts=None,
):
    """Computes variation of information between two partitions of the same data points.

    Meilă, Marina. “Comparing Clusterings by the Variation of Information.” COLT (2003).

    :param cluster_assignment_1: array type, the cluster assignments for each data point under the first partitioning
    :param cluster_assignment_2: array type, the cluster assignments for each data point under the second partitioning
    :param cluster_1_counts: array type, counts of occurences of a particular cluster under the first partitioning, used to calculate probabilities for entropy and mutual information. If this is not given a uniform probability of all clusters will be used.
    :param cluster_2_counts: array type, counts of occurences of a particular cluster under the second partitioning, used to calculate probabilities for entropy and mutual information. If this is not given a uniform probability of all clusters will be used.
    :return: float, the computed variation of information value
    """
    if len(cluster_assignment_1) != len(cluster_assignment_2):
        msg = f"Clusterings must have the same number of data points in order to compare. Clustering 1: {len(cluster_assignment_1)}, Clustering 2: {len(cluster_assignment_2)}"
        logger.error(msg)
        raise ValueError(msg)

    # If no counts are given, a uniform probability is used
    if cluster_1_counts is None and cluster_2_counts is None:
        cluster_1_counts = np.ones(cluster_assignment_1.shape)
        cluster_2_counts = np.ones(cluster_assignment_2.shape)
    elif not (cluster_1_counts is not None and cluster_2_counts is not None):
        msg = "Choose either uniform probability or count based probabilities for cluster comparison, do not mix."
        logger.error(msg)

    cluster_1_indices = sorted(set(cluster_assignment_1))
    cluster_1_probs = get_cluster_probabilities(
        cluster_assignment_1, cluster_1_counts, cluster_1_indices
    )

    cluster_2_indices = sorted(set(cluster_assignment_2))
    cluster_2_probs = get_cluster_probabilities(
        cluster_assignment_2, cluster_2_counts, cluster_2_indices
    )

    clustering_1_entropy = entropy(cluster_1_probs, base=2)
    clustering_2_entropy = entropy(cluster_2_probs, base=2)

    contingency_table = get_contingency_table(
        cluster_assignment_1,
        cluster_assignment_2,
        cluster_1_counts,
        cluster_2_counts,
        cluster_1_indices,
        cluster_2_indices,
    )

    mi = get_mutual_information(contingency_table, cluster_1_probs, cluster_2_probs)

    voi = clustering_1_entropy + clustering_2_entropy - 2 * mi
    return voi


class ClusteringModelFactory:
    """Return appropriate class given input params"""

    AFFINITY_PROP = "affinity"
    AGGLOMERATIVE = "agglomerative"
    KMEANS = "kmeans"
    GENSIM_LDA = "gensimlda"
    SPARK_LDA = "sparklda"

    # Default parameters used when instantiating the model
    DEFAULT_MODEL_PARAMS = {
        AFFINITY_PROP: {
            "affinity": "precomputed",
            "max_iter": 1000,
            "convergence_iter": 50,
            "random_state": 100,
        },
        AGGLOMERATIVE: {
            "n_clusters": 250,
            "linkage": "average",
            "affinity": "cosine",
            "compute_distances": True,
        },
        KMEANS: {"n_clusters": 250, "random_state": 100},
        GENSIM_LDA: {
            "num_topics": 250,
            "alpha": "asymmetric",
            "eta": "symmetric",
            "iterations": 100,
        },
        SPARK_LDA: {
            "num_topics": 250,
            "maxIter": 50,
            "optimizer": "online",
            "use_asymmetric_alpha": True,
            "subsamplingRate": 0.05,
        },
    }

    @classmethod
    def init_clustering_model(
        cls, model_choice, data, index, model_name=None, **kwargs
    ):
        """Returns a ClusteringModel instance instantiated with the appropriate parameters and ready to train on the given data.
        :param model_choice: str, type of model to instantiate
        :param data: data used to train the model, type is dependent on the model choice. For sklearn models, should be gensim KeyedVectors and for LDA can be SparkCorpusIterator or some other kind of iterable data
        :param index: dict, int -> str, how to name each data point, important for exporting data for users and visualizations
        :param model_name: str, used to identify the model in output and string representation. If left as None, then choice value will be used as model name
        :param kwargs: parameters to pass to the sklearn or Gensim model
        """
        if model_name is None:
            model_id = model_choice
        else:
            model_id = model_name

        logger.info("Instantiating %s model with name '%s'", model_choice, model_id)

        if model_choice not in cls.DEFAULT_MODEL_PARAMS:
            raise ValueError(f"Model choice {model_choice} is not supported")

        parameters = {}
        parameters.update(cls.DEFAULT_MODEL_PARAMS[model_choice])
        parameters.update(kwargs)

        logger.info("Specified model parameters: %s", parameters)

        if isinstance(data, gm.keyedvectors.KeyedVectors):
            if parameters.get("affinity", None) == "precomputed":
                logger.debug(
                    "Determining precomputed distances as vector input to model"
                )
                vectors = np.zeros((len(index), len(index)))
                for i, v in index.items():
                    vectors[i] = np.array(data.distances(v))
            else:
                logger.debug("Getting normed vectors as vector input to model")
                vectors = data.get_normed_vectors()
        else:
            vectors = data

        if model_choice == cls.GENSIM_LDA:
            return GensimLDAModel(vectors, model_id, index, **parameters)
        elif model_choice == cls.SPARK_LDA:
            return SparkLDAModel(vectors, model_id, index, **parameters)
        elif model_choice == cls.KMEANS:
            model = KMeans(**parameters)
        elif model_choice == cls.AFFINITY_PROP:
            model = AffinityPropagation(**parameters)
        elif model_choice == cls.AGGLOMERATIVE:
            model = AgglomerativeClustering(**parameters)
        else:
            raise ValueError(f"Model type '{model_choice}' is not supported")

        logger.info("Finished instantiating model")
        return ClusteringModel(vectors, model, model_id, index)


class ClusteringModel:
    """Wrapper around sklearn clustering models. Clusters arbitrary data points"""

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
        logger.info("Fitting ClusteringModel")
        self.clusters = self.clustering_model.fit_predict(self.data)
        logger.info("Finished fitting ClusteringModel")

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
        datapoints = [
            (val, self.clusters[idx]) for idx, val in self.index_to_key.items()
        ]
        cluster_df = pd.DataFrame(
            datapoints, columns=[datapoint_col_name, self.model_name]
        )
        cluster_df[self.model_name] = cluster_df[self.model_name].astype("category")
        if join_df is not None:
            logger.debug(
                "Joining cluster results with input dataframe on '%s'",
                datapoint_col_name,
            )
            cluster_df = pd.merge(
                cluster_df, join_df, how="inner", on=datapoint_col_name, sort=False
            )
        return cluster_df

    def get_metrics(self):
        """Returns Silhouette Coefficient, Caliniski-Harbasz Index and Davis-Bouldin Index for the trained clustering model on the given data as a dictionary.
        Returns an empty dictionary if the model learned only one cluster.
        """
        labels = self.clustering_model.labels_
        if len(set(labels)) > 1:
            silhouette = metrics.silhouette_score(self.data, labels, metric="cosine")
            ch_index = metrics.calinski_harabasz_score(self.data, labels)
            db_index = metrics.davies_bouldin_score(self.data, labels)
            return {
                "Silhouette": silhouette,
                "Calinski-Harabasz": ch_index,
                "Davies-Bouldin": db_index,
            }
        else:
            return {}

    def get_parameters(self):
        """Returns the model name and salient parameters as a dictionary"""
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
        :param parameters_path: str, file type, path to write json to
        """
        with open(parameters_path, "w") as f:
            json.dump(self.get_parameters(), f)

    def load_model(self, model_path):
        """Loads the sklearn model to the specified path

        :param model_path: str or path, joblib file
        """
        self.clustering_model = joblib.load(model_path)

    @classmethod
    def load_model_name(cls, param_json_path):
        """Returns a model name stored in params json

        :param param_json_path: str, path to json file
        """
        return cls.read_model_parameters(param_json_path)[cls.MODEL_NAME_KEY]

    @classmethod
    def read_model_parameters(cls, param_json_path):
        """Returns parameters from json as a dictionary

        :param param_json_path: str or path, path to json file
        """
        with open(param_json_path) as js:
            return json.load(js)

    @classmethod
    def get_model_path(cls, directory):
        """Returns the full path for a model file in the given directory"""
        return os.path.join(directory, cls.MODEL_FILE)

    @classmethod
    def get_param_json_path(cls, directory):
        """Returns the full path for parameters json file in the given directory"""
        return os.path.join(directory, cls.PARAMETERS_JSON)

    @classmethod
    def load(cls, directory, data, index_to_key):
        """Loads a ClusterModel object from a given directory, assuming the filenames are the defaults.
        Creates cluster predictions for the given data without re-fitting the clusters.

        :param directory: Path to the model directory
        :param data: array-like of data points, e.g. numpy array or gensim.KeyedVectors
        :param index_to_key: dict, int -> str, how to name each data point, important for exporting data for users and visualizations
        """
        clustermodel = cls(None, None, None, None)
        clustermodel.load_model(cls.get_model_path(directory))
        clustermodel.index_to_key = index_to_key
        clustermodel.data = data
        clustermodel.clusters = clustermodel.clustering_model.predict(data)

        clustermodel.model_name = cls.load_model_name(
            cls.get_param_json_path(directory)
        )
        return clustermodel


class DocumentClusteringModel(ClusteringModel):
    def get_cluster_results_as_df(self, corpus=None, doc_col_name="id", join_df=None):
        """Returns the topic probabilities for each document in the training corpus as a pandas DataFrame

        :param corpus: SparkCorpus
        :param doc_col_name: str, column name that identifies for documents
        :param join_df: Pandas DataFrame, optionally inner join this dataframe on doc_col_name the in the returned results
        """
        topic_probabilities = list()

        for doc_id, topics in self.get_topic_assignments(corpus).items():
            topic_probabilities.extend([(doc_id, t[0], t[1]) for t in topics])

        topics_df = pd.DataFrame(
            topic_probabilities, columns=[doc_col_name, self.model_name, "probability"]
        )

        topics_df[self.model_name] = topics_df[self.model_name].astype("category")
        if join_df is not None:
            topics_df = pd.merge(
                topics_df, join_df, how="inner", on=doc_col_name, sort=False
            )

        return topics_df

    def get_metrics(self):
        """Returns LDA coherence in dictionary
        .. TODO Add exclusivity and other metrics
        """
        return {"Coherence": self.get_coherence_model().get_coherence()}


class TfIdfDocumentClusters(DocumentClusteringModel):
    # TODO
    pass


class GensimLDAModel(DocumentClusteringModel):
    """Wrapper around the gensim LdaMulticore model to train on an iterable corpus or SparkRedditCorpus object
    See http://dirichlet.net/pdf/wallach09rethinking.pdf for notes on alpha and eta priors, where it was found an asymmetric prior on doc-topic dist and symmetric prior on topic-word dist performs best.
    """

    MODEL_FILE = "gensim_lda.gz"

    def __init__(
        self,
        corpus,
        model_name,
        id2word,
        num_topics=250,
        alpha="asymmetric",
        eta="symmetric",
        iterations=1000,
        **kwargs,
    ):
        """Initializes an LDA model in gensim
        :param corpus: SparkCorpus with vectorized column for document BOW representations
        :param id2word: dict, {int -> str}, indexes the words in the vocabulary
        :param model_name: str, how to identify the model
        :param num_topics: int, number of topics to use for this model
        :param alpha: str, opinionated choice about doc-topic prior passed to Gensim LDA model
        :param eta: str, opinionated choice about topic-word prior passed to Gensim LDA model
        :param iterations: int, maximum number of iterations when infering the model
        :param kwargs: Any other LDA params that should be set, especially consider setting and workers
        """
        self.corpus = corpus
        self.word2id = {v: k for k, v in id2word.items()}
        self.clustering_model = gm.ldamulticore.LdaMulticore(
            num_topics=num_topics,
            id2word=id2word,
            alpha=alpha,
            eta=eta,
            iterations=iterations,
            **kwargs,
        )
        self.model_name = model_name

    def train(self):
        """Trains LDA topic model on the corpus
        Returns topic assignments for each document in the training as {str -> list((int, float))}
        """
        logger.info("Staring GensimLDAModel training")
        self.clustering_model.update(
            self.corpus.collect_column_to_list(self.corpus.vectorized_col, True)
        )
        logger.info("Finished GensimLDAModel training")

    def predict(self, bow_docs):
        """Returns topic assignments as a numpy array of shape (len(bow_docs), num_topics) where each cell represents the probability of a document being associated with a topic

        :param bow_docs: iterable of lists of (int, float) representing docs in bag-of-words format
        """
        logger.info("Starting topic predictions using trained GensimLDAModel")
        result = np.zeros((len(bow_docs), self.clustering_model.num_topics))
        for i, bow in enumerate(bow_docs):
            indices, topic_probs = zip(*self.clustering_model.get_document_topics(bow))
            result[i, indices] = topic_probs

        logger.info("Finished topic predictions with GensimLDAModel")
        return result

    def get_topic_assignments(self, corpus=None):
        """Returns {str-> list((int,float))}, the topic assignments for each document id as a list of list of (int, float)

        :param corpus_iter: SparkCorpus object
        """
        logger.info(
            "Starting topic assignments predictions using trained GensimLDAModel"
        )
        if corpus is None:
            logger.debug("Getting iterator for vectorized docs")
            current_iterator = self.corpus.get_vectorized_column_iterator(
                use_id_col=True
            )
        else:
            current_iterator = corpus.get_vectorized_column_iterator(use_id_col=True)
        results = dict()
        for doc_id, bow_doc in current_iterator:
            results[doc_id] = self.clustering_model.get_document_topics(bow_doc)

        logger.info("Finished topic assignment predictions with GensimLDAModel")

        return results

    def get_top_terms(self, num_terms=20):
        """Returns the top words for each learned topic as list of [(topic_id, [(word, probability)...]),...]
        :param num_terms: int, How many of the top words to return for each topic
        """
        return self.clustering_model.show_topics(
            num_topics=-1, num_words=num_terms, formatted=False
        )

    def get_top_terms_as_dataframe(self, num_terms=20, output_col="top_terms"):
        """Returns the top words for each learned topic as a pandas dataframe
        :param num_terms: int, How many of the top words to return for each topic
        :param output_col: str, What to name the column that sotres terms in output
        """
        topic_ids, word_probs = zip(*self.get_top_terms(num_terms=num_terms))
        term_strings = [
            " ".join([w[0] for w in topic_words]) for topic_words in word_probs
        ]

        return pd.DataFrame({"topic": topic_ids, output_col: term_strings})

    def get_coherence_model(self, topn=20):
        """Returns a Gensim CoherenceModel used to calculate the trained model's coherence on the training corpus

        :param topn: int, number of top words to be extracted for each topic
        """
        coherence_model = gm.coherencemodel.CoherenceModel(
            self.clustering_model,
            corpus=self.corpus.collect_column_to_list(self.corpus.vectorized_col, True),
            coherence="u_mass",
            topn=topn,
        )

        return coherence_model

    def get_metrics(self, topn=20):
        """Returns LDA coherence in dictionary
        .. TODO Add exclusivity and other metrics

        :param topn: int, number of top words to be extracted for each topic
        """
        logger.info("Starting computing LDA metrics")
        metrics = {"Coherence": self.get_coherence_model(topn).get_coherence()}
        logger.info("Finished computing LDA metrics: %s", metrics)
        return metrics

    def get_term_topics(self, word):
        """Returns the most relevant topics to the word as a list of (int, float) representing topic id and probability (relevence to the given word) worded by decreasing probability

        :param word: str, word of interest
        """
        if word in self.word2id:
            return sorted(
                self.clustering_model.get_term_topics(self.word2id[word]),
                key=lambda x: x[1],
                reverse=True,
            )
        else:
            return []

    def get_parameters(self):
        """Returns the model's paramters as a dictionary"""
        params = {}
        params[self.MODEL_NAME_KEY] = self.model_name
        params["num_topics"] = self.clustering_model.num_topics
        params["decay"] = self.clustering_model.decay
        params["offset"] = self.clustering_model.offset
        params["iterations"] = self.clustering_model.iterations
        return params

    def save_model(self, path):
        """Save the LDA model to the path
        :param path: str or open file-like object, Path to save model to file
        """
        logger.debug("Saving Gensim LDA model to %s", path)
        self.clustering_model.save(path)
        logger.debug("Gensim LDA model successfully saved")

    def load_model(self, model_path):
        """Load a Gensim LdaMulticore into the current object and configure the CoherenceModel accordingly
        :param model_path
        """
        self.clustering_model = gm.ldamulticore.LdaMulticore.load(model_path)
        self.word2id = {v: k for k, v in self.clustering_model.id2word.items()}

    @classmethod
    def load(cls, load_dir, corpus):
        """
        :param load_dir: str, directory to load LDA model files from
        :param corpus: SparkCorpus
        """
        loaded_model = cls(corpus, None, {0: "dummy"})
        loaded_model.load_model(cls.get_model_path(load_dir))
        loaded_model.model_name = cls.load_model_name(cls.get_param_json_path(load_dir))
        return loaded_model


class SparkLDAModel(DocumentClusteringModel):
    """Wrapper around Spark LDA to train on SparkRedditCorpus

    # latent-dirichlet-allocation-lda for most parameter options
    See https://spark.apache.org/docs/latest/mllib-clustering.html
    """

    TRANSFORMER_FILE = "spark_lda_transformer"
    MODEL_FILE = "spark_lda_model"
    INDEX_FILE = "index.pickle"

    def __init__(
        self,
        corpus,
        model_name,
        id2word,
        num_topics=250,
        optimizer="online",
        use_asymmetric_alpha=True,
        alpha_doc_concentration=None,
        **kwargs,
    ):
        """
        :param corpus: SparkCorpus object, use its vectorized_col as input to LDA
        :param id2word: dict, {int -> str}, indexes the words in the vocabulary
        :param model_name: str, how to identify the model
        :param num_topics: int, number of topics to use for this model
        :param num_topics: int, how many topics to train model for
        :param optimizer: The optimization algorithm to use, 'online' or 'em'
        :param use_asymmetric_alpha: boolean, use to create an asymmetric alpha (different alpha for each topic), cannot be used with 'em' optimizer
        :param alpha_doc_concentration: float, specify to override the default alpha value
        """
        self.corpus = corpus
        self.model_name = model_name

        alphas = self.get_starting_alphas(
            optimizer, use_asymmetric_alpha, alpha_doc_concentration, num_topics
        )

        self.transformer = sparkmc.LDA(
            featuresCol=corpus.vectorized_col,
            k=num_topics,
            optimizer=optimizer,
            docConcentration=alphas,
            **kwargs,
        )
        self.clustering_model = None
        self.set_index(id2word)

    def set_index(self, id2word):
        """Sets the vocabulary index for the model and creates a Spark function
        for displaying terms in output dataframes, rather than term indices

        :param id2word: dict, maps index to term
        """
        self.id2word = id2word

        # This function is for putting document terms in topic-terms output, rather than indices
        def lookup_terms(index_list):
            return " ".join([id2word[i] for i in index_list])

        self.lookup_terms_udf = fn.udf(lookup_terms, sparktypes.StringType())

    @property
    def num_topics(self):
        return self.transformer.getK()

    def train(self):
        """Fits the LDA model to the corpus.
        Returns topic assignments for each document in the training data as {str -> list((int, float))}
        """
        logger.info("Starting training SparkLDAModel")
        self.clustering_model = self.transformer.fit(self.corpus.document_dataframe)
        logger.info("Finished training SparkLDAModel")

    def predict(self, spark_corpus):
        """Returns topic assignments as a numpy array of shape (len(bow_docs), num_topics) where each cell represents the probability of a document being associated with a topic

        :param spark_corpus: SparkCorpus object, must have a vectorized column with the same name as the vectorized column in training data
        """
        logger.info("Starting predictions with trained SparkLDAModel")
        predictions = ihop.text_processing.SparkCorpus(
            self.clustering_model.transform(spark_corpus.document_dataframe)
        )
        logger.debug("Collecting predictions to list")
        collected_predictions = predictions.collect_column_to_list(
            self.clustering_model.getOutputCol()
        )
        logger.info("Finished predictions using SparkLDAModel")
        return collected_predictions

    def get_topic_assignments(self, spark_corpus=None):
        """Returns {str -> list((int, float))}, the topic assignments for each document id as a list of list of(int, float)

        :param spark_corpus: SparkCorpus object, must have a vectorized column with same name as the vectorized column in training data
        """
        logger.info("Starting topic assignment predictions with SparkLDAModel")
        if spark_corpus is None:
            spark_corpus = self.corpus

        input_data = spark_corpus.document_dataframe
        id_col = spark_corpus.id_col

        topic_dist_column = self.clustering_model.getTopicDistributionCol()
        topic_assignments_df = self.clustering_model.transform(input_data).select(
            id_col, topic_dist_column
        )
        collected_topic_assignments = topic_assignments_df.rdd.map(
            lambda x: (
                x[id_col],
                [(i, p) for i, p in enumerate(x[topic_dist_column]) if p > 0],
            )
        ).collect()
        logger.info("Finished topic assignment predictions with SparkLDAModel")
        return dict(collected_topic_assignments)

    def get_top_terms(self, num_words=20):
        """Returns the top words for each learned topic as list of [(topic_id, [(word, probability)...]),...]
        :param num_words: int, How many of the top words to return for each topic
        """
        # describeTopics produces a dataframe with columns 'topic','termIndices', 'termWeights'
        topics_df = self.clustering_model.describeTopics(maxTermsPerTopic=num_words)
        topics_list = topics_df.rdd.map(
            lambda r: (r["topic"], r["termIndices"], r["termWeights"])
        ).collect()
        return [
            (t[0], list(zip([self.id2word[i] for i in t[1]], t[2])))
            for t in topics_list
        ]

    def get_top_terms_as_dataframe(self, num_terms=20, output_col="top_terms"):
        """Returns the top terms for each learned topic as a pandas dataframe
        :param num_terms: int, How many of the top words to return for each topic
        """
        # describeTopics produces a dataframe with columns 'topic','termIndices', 'termWeights'
        return (
            self.clustering_model.describeTopics(maxTermsPerTopic=num_terms)
            .withColumn(output_col, self.lookup_terms_udf("termIndices"))
            .select("topic", "top_terms")
            .toPandas()
        )

    def get_coherence_model(self, topn=20):
        """Returns a Gensim CoherenceModel corresponding to the
        topic assignments and vocabulary of the training data

        :param topn: int, defaults to 20, how many top terms for a topic to use when computing coherence
        """
        logger.debug("Instantiating CoherenceModel with %s topic terms", topn)
        bow_corpus = self.corpus.collect_column_to_list(
            self.corpus.vectorized_col, True
        )

        top_terms = self.get_top_terms(num_words=topn)
        topics = [[w[0] for w in t[1]] for t in top_terms]
        gensim_dict = gc.Dictionary.from_corpus(bow_corpus, self.id2word)
        return gm.coherencemodel.CoherenceModel(
            topics=topics, corpus=bow_corpus, dictionary=gensim_dict, coherence="u_mass"
        )

    # TODO  - not sure how to implement this for online LDA optimizer
    def get_term_topics(self, word):
        pass

    def get_parameters(self):
        """Returns the model name and salient parameters as a dictionary"""
        param_dict = {}
        param_dict[self.MODEL_NAME_KEY] = self.model_name
        param_dict["num_topics"] = self.num_topics
        for p in [
            "optimizer",
            "maxIter",
            "seed",
            "learningOffset",
            "learningDecay",
            "subsamplingRate",
            "docConcentration",
        ]:
            param_dict[p] = self.transformer.getOrDefault(p)
        return param_dict

    def save(self, output_dir):
        self.save_model(output_dir)
        self.save_index(os.path.join(output_dir, self.INDEX_FILE))
        self.save_parameters(os.path.join(output_dir, self.PARAMETERS_JSON))

    def save_model(self, model_path):
        """Saves the Spark Transformer and Spark Model to the specified directory
        :param model_path: str or Paht, directory containing the Spark model and transformer
        """
        self.clustering_model.save(os.path.join(model_path, self.MODEL_FILE))
        self.transformer.save(os.path.join(model_path, self.TRANSFORMER_FILE))

    def save_index(self, index_path):
        """Save the index to a pickle file
        :param index_path:
        """
        with open(index_path, "wb") as picklefile:
            pickle.dump(self.id2word, picklefile)

    def load_model(self, directory):
        """Loads the LDA transformer and LDA model in the directory to the
        current object

        :param directory: str or path, folder storing Spark LDA transformer and LDAModel
        """
        transformer_path = self.get_transformer_path(directory)
        logger.info("Loading transformer from %s", transformer_path)
        self.transformer = sparkmc.LDA.load(transformer_path)
        logger.debug("Transformer loaded")
        optimizer_type = self.transformer.getOptimizer()
        model_path = self.get_model_path(directory)
        if optimizer_type == "online":
            logger.debug("Detected online optimizer, loading LocalLDAModel")
            self.clustering_model = sparkmc.LocalLDAModel.load(model_path)
        elif optimizer_type == "em":
            logger.debug("Detected em optimizer, loading DistributedLDAModel")
            self.clustering_model = sparkmc.DistributedLDAModel.load(model_path)

    def load_index(self, index_path):
        with open(index_path, "rb") as picklefile:
            self.set_index(pickle.load(picklefile))

    @classmethod
    def load(cls, directory, corpus):
        loaded_lda_model = cls(corpus, None, {}, use_asymmetric_alpha=False)
        loaded_lda_model.model_name = cls.load_model_name(
            cls.get_param_json_path(directory)
        )
        loaded_lda_model.load_model(directory)
        loaded_lda_model.load_index(os.path.join(directory, cls.INDEX_FILE))
        return loaded_lda_model

    @classmethod
    def get_starting_alphas(
        cls, optimizer, use_asymmetric_alpha, starting_alpha, num_topics
    ):
        """Returns a scalar or list of starting alpha or document concentration parameters, checking that the options are allowed by Spark. If no

        :param optimizer: str, 'em' or 'online'
        :param use_asymmetric_alpha: boolean, when True returns a list of length num_topics
        :param starting_alpha: float, value to override the default starting alpha values (see Spark documentation)
        :param num_topics: int, how many topics in the model
        """
        if use_asymmetric_alpha:
            if optimizer == "em":
                raise NotImplementedError(
                    "Spark 'em' LDA optimizer doesn't support asymmetric docConcentration"
                )

            if optimizer == "online":
                if starting_alpha is not None:
                    logger.debug("Using symmetric starting alpha: %s", starting_alpha)
                    return [starting_alpha] * num_topics
                else:
                    logger.debug("Using asymmetric alpha for %s topics", num_topics)
                    asymm_alphas = [] * num_topics
                    offset = np.sqrt(num_topics)
                    # This matches the Gensim default for asymmetric alpha
                    for i in range(num_topics):
                        asymm_alphas.append(1 / (i + offset))

                    return asymm_alphas

        logger.debug("Using symmetric starting alpha: %s", starting_alpha)
        return starting_alpha

    @classmethod
    def get_transformer_path(cls, directory):
        """Returns the filename for a transformer file in the given directory"""
        return os.path.join(directory, cls.TRANSFORMER_FILE)


def main(
    model_choice,
    data,
    index,
    experiment_dir,
    cluster_params,
    clusters_csv_filename="clusters.csv",
    words_csv_filename="keywords.csv",
    metrics_json="metrics.json",
    model_name=None,
    is_quiet=False,
):
    """Main method to train a clustering model, then save model and cluster outputs. Returns the trained model.

    :param model_choice: str, type of model to instantiate
    :param data: data used to train the model, type is dependent on the model choice
    :param index: dict, int -> str, how to name each data point, important for exporting data for users and visualizations
    :param experiment_dir: str, path to model output directory
    :param cluster_params: dict, any keyword arguments to pass along to sklearn or Gensim
    :param clusters_csv_filename: str, how to name the CSV file storing cluster output
    :param words_csv_filename: str, if the model is LDA, how to name csv filename storing topic keywords
    :param metrics_json: str, if specified, save model metrics to this file as a json
    :param model_name: str or None, if not None, overrides the default model name
    :param is_quiet: boolean, set to true to silence print statements for metrics
    """
    model = ClusteringModelFactory.init_clustering_model(
        model_choice, data, index, model_name, **cluster_params
    )
    logger.info("Training model %s", model.model_name)
    model.train()
    logger.info("Finished training model %s", model.model_name)
    logger.info("Saving model %s to %s", model.model_name, experiment_dir)
    model.save(experiment_dir)

    # TODO clean up - encapsulate the following in a save_experimental_results method for the models
    if not is_quiet or metrics_json is not None:
        metrics = model.get_metrics()
        logger.info("Model performance metrics: %s", metrics)
        if metrics_json is not None:
            with open(os.path.join(experiment_dir, metrics_json), "w") as f:
                json.dump(metrics, f, cls=ihop.utils.NumpyFloatEncoder)

    if clusters_csv_filename is not None:
        cluster_csv = os.path.join(experiment_dir, clusters_csv_filename)
        logger.info("Saving clusters to CSV %s", cluster_csv)
        model.get_cluster_results_as_df().to_csv(cluster_csv, index=False)

    if (
        model_choice
        in [ClusteringModelFactory.GENSIM_LDA, ClusteringModelFactory.SPARK_LDA]
        and words_csv_filename is not None
    ):
        words_csv = os.path.join(experiment_dir, words_csv_filename)
        logger.info("Saving topic keywords to CSV %s", words_csv)
        model.get_top_terms_as_dataframe().to_csv(words_csv, index=False)

    return model


parser = argparse.ArgumentParser(description="Produce clusterings of the input data")
parser.add_argument(
    "-q",
    "--quiet",
    action="store_true",
    help="Use to turn off verbose info statements that require extra computation.",
)
parser.add_argument(
    "--config",
    type=pathlib.Path,
    help="JSON file used to override default logging and spark configurations",
)

parser.add_argument(
    "input",
    nargs="+",
    help="Path to the file containing input data in the specified format",
)
parser.add_argument(
    "--output_dir",
    "-o",
    required=True,
    help="Directory to save the trained model and any additional data and parameters",
)

parser.add_argument(
    "--data_type",
    "-d",
    help=f"Specify the format of the input data fed to the clustering model: Gensim KeyedVectors, SparkDocuments for raw Reddit submission and comment text documents in a parquet or SparkVectorized for a folder containing vectorized documents in parquet with a serialized Spark pipeline for the vocab index. Defaults to '{KEYED_VECTORS}'",
    choices=[KEYED_VECTORS, SPARK_DOCS, SPARK_VEC],
    default=KEYED_VECTORS,
)
parser.add_argument(
    "--cluster_type",
    "-c",
    help=f"The type of clustering model to train. Defaults to '{ClusteringModelFactory.KMEANS}'",
    choices=ClusteringModelFactory.DEFAULT_MODEL_PARAMS.keys(),
    default=ClusteringModelFactory.KMEANS,
)
parser.add_argument(
    "--cluster_params",
    "-p",
    nargs="?",
    type=json.loads,
    default="{}",
    help="JSON defining overriding or additional parameters to pass to the sklearn or Gensim model",
)

# Used for text data only
parser.add_argument(
    "--min_doc_frequency",
    default=0.05,
    type=float,
    help="Minimum document frequency. Only used when data type is 'SparkDocuments'. Defaults to 0.05.",
)
parser.add_argument(
    "--max_doc_frequency",
    type=float,
    default=0.95,
    help="Maximum document frequency. Only used when data type is 'SparkDocuments'. Defaults to 0.95.",
)
parser.add_argument(
    "--max_time_delta",
    "-x",
    type=pytimeparse.parse,
    help="Specify a maximum allowed time between the creation time of a submission creation and when a comment is added. Only used when data type is 'SparkDocuments'. Can be formatted like '1d2h30m2s' or '26:30:02'. Defaults to 72h.",
    default="72h",
)
parser.add_argument(
    "--min_time_delta",
    "-m",
    type=pytimeparse.parse,
    help="Optionally specify a minimum allowed time between the creation time of a submission creation and when a comment is added. Only used when data type is 'SparkDocuments'. Can be formatted like '1d2h30m2s' or '26:30:02'. Defaults to 3s.",
    default="3s",
)

parser.add_argument(
    "--model-name",
    type=str,
    help="Override the default model name used for the clustering model.",
)


if __name__ == "__main__":
    try:
        # TODO Clean this up a bit
        args = parser.parse_args()
        config = ihop.utils.parse_config_file(args.config)
        ihop.utils.configure_logging(config[1])
        logger.debug("Script arguments: %s", args)
        if (
            args.data_type == KEYED_VECTORS
            and args.cluster_type == ClusteringModelFactory.GENSIM_LDA
        ):
            raise ValueError("LDA models do not support KeyedVectors data type")

        if args.data_type == KEYED_VECTORS:
            logger.debug("Loading KeyedVectors")
            data = gm.KeyedVectors.load(args.input[0])
            index = dict(enumerate(data.index_to_key))
        else:
            spark = ihop.utils.get_spark_session("IHOP LDA Clustering", config[0])

            if args.data_type == SPARK_DOCS:
                logger.debug("Loading SparkDocuments")
                vectorized_corpus, pipeline = ihop.text_preprocessing.prep_spark_corpus(
                    spark.read.parquet(*args.input),
                    min_time_delta=args.min_time_delta,
                    max_time_delta=args.max_time_delta,
                    min_doc_frequency=args.min_doc_frequency,
                    max_doc_frequency=args.max_doc_frequency,
                    output_dir=args.output_dir,
                )

                if not args.quiet:
                    ihop.text_processing.print_document_length_statistics(
                        vectorized_corpus.document_dataframe
                    )

            elif args.data_type == SPARK_VEC:
                logger.debug("Loading SparkVectorized")
                # TODO The actual corpus path should be an argparse option
                # For now, just assume this args.input is the path to a directory that was previous the output_dir for ihop.text_processing.prep_spark_corpus
                vectorized_corpus = ihop.text_processing.SparkCorpus.load(
                    spark,
                    os.path.join(
                        args.input[0], ihop.text_processing.VECTORIZED_CORPUS_FILENAME
                    ),
                )
                pipeline = ihop.text_processing.SparkTextPreprocessingPipeline.load(
                    args.input[0]
                )

            data = vectorized_corpus

            index = pipeline.get_id_to_word()

        main(
            args.cluster_type,
            data,
            index,
            args.output_dir,
            args.cluster_params,
            is_quiet=args.quiet,
            model_name=args.model_name,
        )
    except Exception:
        logger.error("Fatal error during cluster training", exc_info=True)
