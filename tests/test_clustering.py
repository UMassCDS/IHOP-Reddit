"""Unit tests for ihop.clustering
"""
import collections

import numpy as np
from sklearn.cluster import KMeans
import pytest

import ihop.clustering as ic
import ihop.text_processing as tp

Corpus = collections.namedtuple("Corpus", "corpus index")


@pytest.fixture
def vector_data():
    result = np.zeros((3, 5))
    result[0] = np.full(5, 1)
    result[1] = np.full(5, -1)
    result[2] = np.full(5, 0.5)
    return result


@pytest.fixture
def text_features(spark):
    test_data = [
        {'id': 'a1', 'text': 'Hello! This is a sentence.'},
        {'id': 'b2', 'text': 'Yet another sentence...'},
        {'id': 'c3', 'text': 'The last sentence'}
    ]
    dataframe = spark.createDataFrame(test_data)
    pipeline = tp.SparkTextPreprocessingPipeline('text')
    vectorized_df = tp.SparkCorpus(pipeline.fit_transform(dataframe))
    return Corpus(vectorized_df, pipeline.get_id_to_word())


def test_clustering_model(vector_data):
    model = ic.ClusteringModel(vector_data, KMeans(
        n_clusters=2, max_iter=10), "test", {0: "AskReddit", 1: "aww", 2: "NBA"})
    clusters = model.train()
    assert clusters.shape == (3,)
    assert sorted(list(model.get_metrics().keys())) == [
        'Calinski-Harabasz', 'Davies-Bouldin', 'Silhouette']

    expected_params = {
        'algorithm': 'auto',
        'copy_x': True,
        'init': 'k-means++',
        'max_iter': 10,
        'model_name': 'test',
        'n_clusters': 2,
        'n_init': 10,
        'random_state': None,
        'tol': 0.0001,
        'verbose': 0,
    }

    assert model.get_parameters() == expected_params


def test_cluster_model_serialization(vector_data, tmp_path):
    model = ic.ClusteringModel(vector_data, KMeans(
        n_clusters=2, max_iter=10), "test", {0: "AskReddit", 1: "aww", 2: "NBA"})
    model.train()
    model.save(tmp_path)

    loaded_model = ic.ClusteringModel.load(tmp_path)
    assert loaded_model.index_to_key == model.index_to_key
    assert loaded_model.get_parameters() == model.get_parameters()


def test_lda(text_features):
    index = text_features.index
    corpus_iter = tp.SparkCorpusIterator(
        text_features.corpus.document_dataframe, "vectorized", True)
    lda = ic.GensimLDAModel(corpus_iter, "test_lda", index,
                            num_topics=2, iterations=5)
    lda.train()
    assert lda.get_topic_scores(
        text_features.corpus.iterate_over_doc_vectors()).shape == (3, 2)