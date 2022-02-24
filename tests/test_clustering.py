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
    pipeline = tp.SparkTextPreprocessingPipeline('text', maxDF=10, minDF=0)
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

    # Giving the AskReddit vector again predicts the same cluster
    assert model.predict(np.full((1, 5), 1)) == clusters[0]


def test_cluster_model_serialization(vector_data, tmp_path):
    model = ic.ClusteringModel(vector_data, KMeans(
        n_clusters=2, max_iter=10), "test", {0: "AskReddit", 1: "aww", 2: "NBA"})
    model.train()
    model.save(tmp_path)

    loaded_model = ic.ClusteringModel.load(
        tmp_path, vector_data, {0: "AskReddit", 1: "aww", 2: "NBA"})
    assert loaded_model.index_to_key == model.index_to_key
    assert loaded_model.get_parameters() == model.get_parameters()
    assert loaded_model.clusters.shape == model.clusters.shape
    assert (loaded_model.clusters == model.clusters).all()


def test_lda(text_features):
    index = text_features.index
    corpus_iter = tp.SparkCorpusIterator(
        text_features.corpus.document_dataframe, "vectorized", True)
    lda = ic.GensimLDAModel(corpus_iter, "test_lda", index,
                            num_topics=2, iterations=5, random_state=8)
    lda.train()
    topic_assignments = lda.get_topic_assignments()
    assert len(topic_assignments) == 3
    assert set(topic_assignments.keys()) == {'a1', 'b2', 'c3'}

    predictions = lda.predict([[(0, 1.0), (2, 2.0), (6, 3.0)]])
    assert predictions.shape == (1, 2)

    parameters = lda.get_parameters()
    assert parameters["model_name"] == "test_lda"
    assert parameters["num_topics"] == 2
    assert parameters["decay"] == 0.5
    assert parameters["offset"] == 1.0
    assert parameters["iterations"] == 5

    top_words_list = lda.get_top_words()
    assert len(top_words_list) == 2

    # Vocabulary size for the toy example is 9, so that's the max number of words returned for the topics
    for t in top_words_list:
        assert len(t[1]) == 9

    assert lda.get_top_words_as_dataframe().shape == (2, 2)

    # This is a word in the vocab, so it should return something
    assert len(lda.get_term_topics('sentence')) == 2


def test_lda_serialization(text_features, tmp_path):
    index = text_features.index
    corpus_iter = tp.SparkCorpusIterator(
        text_features.corpus.document_dataframe, "vectorized", True)
    lda = ic.GensimLDAModel(corpus_iter, "test_lda", index,
                            num_topics=2, iterations=5, random_state=8)
    lda.train()
    lda.save(tmp_path)
    loaded_lda = ic.GensimLDAModel.load(tmp_path, corpus_iter)
    assert loaded_lda.lda_model.id2word == index
    assert len(loaded_lda.get_term_topics('sentence')) == 2
    sample_bow = [[(1, 3.0), (7, 2.0)]]
    assert (loaded_lda.predict(sample_bow) == lda.predict(sample_bow)).all()
    assert loaded_lda.get_parameters() == lda.get_parameters()
