"""Unit tests for ihop.clustering
"""
import collections

import gensim.models as gm
import gensim.corpora as gc
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
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
        {"id": "a1", "text": "Hello! This is a sentence."},
        {"id": "b2", "text": "Yet another sentence..."},
        {"id": "c3", "text": "The last sentence"},
    ]
    dataframe = spark.createDataFrame(test_data)
    pipeline = tp.SparkTextPreprocessingPipeline(
        "text", maxDF=10, minDF=0, stopLanguage=None
    )
    vectorized_df = tp.SparkCorpus(pipeline.fit_transform(dataframe))
    return Corpus(vectorized_df, pipeline.get_id_to_word())


def test_clustering_model(vector_data):
    model = ic.ClusteringModel(
        vector_data,
        KMeans(n_clusters=2, max_iter=10),
        "test",
        {0: "AskReddit", 1: "aww", 2: "NBA"},
    )
    model.train()
    clusters = model.clusters
    assert clusters.shape == (3,)
    assert set(model.get_metrics().keys()) == set(
        ["Calinski-Harabasz", "Davies-Bouldin", "Silhouette"]
    )

    expected_params = {
        "algorithm": "auto",
        "copy_x": True,
        "init": "k-means++",
        "max_iter": 10,
        "model_name": "test",
        "n_clusters": 2,
        "n_init": 10,
        "random_state": None,
        "tol": 0.0001,
        "verbose": 0,
    }

    assert model.get_parameters() == expected_params

    # Giving the AskReddit vector again predicts the same cluster
    assert model.predict(np.full((1, 5), 1)) == clusters[0]

    assert model.get_cluster_results_as_df().shape == (3, 2)


def test_cluster_model_serialization(vector_data, tmp_path):
    model = ic.ClusteringModel(
        vector_data,
        KMeans(n_clusters=2, max_iter=10),
        "test",
        {0: "AskReddit", 1: "aww", 2: "NBA"},
    )
    model.train()
    model.save(tmp_path)

    loaded_model = ic.ClusteringModel.load(
        tmp_path, vector_data, {0: "AskReddit", 1: "aww", 2: "NBA"}
    )
    assert loaded_model.index_to_key == model.index_to_key
    assert loaded_model.get_parameters() == model.get_parameters()
    assert loaded_model.clusters.shape == model.clusters.shape
    assert (loaded_model.clusters == model.clusters).all()


def test_main_sklearn(vector_data, tmp_path):
    index = {0: "AskReddit", 1: "aww", 2: "NBA"}
    model = ic.main(
        "agglomerative",
        vector_data,
        index,
        tmp_path,
        {"n_clusters": 2, "linkage": "single"},
        model_name="test_agglomerative",
    )
    assert isinstance(model.clustering_model, AgglomerativeClustering)
    assert model.clustering_model.n_clusters == 2
    assert model.clustering_model.linkage == "single"
    assert model.model_name == "test_agglomerative"

    model_path = tmp_path / "sklearn_cluster_model.joblib"
    assert model_path.exists()

    param_json = tmp_path / "parameters.json"
    assert param_json.exists()

    metrics_json = tmp_path / "metrics.json"
    assert metrics_json.exists()

    clusters_csv = tmp_path / "clusters.csv"
    assert clusters_csv.exists()


def test_gensim_lda(text_features):
    lda = ic.GensimLDAModel(
        text_features.corpus,
        "test_lda",
        text_features.index,
        num_topics=2,
        iterations=5,
        random_state=8,
    )
    lda.train()
    topic_assignments = lda.get_topic_assignments()
    assert len(topic_assignments) == 3
    assert set(topic_assignments.keys()) == {"a1", "b2", "c3"}

    predictions = lda.predict([[(0, 1.0), (2, 2.0), (6, 3.0)]])
    assert predictions.shape == (1, 2)

    parameters = lda.get_parameters()
    assert parameters["model_name"] == "test_lda"
    assert parameters["num_topics"] == 2
    assert parameters["decay"] == 0.5
    assert parameters["offset"] == 1.0
    assert parameters["iterations"] == 5

    top_words_list = lda.get_top_terms()
    assert len(top_words_list) == 2
    # word in first topic
    assert isinstance(top_words_list[0][1][0][0], str)
    assert isinstance(top_words_list[0][1][0][1], np.floating)

    # Vocabulary size for the toy example is 9, so that's the max number of words returned for the topics
    for t in top_words_list:
        assert len(t[1]) == 9

    assert lda.get_top_terms_as_dataframe().shape == (2, 2)
    assert len(lda.get_top_terms()) == 2

    # This is a word in the vocab, so it should return something
    assert len(lda.get_term_topics("sentence")) == 2

    metrics_dict = lda.get_metrics()
    assert len(metrics_dict) == 1
    assert "Coherence" in metrics_dict
    assert isinstance(metrics_dict["Coherence"], np.floating)


def test_gesim_lda_serialization(text_features, tmp_path):
    index = text_features.index
    lda = ic.GensimLDAModel(
        text_features.corpus,
        "test_lda",
        text_features.index,
        num_topics=2,
        iterations=5,
        random_state=8,
    )
    lda.train()

    lda.save(tmp_path)
    loaded_lda = ic.GensimLDAModel.load(tmp_path, text_features.corpus)
    assert loaded_lda.clustering_model.id2word == index
    assert len(loaded_lda.get_term_topics("sentence")) == 2
    sample_bow = [[(1, 3.0), (7, 2.0)]]
    assert (loaded_lda.predict(sample_bow) == lda.predict(sample_bow)).all()
    assert loaded_lda.get_parameters() == lda.get_parameters()


def test_spark_lda(text_features):
    lda_model = ic.SparkLDAModel(
        text_features.corpus, "test_spark", text_features.index, num_topics=4
    )
    lda_model.transformer.getDocConcentration() == [1 / 2, 1 / 3, 1 / 4, 1 / 5]
    lda_model.train()
    topic_assignments = lda_model.get_topic_assignments()
    assert len(topic_assignments) == 3
    assert set(topic_assignments.keys()) == {"a1", "b2", "c3"}

    top_terms_df = lda_model.get_top_terms_as_dataframe()
    assert top_terms_df.shape == (4, 2)

    top_terms_list = lda_model.get_top_terms()
    assert len(top_terms_list) == 4
    # word in first topic
    assert isinstance(top_terms_list[0][1][0][0], str)
    assert isinstance(top_terms_list[0][1][0][1], float)
    assert len(top_terms_list) == 4

    metrics_dict = lda_model.get_metrics()
    assert len(metrics_dict) == 1
    assert "Coherence" in metrics_dict
    assert isinstance(metrics_dict["Coherence"], float)


def test_spark_lda_serialization(text_features, tmp_path):
    lda_model = lda_model = ic.SparkLDAModel(
        text_features.corpus, "test_spark", text_features.index, num_topics=4
    )
    lda_model.train()
    lda_model.save(tmp_path)
    loaded_model = ic.SparkLDAModel.load(tmp_path, text_features.corpus)
    assert lda_model.id2word == loaded_model.id2word
    assert (
        lda_model.clustering_model.getDocConcentration()
        == loaded_model.clustering_model.getDocConcentration()
    )
    assert (
        lda_model.clustering_model.estimatedDocConcentration()
        == loaded_model.clustering_model.estimatedDocConcentration()
    )
    assert lda_model.get_parameters() == loaded_model.get_parameters()


def test_main_lda(text_features, tmp_path):
    model = ic.main(
        "gensimlda",
        text_features.corpus,
        text_features.index,
        tmp_path,
        {"num_topics": 2, "alpha": "symmetric"},
        model_name="test_lda",
    )
    assert isinstance(model.clustering_model, gm.LdaMulticore)
    assert model.clustering_model.num_topics == 2
    assert model.model_name == "test_lda"

    model_path = tmp_path / "gensim_lda.gz"
    assert model_path.exists()

    param_json = tmp_path / "parameters.json"
    assert param_json.exists()

    metrics_json = tmp_path / "metrics.json"
    assert metrics_json.exists()

    clusters_csv = tmp_path / "clusters.csv"
    assert clusters_csv.exists()

    words_csv = tmp_path / "keywords.csv"
    assert words_csv.exists()
