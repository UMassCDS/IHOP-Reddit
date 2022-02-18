"""Unit tests for ihop.clustering
"""
import numpy as np
from sklearn.cluster import KMeans

import pytest

import ihop.clustering as ic


@pytest.fixture
def vector_data():
    result = np.zeros((3, 5))
    result[0] = np.full(5, 1)
    result[1] = np.full(5, -1)
    result[2] = np.full(5, 0.5)
    return result


def test_clustering_model(vector_data, tmp_path):
    model = ic.ClusteringModel(vector_data, KMeans(
        n_cluster=2, max_iter=10), "test", {0: "AskReddit", 1: "aww", 2: "NBA"})
    clusters = model.train()
    assert clusters.shape == (2,)
    assert sorted(list(model.get_metrics().keys())) == [
        'Calinski-Harabasz', 'Davies-Bouldin', 'Silhouette']
    assert model.get_parameters() == {}
