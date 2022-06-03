"""Tests for ihop.visualizations
"""
import json

import pandas as pd

import ihop.visualizations as iv


def test_jsonify_stored_df():
    dataframe = pd.DataFrame({"col1": ["a", "b", "a"], "col2": [1, 2, 3]})

    json_results = json.loads(iv.jsonify_stored_df(dataframe))
    assert set(json_results.keys()) == set(["columns", "index", "data"])

    assert json_results["columns"] == ["col1", "col2"]
    data = json_results["data"]

    assert len(data) == 3
    for vals in [["a", 1], ["b", 2], ["a", 3]]:
        assert vals in data


def test_unjsonify_stored_data():
    json_data = '{"columns": ["col1", "col2"], "index": [1, 2, 3], "data": [["a", 1], ["b", 2], ["a", 3]]}'
    dataframe = iv.unjsonify_stored_df(json_data, categorical_columns=["col1"])
    assert list(dataframe.columns) == ["col1", "col2"]
    assert len(dataframe) == 3


def test_assign_other_category_column():
    df = pd.DataFrame(
        {
            "subreddit": ["democrats", "liberals", "conservatives"],
            "cluster_assignment": [2, 2, 4],
        }
    )
    new_df = iv.assign_other_category_column(
        df, "cluster_assignment", "display_clusters", set([2]), "other"
    )
    assert list(new_df.columns) == [
        "subreddit",
        "cluster_assignment",
        "display_clusters",
    ]
    assert list(new_df["cluster_assignment"]) == [2, 2, "other"]

