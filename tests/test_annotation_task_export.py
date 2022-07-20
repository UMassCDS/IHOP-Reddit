"""Unit tests for ihop.annotation_task_export"""
import pandas as pd

import ihop.annotation_task_export as ia


def test_export_cluster_label_agreement_task():
    cluster_assignment_df = pd.DataFrame(
        {
            "subreddit": ["aww", "AskReddit", "conservatives", "leopardsatemyface"],
            "May 2021 c2v kmeans model": [0, 0, 1, 2],
        }
    )

    expected_df = pd.DataFrame(
        {
            "Model ID": ["May 2021 c2v kmeans model"] * 3,
            "Cluster ID": [0, 1, 2],
            "subreddits": ["aww AskReddit", "conservatives", "leopardsatemyface"],
            "Cluster is coherent? (y/n)": [""] * 3,
            "Cluster label": [""] * 3,
        }
    )

    assert expected_df.equals(
        ia.export_cluster_label_agreement_task(
            cluster_assignment_df, "May 2021 c2v kmeans model"
        )
    )


def test_shuffle_intruder():
    shuffled, index = ia.shuffle_intruder(["a", "b", "c"], "z")
    assert len(shuffled) == 4
    assert sorted(shuffled) == ["a", "b", "c", "z"]
    assert index >= 0 and index < 4


def test_get_eligible_intruders():
    source_df = pd.DataFrame(
        {
            "subreddit": [
                "aww",
                "AskReddit",
                "conservatives",
                "leopardsatemyface",
                "knitting",
            ],
            "May 2021 c2v kmeans model": [0, 0, 1, 2, 3],
            "count": [100, 100, 90, 50, 40],
        }
    )
    model_name = "May 2021 c2v kmeans model"
    select_group = source_df[source_df[model_name] == 0]
    intruders_df = ia.get_eligible_intruders(source_df, select_group, 20, model_name, 0)

    assert intruders_df.shape == (1, 3)
    assert intruders_df["subreddit"].iloc[0] == "conservatives"
