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
