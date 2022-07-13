"""Scripts for producing subreddit clusters from a Community2Vec -> KMeans cluster model
to be annotated in two ways:
1) Cluster Label Agreement: Determine how much human experts find the clusters of subreddits produced by the approach coherent
enough to identify a common theme or topic.
2) Subreddit Intruder Identification: Task where annotators pick out a randomly added subreddit from a list of the top subreddits in a cluster in order to determine how well the model aligns with human intuitions about related subreddits.
"""
import argparse
import logging
import pathlib

import pandas as pd

import ihop.utils

logger = logging.getLogger(__name__)

MODEL_ID_KEY = "Model ID"
CLUSTER_ID_KEY = "Cluster ID"
COHERENT_KEY = "Cluster is coherent? (y/n)"
CLUSTER_LABEL_KEY = "Cluster label"
SUBREDDIT_KEY = "subreddit"


def export_cluster_label_agreement_task(cluster_assignments_df, model_name):
    """Group clusters to a more easy-to-annotate format, returning a dataframe that can be written to CSV. This just groups by cluster ID and adds a few columns for annotators' convenience.

    Note that the order of subreddits is arbitrary, unless cluster_assignments_df is sorted before calling this method.

    :param cluster_assignments_df: Dataframe with at least two columns: subreddit, model_name (matches model_name str)
    :param model_name: str, A unique identifier for the model
    """
    dataframe_to_write = (
        cluster_assignments_df.groupby(model_name)[SUBREDDIT_KEY]
        .apply(" ".join)
        .reset_index()
    )
    subreddit_col_header = "subreddits"
    dataframe_to_write.rename(
        columns={model_name: CLUSTER_ID_KEY, SUBREDDIT_KEY: subreddit_col_header},
        inplace=True,
    )
    # Fill in fields needed for annotators
    dataframe_to_write[COHERENT_KEY] = ""
    dataframe_to_write[CLUSTER_LABEL_KEY] = ""
    dataframe_to_write[MODEL_ID_KEY] = model_name

    # Reorder columns
    dataframe_to_write = dataframe_to_write[
        [
            MODEL_ID_KEY,
            CLUSTER_ID_KEY,
            subreddit_col_header,
            COHERENT_KEY,
            CLUSTER_LABEL_KEY,
        ]
    ]

    return dataframe_to_write


def export_intruder_task(cluster_assignments_df, model_name, top_n=5):
    """Groups the subreddit clusters, then takes

    :param cluster_assignments_df: _description_
    :param model_name:
    """
    count_stddev = cluster_assignments_df["count"].std()


def get_answer_key_filename(intruder_csv):
    """Returns the filename for the answer key corresponding to an intruder task CSV file.

    :param intruder_csv_path: Path for CSV file which will be given to annotators
    """
    csv_path = pathlib.Path(intruder_csv)
    answer_key_path = csv_path.parent / (csv_path.stem + "_answers.csv")
    return answer_key_path


def main(
    cluster_assignments_csv,
    subreddit_counts_csv,
    label_task_csv=None,
    intruder_task_csv=None,
):
    """Generates data for the specified annotation tasks and writes the corresponding CSV files.

    :param cluster_assignments_csv: Path to a CSV where the first column is the subreddit and the second is the cluster assignment for the model
    :param subreddit_counts_csv: Path to a CSV where the first column is the subreddit and the second is the number of comments posted in that subreddit over the model timeframe
    :param label_task_csv: Path to desired output file for the cluster coherence and labeling task or None to skip that task
    :param intruder_task_csv: Path to the desired output file for the subreddit intruder task or None to skip that task
    """
    if label_task_csv is None and intruder_task_csv is None:
        raise ValueError("No annotation task output specified.")

    cluster_assignments_df = pd.read_csv(cluster_assignments_csv, header=0)
    model_name = cluster_assignments_df.columns[1]

    subreddit_counts_df = pd.read_csv(subreddit_counts_csv, header=0)

    # Join in subreddit counts, sorting with most popular subreddits first
    cluster_assignments_df = cluster_assignments_df.join(
        subreddit_counts_df, on="subreddit"
    ).sort_values(by="count", ascending=False)

    if label_task_csv is not None:
        label_task_df = export_cluster_label_agreement_task(
            cluster_assignments_df, model_name
        )
        label_task_df.to_csv(label_task_csv, index=False)

    if intruder_task_csv is not None:
        intruder_task_df, answer_key_df = export_intruder_task(cluster_assignments_df)
        answer_key_csv = get_answer_key_filename(intruder_task_csv)
        intruder_task_df.to_csv(intruder_task_csv, index=False)
        answer_key_df.to_csv(answer_key_csv, index=False)


parser = argparse.ArgumentParser(
    description="Given a subreddit clustering model and statistics about number of comments in a subreddit in that time frame, produce data in annotatable formats for cluster label agreement and subreddit intruder tasks."
)

parser.add_argument(
    "--config",
    type=pathlib.Path,
    help="JSON file used to override default logging and spark configurations",
)

parser.add_argument(
    "cluster_assignment_csv",
    help="Path to CSV with cluster assignments from a given subreddit clustering model",
)
parser.add_argument(
    "subbreddit_counts_csv",
    help="Path to CSV storing counts of entries in that subreddit over the same time period as the cluster model.",
)

parser.add_argument(
    "--cluster-labeling-csv",
    "-c",
    help="Output path to CSV for the cluster labeling task. If not specified, task will be skipped.",
)

parser.add_argument(
    "--intruder-csv",
    "-i",
    help="Output path to CSV for the subreddit intruder identification task. If not specified, task will be skipped.",
)

parser.add_argument("--intruder-key", "-k")


if __name__ == "__main__":
    try:
        args = parser.parse_args()
        config = ihop.utils.parse_config_file(args.config)
        ihop.utils.configure_logging(config[1])
        logger.debug("Script arguments: %s", args)
        main(
            args.cluster,
            args.cluster_labeling_csv,
            args.intruder_csv,
            args.subreddit_counts_csv,
        )
    except Exception:
        logger.error("Fatal error during annotation task export", exc_info=True)
