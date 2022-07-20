"""Scripts for producing subreddit clusters from a Community2Vec -> KMeans cluster model
to be annotated in two ways:
1) Cluster Label Agreement: Determine how much human experts find the clusters of subreddits produced by the approach coherent
enough to identify a common theme or topic.
2) Subreddit Intruder Identification: Task where annotators pick out a randomly added subreddit from a list of the top subreddits in a cluster in order to determine how well the model aligns with human intuitions about related subreddits.
"""
import argparse
import logging
from multiprocessing.sharedctypes import Value
import pathlib
import random

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
    # Blank fields that annotators will fill in
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


def get_eligible_intruders(
    source_df, group, overall_stddev, model_name, cluster_id, popularity_col="count"
):
    """Returns a dataframe

    :param source_df: pandas DataFrame to filter and sample intruders from
    :param group: pandas GroupBy object containing the popularity_col
    :param overall_stddev: numeric, used to determine the upper and lower bounds for popularity filtering
    :param cluster_id: obj, the id of the current cluster, the intruder cannot be in the same cluster
    :param popularity_col: str, name of column for filtering for an itruder subreddit with similar popularity to the given cluster, defaults to "count"
    """
    count_mean = group[popularity_col].mean()
    upper_bound = count_mean + overall_stddev
    lower_bound = count_mean - overall_stddev

    eligible_intruders = source_df[
        (source_df[popularity_col] >= lower_bound)
        & (source_df[popularity_col] <= upper_bound)
        & (source_df[model_name] != cluster_id)
    ]

    return eligible_intruders


def shuffle_intruder(true_elements, intruder_element):
    """Returns a new list containing all elements with the intruder shuffled in and the index of the intruder

    :param true_elements: list of elements
    :param intruder_element: object
    :raises: ValueError if the intruder is already in true_elements list
    """
    if intruder_element in true_elements:
        raise ValueError(
            f"List of true elements {true_elements} already contains intruder '{intruder_element}'"
        )
    result_list = true_elements + [intruder_element]
    random.shuffle(result_list)

    intruder_idx = result_list.index(intruder_element)
    return result_list, intruder_idx


def export_intruder_task(
    cluster_assignments_df,
    model_name,
    top_n=5,
    do_sort=True,
    random_seed=None,
    popularity_col="count",
):
    """Groups the subreddit clusters, then takes top_n most popular subreddits in group and inserts a random intruder
    that is within 1 stdev of popularity relative the to average popularity of the top_n subreddits of the cluster.

    Returns a dataframe with randomized intruders for the cluster and

    :param cluster_assignments_df: pandas DataFrame, must have a 'count' column containing a numeric indication of subreddits' popularity and a 'subreddit' column
    :param model_name: str, key identifying the model and cluster assignment column
    :param top_n: int, how many of the top subreddits in the cluster (besides the intruder) to keep
    :param do_sort: boolean, True to sort by decreasing popularity before grouping
    """
    logger.info("Determining intruders for model %s", model_name)
    count_stddev = cluster_assignments_df[popularity_col].std()
    logger.info("Standard deviation of subreddit counts: %s", count_stddev)

    random.seed(random_seed)

    source_df = cluster_assignments_df
    if do_sort:
        source_df = cluster_assignments_df.sort_values(
            by=popularity_col, ascending=False
        )

    annotatable_results = list()
    answer_key_results = list()

    cluster_groups = source_df.groupby(model_name).head(top_n)
    for cluster_id, group in cluster_groups:
        logger.info(
            "Finding intruder for cluster id %s with top subreddits: %s",
            cluster_id,
            group["subreddit"],
        )
        eligible_intruders = get_eligible_intruders(
            source_df, group, count_stddev, model_name, cluster_id, popularity_col
        )
        subreddits = list(group["subreddits"].values)

        if len(eligible_intruders) == 0:
            logger.warning("No eligible intruders for cluster id %s", cluster_id)
            continue

        if random_seed is not None:
            intruder = eligible_intruders.sample(random_state=random_seed)[
                "subreddit"
            ].iloc[0]
        else:
            intruder = eligible_intruders.sample()["subreddit"].iloc[0]
        shuffled_elements, intruder_idx = shuffle_intruder(subreddits, intruder)

        curr_annotation_row = [model_name, cluster_id] + shuffled_elements
        annotatable_results.append(curr_annotation_row + [""])

        # Since humans will be reviewing this, we'll index starting at 1, not 0
        curr_answer_key_row = curr_annotation_row + [intruder_idx + 1]
        answer_key_results.append(curr_answer_key_row)

        cols = ["Model ID", "Cluster ID"] + [i + 1 for i in range(top_n + 1)]
        annotation_df = pd.DataFrame.from_records(
            annotatable_results, columns=cols + ["Index of intruder (annotated)"]
        )

        answer_key_df = pd.DataFrame.from_records(
            answer_key_results, columns=cols + ["Index of intruder (answer)"]
        )

        return annotation_df, answer_key_df


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
