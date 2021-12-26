"""Loads Reddit json data to a Spark dataframe, which can be filtered using
SQL-like operations, operated on using Spark's ML library or exported to pandas/sklearn formats.

..TODO Consider how best to match up most active subreddits by number of submissions and number of comments
..TODO Best output formats?
"""
import argparse

from pyspark.sql import SparkSession
from pyspark.sql.functions import count_distinct

COMMENTS="comments"
SUBMISSIONS="submissions"
DEFAULT_TOP_N=10000

# How deleted authors are indicated in json
AUTHOR_DELETED = "[deleted]"

# See https://github.com/pushshift/api for details
SCHEMAS = {
        COMMENTS: "id STRING, parent_id STRING, score INTEGER, author_flair_css_class STRING, author_flair_text STRING, link_id STRING, author STRING, subreddit STRING, body STRING, edited INTEGER, gilded STRING, controversiality INTEGER, created_utc STRING, distinguished STRING",
        SUBMISSIONS: "author STRING, author_flair_css_class STRING, author_flair_text STRING, created_utc STRING, distinguished STRING, domain STRING, edited INTEGER, gilded STRING, id STRING, is_self BOOLEAN, over_18 BOOLEAN, score INTEGER, selftext STRING, title STRING, url STRING, subreddit STRING"
        }

def filter_top_n(dataframe, col='subreddit', n=DEFAULT_TOP_N):
    """Determine the top n most frequent values in a column, then filter the dataframe to only results with those values. Returns a dataframe that's a subset of the original input dataframe.

    :param dataframe: Spark dataframe
    :param col: str, Column to use for top n elements
    :param n: int, number of top elements to consider
    """
    top_n_counts = dataframe.groupBy(col).count().orderBy(['count', col], ascending=[0,1]).limit(n)
    result = dataframe.join(top_n_counts, dataframe[col] == top_n_counts[col], 'inner')
    return result


def remove_deleted_authors(dataframe):
    """Filters out comments or submissions that have had the author deleted
    """
    return dataframe.where(dataframe.author != AUTHOR_DELETED)


def print_comparison_stats(original_df, top_n_df):
    """Compares the number of unique subreddits and users in the original and filtered datasets.
    :param original_df: The full unfiltered Spark Dataframe
    :param top_n_df: The filtered SparkDataframe
    """
    original_distinct_subreddits = original_df.agg(count_distinct(original_df.subreddit)).collect()[0].count
    filtered_distinct_subreddits = top_n_df.agg(count_distinct(top_n_df.subreddit)).collect()[0].count
    print("Number of subreddits overall:", original_distinct_subreddits)
    print("Number of subreddits after filtering (sanity check, should match n):", filtered_distinct_subreddits)

    original_comments = original_df.count()
    comments_after_filtering = top_n_df.count()
    print("Number comments before filtering:", original_comments)
    print("Number comments after filtering:", comments_after_filtering)
    print("Percentage of original comments covered:", comments_after_filtering/original_comments)

    original_users = original_df.agg(count_distinct(original_df.author)).collect()[0].count
    filtered_users = top_n_df.agg(count_distinct(top_n_df.author)).collect()[0].count
    print("Number users before filtering:", original_users)
    print("Number users after filtering:", filtered_users)
    print("Percentage of original comments covered:", filtered_users/original_users)


def get_spark_dataframe(inputs, spark, reddit_type):
    """
    :param inputs: Paths to Reddit json data
    :param spark: SparkSession
    :param reddit_type: "comments" or "submissions"
    """
    return spark.read.format("json").option("encoding", "UTF-8").schema(SCHEMAS[reddit_type]).load(inputs)


def community2vec(output_path, inputs, reddit_type=COMMENTS, top_n = DEFAULT_TOP_N):
    """Writes output data for training community2vec (users as 'documents/context', subreddits as 'words')
    :param output_path: Where to write out the parquet? data format
    :param inputs: Paths to read JSON Reddit data from
    :param reddit_type: 'comments' or 'submissions'
    """
    # create the Spark session
    spark = SparkSession.builder.appName("IHOP import data").getOrCreate()

    if reddit_type in [COMMENTS, SUBMISSIONS]:
        spark_df = get_spark_dataframe(inputs, spark, reddit_type)
        top_n_df = filter_top_n(spark_df, top_n = top_n)
        top_n_df = remove_deleted_authors(top_n_df)
        print_comparison_stats(spark_df, top_n_df)

        # TODO Combine on user, the write out to file

    else:
        raise ValueError(f"Reddit data type {reddit_type} is not valid.")


parser = argparse.ArgumentParser(description="Parse Pushshift Reddit data to Spark parquet dataframe")
subparsers = parser.add_subparsers(dest='subparser_name')

c2v_parser = subparsers.add_parser('c2v', help="Output data to a format that can be used for training community2vec models in Tensorflow")
c2v_parser.add_argument("output", help="Path to output file")
c2v_parser.add_argument("input", nargs='+', help="Paths to input files. They should all be the same type ('comments' or 'submissions')")
c2v_parser.add_argument("-t", "--type", choices=[COMMENTS, SUBMISSIONS], help = "Are these 'comments' or 'submissions' (posts)?")
c2v_parser.add_argument("-n", "--top_n", type=int, default=DEFAULT_TOP_N, help="Use to filter to the top most activae subreddits (by number of comments/submssions).")

topic_modeling_parser = subparsers.add_parser('topic-model', help="Output data to a fomrat that can be used for training topic models in Mallet (or pre-trained WE clusters?)")


if __name__ == "__main__":
    args = parser.parse_args()
    if args.subparser_name=='c2v':
        community2vec(args.output, args.input, args.type)
    elif args.subparser_name=='topic-model':
        # TODO
        print("Topic modeling format not implemented.")
