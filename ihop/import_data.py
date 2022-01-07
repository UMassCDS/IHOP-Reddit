"""Loads Reddit json data to a Spark dataframe, which can be filtered using
SQL-like operations, operated on using Spark's ML library or exported to pandas/sklearn formats.
"""
import argparse

from pyspark.sql.functions import concat_ws, count_distinct, collect_list
import seaborn as sns

import ihop.utils

COMMENTS="comments"
SUBMISSIONS="submissions"
DEFAULT_TOP_N=10000

# How deleted authors are indicated in json
AUTHOR_DELETED = "[deleted]"

# See https://github.com/pushshift/api for details
# TODO: subreddit ids aren't tracked, unclear if this is necessary
SCHEMAS = {
        COMMENTS: "id STRING, parent_id STRING, score INTEGER, link_id STRING, author STRING, subreddit STRING, body STRING, created_utc INTEGER",
        SUBMISSIONS: "author STRING, created_utc STRING, distinguished STRING, domain STRING, edited INTEGER, gilded STRING, id STRING, is_self BOOLEAN, over_18 BOOLEAN, score INTEGER, selftext STRING, title STRING, url STRING, subreddit STRING"
        }


def get_top_n_counts(dataframe, col='subreddit', n=DEFAULT_TOP_N):
    """Determine the top n most frequent values in a column. Return a dataframe of those values with their counts. Results are ordered by counts, then alphabetical ordering of the values in the column, to break ties at the lower end.
    :param dataframe: Spark DataFrame
    :param col: str, the column to groupBy for counts
    :param n: int, limit results to the top n most frequent values
    """
    return dataframe.groupBy(col).count().orderBy(['count', col], ascending=[0,1]).limit(n)



def filter_top_n(dataframe, top_n_counts, col='subreddit'):
    """Filter the dataframe to only results with those values in top_n_counts.
    Returns a dataframe that's a subset of the original input dataframe.

    :param dataframe: Spark dataframe to be filtered
    :param top_n_counts: Spark dataframe with values to be filtered
    :param col: str, Column to use for top n elements
    """
    return dataframe.join(top_n_counts, col, 'inner')


def remove_deleted_authors(dataframe):
    """Filters out comments or submissions that have had the author deleted
    """
    return dataframe.where(dataframe.author != AUTHOR_DELETED)


def print_comparison_stats(original_df, filtered_df, top_n_df):
    """Compares the number of unique subreddits and users in the original and filtered datasets.
    :param original_df: The full unfiltered Spark Dataframe
    :param filtered_df: The original dataframe filtered to include only top
    :param top_n_df: The filtered SparkDataframe
    """
    original_distinct_subreddits = original_df.agg(count_distinct(original_df.subreddit).alias('sr_count')).collect()[0].sr_count
    filtered_distinct_subreddits = top_n_df.agg(count_distinct(top_n_df.subreddit).alias('sr_count')).collect()[0].sr_count
    print("Number of subreddits overall:", original_distinct_subreddits)
    print("Number of subreddits after filtering (sanity check, should match n):", filtered_distinct_subreddits)

    original_comments = original_df.count()
    comments_after_filtering = top_n_df.count()
    print("Number comments before filtering:", original_comments)
    print("Number comments after filtering:", comments_after_filtering)
    print("Percentage of original comments covered:", comments_after_filtering/original_comments)

    original_users = original_df.agg(count_distinct(original_df.author).alias('author_count')).collect()[0].author_count
    filtered_users = filtered_df.agg(count_distinct(filtered_df.author).alias('author_count')).collect()[0].author_count
    print("Number users before filtering:", original_users)
    print("Number users after filtering:", filtered_users)
    print("Percentage of original comments covered:", filtered_users/original_users)


def get_spark_dataframe(inputs, spark, reddit_type):
    """
    :param inputs: Paths to Reddit json data
    :param spark: SparkSession
    :param reddit_type: "comments" or "submissions"
    """
    return spark.read.format("json").option("mode", "DROPMALFORMED").option("encoding", "UTF-8").schema(SCHEMAS[reddit_type]).load(inputs)


def aggregate_for_vectorization(dataframe, context_col="author", word_col="subreddit"):
    """Aggregates data on the context col, using whitespace concatenation to combine the values in the word column.
    Returns a dataframe with one column, the aggregated word column.

    :param dataframe: Spark dataframe
    :param context_col: str, column to group by
    :param word_col: str, column to concatenate together
    """
    return dataframe.groupBy(context_col).agg(concat_ws(" ", collect_list(dataframe[word_col])).alias(word_col)).drop(context_col)


def community2vec(inputs, spark, reddit_type=COMMENTS, top_n=DEFAULT_TOP_N, quiet=False):
    """Returns data for training community2vec using skipgrams (users as 'documents/context', subreddits as 'words') as Spark dataframes. Deleted comments are counted when determining the top most frequent values.
    Returns 2 dataframes: counts of subreddits (vocabulary for community2vec), subreddit comments/submissions aggregated into a list for each author

    :param inputs: Paths to read JSON Reddit data from
    :param spark: SparkSession
    :param reddit_type: 'comments' or 'submissions'
    :param quet: Boolean, true to skip statsitics and plots
    """

    if reddit_type in [COMMENTS, SUBMISSIONS]:
        spark_df = get_spark_dataframe(inputs, spark, reddit_type)
        if not quiet:
            print("Spark dataframe from json:")
            spark_df.show()
        top_n_df = get_top_n_counts(spark_df, n=top_n)
        filtered_df = filter_top_n(spark_df, top_n_df)
        filtered_df = remove_deleted_authors(filtered_df)
        context_word_df = aggregate_for_vectorization(filtered_df)
        if not quiet:
            print("Filtered dataframe head snippet:")
            print(top_n_df.head(10))
            print("Filtered dataframe tail snippet:")
            print(top_n_df.tail(10))
            print_comparison_stats(spark_df, filtered_df, top_n_df)
    else:
        raise ValueError(f"Reddit data type {reddit_type} is not valid.")

    return top_n_df, context_word_df


parser = argparse.ArgumentParser(description="Parse Pushshift Reddit data to formats for community2vec and topic modeling.")
parser.add_argument("-q", "--quiet", action='store_true', help="Use to turn off dataset descriptions and plots")
subparsers = parser.add_subparsers(dest='subparser_name')
c2v_parser = subparsers.add_parser('c2v', help="Output data as indexed subreddits for each user in a format that can be used for training community2vec models in Tensorflow")
c2v_parser.add_argument("subreddit_counts_csv", help="Path to CSV file for counts of top N subreddits")
c2v_parser.add_argument("context_word_dir", help="Path to directory for a compressed file with subreddits a user commented on, one user per line")
c2v_parser.add_argument("input", nargs='+', help="Paths to input files. They should all be the same type ('comments' or 'submissions')")
c2v_parser.add_argument("-t", "--type", choices=[COMMENTS, SUBMISSIONS], help = "Are these 'comments' or 'submissions' (posts)? Default to 'comments'", default=COMMENTS)
c2v_parser.add_argument("-n", "--top_n", type=int, default=DEFAULT_TOP_N, help="Use to filter to the top most active subreddits (by number of comments/submssions). Deleted authors/comments/submissions are considered when calculating counts.")

topic_modeling_parser = subparsers.add_parser('topic-model', help="Output data to a format that can be used for training topic models in Mallet (or pre-trained WE clusters?)")


if __name__ == "__main__":
    args = parser.parse_args()
    if args.subparser_name=='c2v':
        spark = ihop.utils.get_spark_session("IHOP import data", args.quiet)
        top_n_df, context_word_df = community2vec(args.input, spark,
                reddit_type=args.type, top_n=args.top_n, quiet=args.quiet)
        top_n_df.toPandas().to_csv(args.subreddit_counts_csv, index=False)
        context_word_df.coalesce(1).write.option("compression", "bzip2").csv(args.context_word_dir)
    elif args.subparser_name=='topic-model':
        # TODO
        print("Text-based topic modeling format not implemented.")
