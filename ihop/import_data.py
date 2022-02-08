"""Loads Reddit json data to a Spark dataframe, which can be filtered using
SQL-like operations, operated on using Spark's ML library or exported to pandas/sklearn formats.

.. TODO: Remove print statements in favor of logging
"""
import argparse
import logging

import pyspark.sql.functions as fn
from pyspark.sql.window import Window

import ihop.utils

logger = logging.getLogger(__name__)

COMMENTS="comments"
SUBMISSIONS="submissions"
DEFAULT_TOP_N=10000
DEFAULT_USER_EXCLUDE=0.02

# How deleted authors or posts are indicated in json (user removed)
DELETED = "[deleted]"
# How posts removed by moderators/filters are indicated in json
REMOVED = "[removed]"

# The timestamp column
CREATED_UTC = "created_utc"

# See https://github.com/pushshift/api for details
# TODO: subreddit ids aren't tracked, unclear if this is necessary
SCHEMAS = {
    COMMENTS: "id STRING, parent_id STRING, score INTEGER, link_id STRING, author STRING, subreddit STRING, body STRING, created_utc INTEGER",
    SUBMISSIONS: "author STRING, created_utc STRING, id STRING, score INTEGER, selftext STRING, title STRING, url STRING, subreddit STRING"
}

MAIN_TEXT_FIELD = {COMMENTS: 'body', SUBMISSIONS:'selftext'}

# The Reddit API prefixes IDs to distinguish links to different kinds of objects
ID_PREFIX = {COMMENTS: 't1_', SUBMISSIONS: 't3_'}

# List of columns which overlap between comment and submission data, use for renaming
OVERLAPPING_COLS = ['id', 'author', 'subreddit', 'created_utc']


def get_top_n_counts(dataframe, col='subreddit', n=DEFAULT_TOP_N):
    """Determine the top n most frequent values in a column. Return a dataframe of those values with their counts. Results are ordered by counts, then alphabetical ordering of the values in the column, to break ties at the lower end.
    :param dataframe: Spark DataFrame
    :param col: str, the column to groupBy for counts
    :param n: int, limit results to the top n most frequent values
    """
    return dataframe.groupBy(col).count().orderBy(['count', col], ascending=[0,1]).limit(n)


def filter_top_n(dataframe, top_n_counts, col='subreddit'):
    """Filter the dataframe to only results with those values in top_n_counts.
    Returns a Spark DataFrame that's a subset of the original input dataframe.

    :param dataframe: Spark DataFrame to be filtered
    :param top_n_counts: Spark dataframe with values to be filtered
    :param col: str, Column to use for top n elements
    """
    return dataframe.join(top_n_counts, col, 'inner')


def remove_deleted_authors(dataframe):
    """Filters out comments or submissions that have had the author deleted.
    Returns a Spark DataFrame

    :param dataframe: Spark DataFrame to be filtered
    """
    return dataframe.where(dataframe.author != DELETED)


def remove_rows_with_deleted_text(dataframe, reddit_type):
    """Filters out comments or submissions that have had their text deleted or removed.
    Returns a Spark DataFrame

    :param dataframe: Spark DataFrame to be filtered
    :param reddit_type: str, 'comments' or 'submissions'
    """
    return dataframe.filter(~dataframe[MAIN_TEXT_FIELD[reddit_type]].isin(REMOVED, DELETED))


def print_comparison_stats(original_df, filtered_df, top_n_df):
    """Compares the number of unique subreddits and users in the original and filtered datasets.
    :param original_df: The full unfiltered Spark Dataframe
    :param filtered_df: The original dataframe filtered to include only top
    :param top_n_df: The filtered SparkDataframe
    """
    original_distinct_subreddits = original_df.agg(fn.countDistinct(original_df.subreddit).alias('sr_count')).collect()[0].sr_count
    filtered_distinct_subreddits = top_n_df.agg(fn.countDistinct(top_n_df.subreddit).alias('sr_count')).collect()[0].sr_count
    print("Number of subreddits overall:", original_distinct_subreddits)
    print("Number of subreddits after filtering (sanity check, should match n):", filtered_distinct_subreddits)

    original_comments = original_df.count()
    comments_after_filtering = filtered_df.count()
    print("Number comments before filtering:", original_comments)
    print("Number comments after filtering:", comments_after_filtering)
    print("Percentage of original comments covered:", comments_after_filtering/original_comments)

    original_users = original_df.agg(fn.countDistinct(original_df.author).alias('author_count')).collect()[0].author_count
    filtered_users = filtered_df.agg(fn.countDistinct(filtered_df.author).alias('author_count')).collect()[0].author_count
    print("Number users before filtering:", original_users)
    print("Number users after filtering:", filtered_users)
    print("Percentage of original users covered:", filtered_users/original_users)


def get_spark_dataframe(inputs, spark, reddit_type):
    """
    :param inputs: Paths to Reddit json data
    :param spark: SparkSession
    :param reddit_type: 'comments' or 'submissions'
    """
    return spark.read.format("json"). \
            option("mode", "PERMISSIVE"). \
            option("encoding", "UTF-8"). \
            schema(SCHEMAS[reddit_type]). \
            load(inputs). \
            withColumn(CREATED_UTC, fn.to_timestamp(fn.col(CREATED_UTC)))



def exclude_top_percentage_of_users(user_df, count_col="context_length",  exclude_top_perc=DEFAULT_USER_EXCLUDE):
    """Returns the user dataframe excluding the specified top percentage of users with the most comments.

    :param user_df: Spark DataFrame with a count_col column
    :param count_col: str, the name of the column to use for determining the percentile ranks
    :param exclude_top_perc: float, the percentage of top commenting users to exclude
    """
    if exclude_top_perc == 0.0:
        return user_df

    percentile_col='percentile'
    perc_rank = 1.0 - exclude_top_perc
    result_df = user_df.select("*",
        fn.percent_rank().over(
            Window.partitionBy(). \
                orderBy(user_df[count_col])).alias(percentile_col))
    result_df = result_df.filter(result_df[percentile_col] <= perc_rank)
    result_df.drop(percentile_col)
    return result_df


def aggregate_for_vectorization(dataframe, context_col="author", word_col="subreddit", word_out_col="subreddit_concat", context_len_col="context_length", min_sentence_length=2, exclude_top_perc=DEFAULT_USER_EXCLUDE):
    """Returns a dataframe where each row represents a context with two columns:
    - word_out_col: stores words for each context as white-space delmited string
    - context_len_col: the number of words in each context

    Aggregates data on the context col, using whitespace concatenation
    to combine the values in the word column, dropping rows that
    have contexts smaller than min_sentence_length.

    :param dataframe: Spark dataframe
    :param context_col: str, column to group by
    :param word_col: str, column to concatenate together
    :param word_out_col: str, name of the output column where words for each context are concatenated
    :param context_len_col: str, name of column that stores the number of words concatenated for each context
    :param min_sentence_length: int, the minimum number of comments allowed for a user to be included in the dataset
    :param exclude_top_perc: float, the percentage of top commenting users to exclude
    """
    agg_df = dataframe.groupBy(context_col) \
        .agg(
            fn.concat_ws(" ", fn.collect_list(dataframe[word_col])).alias(word_out_col),
            fn.count(context_col).alias(context_len_col)
        )

    agg_df = exclude_top_percentage_of_users(agg_df, count_col=context_len_col, exclude_top_perc=exclude_top_perc)

    agg_df = agg_df.where(agg_df[context_len_col] >= min_sentence_length)

    return agg_df.drop(context_col)

def prefix_id_column(dataframe, reddit_type=SUBMISSIONS, id_col="id", output_col="fullname_id"):
    """Adds a new column in the dataframe with the appropriate
    Reddit API prefix added.
    :param dataframe: Spark DataFrame to add the columns to
    :param reddit_type: str, 'comments' or 'submissions'
    :param id_col: str, the name of the original unprefixed id column
    :param output_col: str, the name of the column to add
    """
    return dataframe.withColumn(output_col, fn.concat_ws('', fn.lit(ID_PREFIX[reddit_type]), dataframe[id_col]))


def collect_max_context_length(aggregated_df, array_len_col="context_length"):
    """Return the maximum context length according to the array_len_col in the aggregated df.
    :param aggregated_df: Spark dataframe with a column array_len_col storing integers
    :param array_len_col: str, the column to query for the maximum value
    """
    return aggregated_df.agg(fn.max(array_len_col)).head()[0]


def rename_columns(dataframe, columns=None, prefix=COMMENTS):
    """Return a Dataframe of Reddit with the specified columns renamed with the given prefix joined to the original column name with an underscore

    :param dataframe: Spark DataFrame with columns to rename
    :param columns: list of str for column name or defaults to OVERLAPPING_COLS
    :param prefix: str, prefix to prepend to column name
    """
    if columns is None:
        columns = OVERLAPPING_COLS

    result = dataframe
    for c in columns:
        result = result.withColumnRenamed(c, f'{prefix}_{c}')

    return result


def join_submissions_and_comments(submissions_df, comments_df, submission_id_col='fullname_id', comments_link_col='link_id', max_time_delta=None):
    """Returns a DataFrame with comments paired up with their submission using an inner join.

    :param submissions_df: Spark DataFrame containing submissions
    :param comments_df: Spark DataFrame containing comments
    :param submission_id_col: The id column in submissions to use for the join
    :param comments_link_col: The column in the comments dataframe that identifies submissions
    :param max_time_delta: TODO
    """
    renamed_comments = rename_columns(comments_df)
    result_df = submissions_df.join(renamed_comments, submissions_df[submission_id_col] == renamed_comments[comments_link_col])

    if max_time_delta:
        #TODO restict results to
        pass

    return result_df


def community2vec(inputs, spark, reddit_type=COMMENTS, top_n=DEFAULT_TOP_N, min_sentence_length=2, exclude_top_perc=DEFAULT_USER_EXCLUDE, quiet=False):
    """Returns data for training community2vec using skipgrams (users as 'documents/context', subreddits as 'words') as Spark dataframes. Deleted comments are counted when determining the top most frequent values.
    Returns 2 dataframes: counts of subreddits (vocabulary for community2vec), subreddit comments/submissions aggregated into a list for each author

    :param inputs: list of Paths to read JSON Reddit data from
    :param spark: SparkSession
    :param reddit_type: 'comments' or 'submissions'
    :param top_n: int, how many subreddits to consider for c2v, vocab size
    :param min_sentence_length: int, minimum size of context for c2v, min sentence length
    :param quiet: Boolean, true to skip statsitics and plots
    """
    if reddit_type in [COMMENTS, SUBMISSIONS]:
        spark_df = get_spark_dataframe(inputs, spark, reddit_type)
        if not quiet:
            print("Spark dataframe from json:")
            spark_df.show()
        top_n_df = get_top_n_counts(spark_df, n=top_n)
        filtered_df = filter_top_n(spark_df, top_n_df)
        filtered_df = remove_deleted_authors(filtered_df)
        if not quiet:
            print_comparison_stats(spark_df, filtered_df, top_n_df)

        context_word_df = aggregate_for_vectorization(filtered_df,
                            min_sentence_length=min_sentence_length,
                            exclude_top_perc=exclude_top_perc)
        if not quiet:
            max_sentence_length = collect_max_context_length(context_word_df)
            print("Maximum sentence length in data:", max_sentence_length)
    else:
        raise ValueError(f"Reddit data type {reddit_type} is not valid.")

    return top_n_df, context_word_df.drop("context_lenth")


def bag_of_words(spark, comments_paths, submissions_paths, max_time_delta=None, top_n=DEFAULT_TOP_N, type_for_top_n=COMMENTS, quiet=False):
    """Returns the data for training bag of words models in a dataframe.

    :param spark: SparkSession
    :param comments_paths: list of Paths to read JSON Reddit comments from
    :param submissions_paths: list of Paths to read JSON Reddit submissions/posts from
    :param max_time_delta: TODO
    :param top_n: int, how many of the top most popular subreddits to keep
    :param type_for_top_n: 'comments' or 'submissions'
    :param quiet: boolean, set to True for verbose & computationally expensive dataframe comparisons
    """
    comments_df = get_spark_dataframe(comments_paths, spark, COMMENTS)
    submissions_df = get_spark_dataframe(submissions_paths, spark, SUBMISSIONS)
    if type_for_top_n == COMMENTS:
        top_n_df = get_top_n_counts(comments_df, top_n)
    else:
        top_n_df = get_top_n_counts(submissions_df, top_n)

    filtered_comments = remove_deleted_authors(remove_rows_with_deleted_text(filter_top_n(comments_df, top_n_df), COMMENTS))
    filtered_submissions = remove_deleted_authors(remove_rows_with_deleted_text(filter_top_n(submissions_df, top_n_df), SUBMISSIONS))

    if not quiet:
        print("Submissions stats after filtering")
        print_comparison_stats(submissions_df, filtered_submissions, top_n_df)

    filtered_submissions = prefix_id_column(filtered_submissions)
    joined_df = join_submissions_and_comments(filtered_submissions, filtered_comments, max_time_delta=max_time_delta)

    # TODO Preprocess text by replacing tabs & newlines with single whitespace? (Double check if needed, depends on Gensim formats)
    # TODO Group by submission, order comments by creation time, concatenate aggregated comment text
    # TODO Add submission title and selftext to text aggregation



parser = argparse.ArgumentParser(description="Parse Pushshift Reddit data to formats for community2vec and topic modeling.")
parser.add_argument("-q", "--quiet", action='store_true', help="Use to turn off dataset descriptions and extra statistics. This will make pre-processing faster, but skips useful statistics about the datasets.")
subparsers = parser.add_subparsers(dest='subparser_name')
c2v_parser = subparsers.add_parser('c2v', help="Output the subreddits a user has commented on, one user per line, in a compressed CSV format that can be used for training Community2Vec using Gensim's Word2Vec implementation.")
c2v_parser.add_argument("subreddit_counts_csv", help="Desired output path to CSV file for counts of top N subreddits")
c2v_parser.add_argument("context_word_dir", help="Desired output path to directory for a compressed file with subreddits a user commented on, one user per line")
c2v_parser.add_argument("input", nargs='+', help="Paths to input files. They should all be the same type ('comments' or 'submissions')")
c2v_parser.add_argument("-t", "--type", choices=[COMMENTS, SUBMISSIONS], help = "Are these 'comments' or 'submissions' (posts)? Default is 'comments'.", default=COMMENTS)
c2v_parser.add_argument("-n", "--top_n", type=int, default=DEFAULT_TOP_N, help="Use to filter to the top most active subreddits (by number of comments/submssions). Deleted authors/comments/submissions are considered when calculating counts.")
c2v_parser.add_argument("-p", "--exclude_top_user_perc", type=float, default=DEFAULT_USER_EXCLUDE, help="The percentage of top most active users to exclude by number of comments over the time period")

topic_modeling_parser = subparsers.add_parser('bow', help="Output data to a format that can be used for training topic models based on bag of words methods (LDA, tf-idf, etc...)")
topic_modeling_parser.add_argument("--submissions", "-s", nargs="+", help="Path to submissions input data in json format.")
topic_modeling_parser.add_argument("--comments", "-c", nargs="+", help="Path to comments input in json format.")
# TODO: Figure out how to parse timestamp
topic_modeling_parser.add_argument("--max_time_delta", "-d", help="TODO")
topic_modeling_parser.add_argument("-n", "--top_n", type=int, default=DEFAULT_TOP_N, help="Use to filter to the top most active subreddits (by number of comments/submssions). Deleted authors/comments/submissions are considered when calculating counts.")
topic_modeling_parser.add_argument('--type_for_top_n', '-t', default=COMMENTS, choices=[COMMENTS, SUBMISSIONS], help="Is the number of 'comments' or 'submissions' used to determine the top n most popular subreddits?")


if __name__ == "__main__":
    args = parser.parse_args()
    spark = ihop.utils.get_spark_session("IHOP import data", args.quiet)
    if args.subparser_name=='c2v':
        top_n_df, context_word_df = community2vec(args.input, spark,
                                        reddit_type=args.type, top_n=args.top_n,
                                        exclude_top_perc=args.exclude_top_user_perc,
                                        quiet=args.quiet)

        if not args.quiet:
            print("Writing subreddit counts to", args.subreddit_counts_csv)
        top_n_df.toPandas().to_csv(args.subreddit_counts_csv, index=False)

        if not args.quiet:
            print("Writing user contexts to bzip2 in", args.context_word_dir)
        context_word_df.write.option("compression", "bzip2").csv(args.context_word_dir)

    elif args.subparser_name=='bow':
        # TODO Write dataframe to Gensim compatible format
        bag_of_words_df = bag_of_words(spark, args.comments, args.submissions,
                                args.max_time_delta, args.top_n, args.type_for_top_n,
                                args.quiet)
