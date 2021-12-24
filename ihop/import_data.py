"""Loads Reddit json data to a Spark dataframe, which can be filtered using
SQL-like operations, operated on using Spark's ML library or exported to pandas/sklearn formats.

..TODO Consider how best to match up most active subreddits by number of submissions and number of comments
"""
import argparse

from pyspark.sql import SparkSession

COMMENTS="comments"
SUBMISSIONS="submissions"

# See https://github.com/pushshift/api for details
SCHEMAS = {
        COMMENTS: "id STRING, parent_id STRING, score INTEGER, author_flair_css_class STRING, author_flair_text STRING, link_id STRING, author STRING, subreddit STRING, body STRING, edited INTEGER, gilded STRING, controversiality INTEGER, created_utc STRING, distinguished STRING",
        SUBMISSIONS: "author STRING, author_flair_css_class STRING, author_flair_text STRING, created_utc STRING, distinguished STRING, domain STRING, edited INTEGER, gilded STRING, id STRING, is_self BOOLEAN, over_18 BOOLEAN, score INTEGER, selftext STRING, title STRING, url STRING, subreddit STRING"
        }

def filter_top_n_subreddits(n=1000):
    """Filters results by the top-n most active subreddits by number of unique posts/comments dataset.
    ..TODO
    """
    pass

def get_spark_dataframe(inputs, spark, reddit_type):
    """
    :param inputs: Paths to Reddit json data
    :param spark: SparkSession
    :param reddit_type: "comments" or "submissions"
    """
    return spark.read.format("json").option("encoding", "UTF-8").schema(SCHEMAS[reddit_type]).load(inputs)

def main(output_path, inputs, reddit_type=COMMENTS):
    """
    :param output_path: Where to write out the parquet data format
    :param inputs: Paths to read JSON Reddit data from
    :param reddit_type: 'comments' or 'submissions'
    """
    # create the Spark session
    spark = SparkSession.builder.appName("IHOP import data").getOrCreate()

    if reddit_type in [COMMENTS, SUBMISSIONS]:
        spark_df = get_spark_dataframe(inputs, spark, reddit_type)
        spark_df.show()
    else:
        raise ValueError(f"Reddit data type {reddit_type} is not valid.")


parser = argparse.ArgumentParser(description="Parse Pushshift Reddit data to Spark parquet dataframe")
parser.add_argument("output", help="Path to output file")
parser.add_argument("input", nargs='+', help="Paths to input files. They should all be the same type ('comments' or 'submissions')")
parser.add_argument("-t", "--type", choices=[COMMENTS, SUBMISSIONS], help = "Are these 'comments' or 'submissions' (posts)?")
parser.add_argument("-n", "--top_n", type=int, default=1000, help="Use to filter to the top most activae subreddits (by number of comments/submssions).")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args.output, args.input, args.type)