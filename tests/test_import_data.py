"""Unit tests for ihop.import_data.py
"""
import os
import pytest

from ihop.import_data import *

FIXTURE_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'test_files')

COMMENTS_DIRS = [os.path.join(FIXTURE_DIR, 'comments1.json'),  os.path.join(FIXTURE_DIR, 'comments2.json')]

SUBMISSIONS_DIR = [os.path.join(FIXTURE_DIR, 'submissions.json')]

@pytest.fixture
def comments(spark):
    return get_spark_dataframe(COMMENTS_DIRS, spark, "comments")

@pytest.fixture
def submissions(spark):
    return get_spark_dataframe(SUBMISSIONS_DIR, spark, "submissions")

@pytest.fixture
def context_dataframe(spark):
    data = [{'author':'auth1', 'subreddit':"scifi"},
            {'author':'auth1', 'subreddit':"fantasy"},
            {'author':'auth1', 'subreddit':"books"},
            {'author':'auth1', 'subreddit':"fantasy"},
            {'author':'auth2', 'subreddit':"movies"},
            {'author':'auth2', 'subreddit':"personalfinance"}
            ]
    return spark.createDataFrame(data)


def test_get_spark_dataframe_comments(comments):
    assert len(comments.columns) == 8
    assert comments.count() == 3


def test_get_spark_dataframe_submissions(submissions):
    assert len(submissions.columns) == 8
    assert submissions.count() == 3


def get_top_n_counts(comments):
    result = get_top_n_counts(comments, n=1).collect()
    assert len(result) == 1
    assert result.subreddit == 'dndnext'
    assert result.count == 1


def test_filter_top_n(comments, spark):
    top_n_counts = spark.createDataFrame([{'subreddit':'dndnext','count':1}])
    filtered_df = filter_top_n(comments, top_n_counts)
    as_list = sorted(filtered_df.collect(), key = lambda x: x.subreddit)
    assert len(as_list) == 2
    assert as_list[0].subreddit == 'dndnext'
    assert as_list[1].subreddit == 'dndnext'
    assert as_list[0].author == '[deleted]'
    assert as_list[1].author == 'sampleauth2'


def test_remove_deleted_authors(comments):
    filtered = sorted(remove_deleted_authors(comments).collect(), key = lambda x: x.author)
    assert len(filtered) == 2
    assert filtered[0].author == 'sampleauth1'
    assert filtered[1].author == 'sampleauth2'


def test_exclude_top_percentage_of_users(spark):
    data = [{'subreddit_concat':'AskReddit', 'context_length':1},
            {'subreddit_concat':'aww pictures', 'context_length':2},
            {'subreddit_concat':'nba nba celtics', 'context_length':3},
            {'subreddit_concat':'books books fantasy movies', 'context_length':4},
            {'subreddit_concat':'wallstreetbets stonks personalfinance wallstreetbets wallstreetbets', 'context_length':5},
            ]
    agg_df = spark.createDataFrame(data)
    result_list = exclude_top_percentage_of_users(agg_df, exclude_top_perc=0.2).collect()
    result_counts = [x.context_length for x in result_list]
    assert list(range(1, 5)) == sorted(result_counts)


def test_aggregate_for_vectorization(context_dataframe):
    agg_df = aggregate_for_vectorization(context_dataframe, exclude_top_perc=0.0)
    assert agg_df.columns == ["subreddit_concat", "context_length"]
    aggregate_result = [x.subreddit_concat for x in agg_df.collect()]
    assert len(aggregate_result) == 2
    assert "scifi fantasy books fantasy" in aggregate_result
    assert "movies personalfinance" in aggregate_result


def test_collect_max_context_length(context_dataframe):
    assert collect_max_context_length(aggregate_for_vectorization(context_dataframe, exclude_top_perc=0.0)) == 4


def test_community2vec(spark):
    top_n_subreddits, user_contexts = community2vec(COMMENTS_DIRS, spark, min_sentence_length=0, exclude_top_perc=0.0)
    top_n_list = sorted(top_n_subreddits.collect(), key=lambda x: x.subreddit)
    print(top_n_list)
    assert len(top_n_list) == 2
    assert top_n_list[0].subreddit == "NBA2k"
    assert top_n_list[0]['count'] == 1
    assert top_n_list[1].subreddit == "dndnext"
    assert top_n_list[1]['count'] == 2

    user_contexts_list = [x.subreddit_concat for x in user_contexts.collect()]
    assert len(user_contexts_list) == 2

    assert "NBA2k" in user_contexts_list
    assert "dndnext" in user_contexts_list


def test_remove_deleted_comments(spark):
    data = [{'author':'a1', 'body':'[removed]'},
            {'author':'a2', 'body':'[deleted]'},
            {'author':'a3', 'body': "This wasn't deleted"}]
    df = spark.createDataFrame(data)
    result = remove_rows_with_deleted_text(df, 'comments').collect()
    assert len(result) == 1
    assert result[0].author == 'a3'


def test_remove_deleted_submissions(spark):
    data = [{'author':'a1', 'selftext':'[removed]'},
            {'author':'a2', 'selftext':'[deleted]'},
            {'author':'a3', 'selftext': "This wasn't deleted"}]
    df = spark.createDataFrame(data)
    result = remove_rows_with_deleted_text(df, 'submissions').collect()
    assert len(result) == 1
    assert result[0].author == 'a3'


def test_prefix_id_column(submissions):
    df = prefix_id_column(submissions)
    ids_list = [x.fullname_id for x in df.collect()]
    assert len(ids_list) == 3
    assert set(ids_list) == set(['t3_6xauyf', 't3_6xauys', 't3_6xauyh'])

def test_join_submissions_and_comments(spark):
    submissions_data = [{'id': 'a12', 'fullname_id':'t3_a12', 'selftext':'This is my first post!', 'title':'Saying hi!'},
        {'id':'b12', 'fullname_id':'t3_b12', 'selftext':"It's very cute", 'title':'Check out this dog video'},
        {'id': 'c12', 'fullname_id': 't3_c12', 'selftext':'', 'title':'Another tiktok video'}]
    submissions_df = spark.createDataFrame(submissions_data)
    comments_data = [{'link_id':'t3_b12', 'body':'so cute much wow', 'id':'abc'},
                     {'link_id':'t3_b12', 'body':'what kind of dog is it', 'id':'efg'},
                     {'link_id':'t3_z89', 'body':'some comment on a random submission', 'id':'hij'},
                     {'link_id':'t3_c12', 'body':'tiktok dances are the best', 'id':'klm'}]
    comments_df = spark.createDataFrame(comments_data)
    joined = join_submissions_and_comments(submissions_df, comments_df).collect()
    assert len(joined) == 3


def test_join_submissions_and_comments_with_timestamp(spark):
    # TODO
    pass