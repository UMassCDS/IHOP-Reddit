"""Unit tests for ihop.import_data.py
"""
import os
import pytest

from ihop.import_data import aggregate_for_vectorization, community2vec, get_spark_dataframe, filter_top_n, remove_deleted_authors

FIXTURE_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'test_files')

COMMENTS_DIRS = [os.path.join(FIXTURE_DIR, 'comments1.json'),  os.path.join(FIXTURE_DIR, 'comments2.json')]

SUBMISSIONS_DIR = [os.path.join(FIXTURE_DIR, 'submissions.json')]

@pytest.fixture
def comments(spark):
    return get_spark_dataframe(COMMENTS_DIRS, spark, "comments")

@pytest.fixture
def submissions(spark):
    return get_spark_dataframe(SUBMISSIONS_DIR, spark, "submissions")


def test_get_spark_dataframe_comments(comments):
    assert len(comments.columns) == 8
    assert comments.count() == 3


def test_get_spark_dataframe_submissions(submissions):
    assert len(submissions.columns) == 14
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


def test_aggregate_for_vectorization(spark):
    data = [{'author':'auth1', 'subreddit':"r/scifi"},
            {'author':'auth1', 'subreddit':"r/fantasy"},
            {'author':'auth1', 'subreddit':"r/books"},
            {'author':'auth1', 'subreddit':"r/fantasy"},
            {'author':'auth2', 'subreddit':"r/movies"},
            {'author':'auth2', 'subreddit':"r/personalfinance"}
            ]
    df = spark.createDataFrame(data)
    aggregate_result = [x.subreddit for x in aggregate_for_vectorization(df).collect()]
    assert len(aggregate_result) == 2
    assert "r/scifi r/fantasy r/books r/fantasy" in aggregate_result
    assert "r/movies r/personalfinance" in aggregate_result

def test_community2vec(spark):
    top_n_subreddits, user_contexts = community2vec(COMMENTS_DIRS, spark)
    top_n_list = sorted(top_n_subreddits.collect(), key=lambda x: x.subreddit)
    print(top_n_list)
    assert len(top_n_list) == 2
    assert top_n_list[0].subreddit == "NBA2k"
    assert top_n_list[0]['count'] == 1
    assert top_n_list[1].subreddit == "dndnext"
    assert top_n_list[1]['count'] == 2

    user_contexts_list = [x.subreddit for x in user_contexts.collect()]
    assert len(user_contexts_list) == 2

    assert "NBA2k" in user_contexts_list
    assert "dndnext" in user_contexts_list
