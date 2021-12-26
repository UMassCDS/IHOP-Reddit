"""Unit tests for ihop.import_data.py
"""
import os
import pytest

from ihop.import_data import SUBMISSIONS, get_spark_dataframe, filter_top_n, remove_deleted_authors

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
    assert len(comments.columns) == 14
    assert comments.count() == 3


def test_get_spark_dataframe_submissions(submissions):
    assert len(submissions.columns) == 16
    assert submissions.count() == 3


def test_filter_top_n(comments):
    filtered_df = filter_top_n(comments, n=1)
    as_list = sorted(filtered_df.collect(), key = lambda x: x.author)
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












