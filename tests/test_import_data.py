"""Unit tests for ihop.import_data.py
"""
import os

import pytest

from ihop.import_data import *


@pytest.fixture
def comments(spark, fixture_dir):
    return get_spark_dataframe(
        [
            os.path.join(fixture_dir, "comments1.json"),
            os.path.join(fixture_dir, "comments2.json"),
        ],
        spark,
        "comments",
    )


@pytest.fixture
def submissions(spark, fixture_dir):
    return get_spark_dataframe(
        [os.path.join(fixture_dir, "submissions.json")], spark, "submissions"
    )


@pytest.fixture
def context_dataframe(spark):
    data = [
        {"author": "auth1", "subreddit": "scifi"},
        {"author": "auth1", "subreddit": "fantasy"},
        {"author": "auth1", "subreddit": "books"},
        {"author": "auth1", "subreddit": "fantasy"},
        {"author": "auth2", "subreddit": "movies"},
        {"author": "auth2", "subreddit": "personalfinance"},
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
    assert result.subreddit == "dndnext"
    assert result.count == 1


def test_filter_top_n(comments, spark):
    top_n_counts = spark.createDataFrame([{"subreddit": "dndnext", "count": 1}])
    filtered_df = filter_top_n(comments, top_n_counts)
    as_list = sorted(filtered_df.collect(), key=lambda x: x.author)
    assert len(as_list) == 2
    assert as_list[0].subreddit == "dndnext"
    assert as_list[1].subreddit == "dndnext"
    assert as_list[0].author == "[deleted]"
    assert as_list[1].author == "sampleauth2"


def test_remove_deleted_authors(comments):
    filtered = sorted(
        remove_deleted_authors(comments).collect(), key=lambda x: x.author
    )
    assert len(filtered) == 2
    assert filtered[0].author == "sampleauth1"
    assert filtered[1].author == "sampleauth2"


def test_exclude_top_percentage_of_users(spark):
    data = [
        {"subreddit_concat": "AskReddit", "context_length": 1},
        {"subreddit_concat": "aww pictures", "context_length": 2},
        {"subreddit_concat": "nba nba celtics", "context_length": 3},
        {"subreddit_concat": "books books fantasy movies", "context_length": 4},
        {
            "subreddit_concat": "wallstreetbets stonks personalfinance wallstreetbets wallstreetbets",
            "context_length": 5,
        },
    ]
    agg_df = spark.createDataFrame(data)
    result_list = exclude_top_percentage_of_users(
        agg_df, exclude_top_perc=0.2
    ).collect()
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
    assert (
        collect_max_context_length(
            aggregate_for_vectorization(context_dataframe, exclude_top_perc=0.0)
        )
        == 4
    )


def test_community2vec(spark, fixture_dir):
    top_n_subreddits, user_contexts = community2vec(
        [
            os.path.join(fixture_dir, "comments1.json"),
            os.path.join(fixture_dir, "comments2.json"),
        ],
        spark,
        min_sentence_length=0,
        exclude_top_perc=0.0,
    )
    top_n_list = sorted(top_n_subreddits.collect(), key=lambda x: x.subreddit)
    print(top_n_list)
    assert len(top_n_list) == 2
    assert top_n_list[0].subreddit == "NBA2k"
    assert top_n_list[0]["count"] == 1
    assert top_n_list[1].subreddit == "dndnext"
    assert top_n_list[1]["count"] == 2

    user_contexts_list = [x.subreddit_concat for x in user_contexts.collect()]
    assert len(user_contexts_list) == 2

    assert "NBA2k" in user_contexts_list
    assert "dndnext" in user_contexts_list


def test_remove_deleted_comments(spark):
    data = [
        {"author": "a1", "body": "[removed]"},
        {"author": "a2", "body": "[deleted]"},
        {"author": "a3", "body": "This wasn't deleted"},
    ]
    df = spark.createDataFrame(data)
    result = remove_deleted_text(df, "comments").collect()
    assert len(result) == 1
    assert result[0].author == "a3"


def test_remove_deleted_submissions(spark):
    data = [
        {"author": "a1", "selftext": "[removed]"},
        {"author": "a2", "selftext": "[deleted]"},
        {"author": "a3", "selftext": "This wasn't deleted"},
    ]
    df = spark.createDataFrame(data)
    result = remove_deleted_text(df, "submissions").collect()
    assert len(result) == 1
    assert result[0].author == "a3"


def test_prefix_id_column(submissions):
    df = prefix_id_column(submissions)
    ids_list = [x.fullname_id for x in df.collect()]
    assert len(ids_list) == 3
    assert set(ids_list) == set(["t3_6xauyf", "t3_6xauys", "t3_6xauyh"])


def test_rename_columns(comments):
    renamed = rename_columns(comments)
    assert renamed.columns == [
        "comments_id",
        "parent_id",
        "comments_score",
        "link_id",
        "comments_author",
        "comments_subreddit",
        "body",
        "comments_created_utc",
    ]


def test_join_submissions_and_comments(spark):
    submissions_data = [
        {
            "id": "a12",
            "fullname_id": "t3_a12",
            "selftext": "This is my first post!",
            "title": "Saying hi!",
        },
        {
            "id": "b12",
            "fullname_id": "t3_b12",
            "selftext": "It's very cute",
            "title": "Check out this dog video",
        },
        {
            "id": "c12",
            "fullname_id": "t3_c12",
            "selftext": "",
            "title": "Another tiktok video",
        },
    ]
    submissions_df = spark.createDataFrame(submissions_data)
    comments_data = [
        {"link_id": "t3_b12", "body": "so cute much wow", "id": "abc"},
        {"link_id": "t3_b12", "body": "what kind of dog is it", "id": "efg"},
        {
            "link_id": "t3_z89",
            "body": "some comment on a random submission",
            "id": "hij",
        },
        {"link_id": "t3_c12", "body": "tiktok dances are the best", "id": "klm"},
    ]
    comments_df = spark.createDataFrame(comments_data)
    joined = join_submissions_and_comments(submissions_df, comments_df).collect()
    expected_joins = set([("b12", "abc"), ("b12", "efg"), ("c12", "klm")])
    assert set([(r.id, r.comments_id) for r in joined]) == expected_joins
    assert len(joined) == 3


def test_filter_time_stamp(spark, comments):
    submissions_data = [
        {
            "id": "73hbg4",
            "fullname_id": "t3_73hbg4",
            "selftext": "A question about world building",
            "title": "D&D world creation question!",
            "created_utc": 1506815950,
        },
        {
            "id": "73gee6",
            "fullname_id": "t3_73gee6",
            "selftext": "An opinion about basketball",
            "title": "Basketball",
            "created_utc": 1506815600,
        },
    ]
    submissions_df = spark.createDataFrame(submissions_data)
    joined_with_filter = filter_by_time_between_submission_and_comment(
        join_submissions_and_comments(submissions_df, comments), max_time_delta=100
    ).collect()
    assert len(joined_with_filter) == 1
    assert joined_with_filter[0].id == "73hbg4"
    assert joined_with_filter[0].comments_id == "98765a"
    assert joined_with_filter[0].time_to_comment_in_seconds == 50


def test_filter_out_top_users(spark):
    test_df = spark.createDataFrame(
        [
            {"author": "author1"},
            {"author": "author1"},
            {"author": "top"},
            {"author": "top"},
            {"author": "top"},
            {"author": "author2"},
            {"author": "author3"},
        ]
    )
    filtered = filter_out_top_users(test_df, exclude_top_perc=0.25)
    filtered.show()
    assert filtered.columns == ["author"]
    results_list = sorted(filtered.collect(), key=lambda x: x.author)
    assert len(results_list) == 4
    results_list[0] == "author1"
    results_list[1] == "author1"
    results_list[2] == "author2"
    results_list[3] == "author3"

