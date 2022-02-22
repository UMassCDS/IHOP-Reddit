"""Unit tests for ihop.topic_modeling.py
"""
import pytest

from ihop.text_processing import SparkCorpus, SparkTextPreprocessingPipeline


@pytest.fixture
def joined_reddit_dataframe(spark):
    test_data = [
        {'id': 's1', 'selftext': 'Post 1 text.', 'title': 'MY FIRST POST!!!!', 'comments_id': 'c1',
            'body': "Ain't this hard to tokenize: #hashtag, yo-yo www.reddit.com?", 'time_to_comment_in_seconds': 600, 'subreddit': 'AskReddit'},
        {'id': 's1', 'selftext': 'Post 1 text.', 'title': 'MY FIRST POST!!!!', 'comments_id': 'c2',
            'body': "@someone some.one@email.com", 'time_to_comment_in_seconds': 10, 'subreddit': 'AskReddit'},
        {'id': 's2', 'selftext': '', 'title': 'Look @ this cute animal!', 'comments_id': 'c3',
            'body': "aww- - adorable...", 'time_to_comment_in_seconds': 100, 'subreddit': 'aww'}
    ]

    return spark.createDataFrame(test_data)


@pytest.fixture
def simple_vocab_df(spark):
    simple_input = [{"id": "a1", "document_text": "a a a"},
                    {"id": "b2", "document_text": "b b b"},
                    {"id": "c3", "document_text": "a b"}]
    return spark.createDataFrame(simple_input)


@pytest.fixture
def corpus(joined_reddit_dataframe):
    return SparkCorpus.init_from_joined_dataframe(joined_reddit_dataframe)


def test_init_spark_reddit_corpus(joined_reddit_dataframe):
    corpus = SparkCorpus.init_from_joined_dataframe(
        joined_reddit_dataframe)
    assert corpus.document_dataframe.count() == 2
    assert corpus.document_dataframe.columns == ['id', 'document_text']
    doc_set = set([d for d in corpus.get_column_iterator('document_text')])
    assert "MY FIRST POST!!!! Post 1 text. @someone some.one@email.com Ain't this hard to tokenize: #hashtag, yo-yo www.reddit.com?" in doc_set
    assert "Look @ this cute animal!  aww- - adorable..." in doc_set


def test_init_spark_reddit_corpus_with_timedeltas(joined_reddit_dataframe):
    corpus = SparkCorpus.init_from_joined_dataframe(
        joined_reddit_dataframe, min_time_delta=20, max_time_delta=500)
    assert corpus.document_dataframe.count() == 1
    doc_set = set([d for d in corpus.get_column_iterator('document_text')])
    assert "Look @ this cute animal!  aww- - adorable..." in doc_set


def test_spark_text_processing_pipeline(corpus):
    pipeline = SparkTextPreprocessingPipeline(
        "document_text", "vectorized", maxDF=1000, minDF=0.0)
    transformed_corpus = pipeline.fit_transform(corpus.document_dataframe)
    assert transformed_corpus.columns == [
        "id", "document_text", "tokenized", "vectorized"]
    results = sorted(transformed_corpus.collect(), key=lambda x: x.id)
    text_1 = "my first post post 1 text @someone some.one@email.com ain't this hard to tokenize #hashtag yo-yo www.reddit.com".split()
    text_2 = "look this cute animal aww adorable".split()
    assert results[0].tokenized == text_1
    assert results[1].tokenized == text_2
    vocabulary = set(text_1).union(set(text_2))
    index = pipeline.get_id_to_word()
    assert len(index) == len(vocabulary)
    assert set(index.values()) == vocabulary
    assert set(index.keys()) == set(range(len(vocabulary)))


def test_index_words(simple_vocab_df):
    pipeline = SparkTextPreprocessingPipeline("document_text", "vectorized")
    corpus = SparkCorpus(pipeline.fit_transform(simple_vocab_df))

    vector_docs = list(corpus.get_vectorized_column_iterator("vectorized"))
    inv_index = pipeline.get_word_to_id()
    a_id = inv_index['a']
    b_id = inv_index['b']
    assert list(vector_docs[0]) == [(a_id, 3)]
    assert list(vector_docs[1]) == [(b_id, 3)]
    assert set(vector_docs[2]) == set([(a_id, 1), (b_id, 1)])


def test_save_load_spark_pipeline(simple_vocab_df, tmp_path):
    pipeline_to_save = SparkTextPreprocessingPipeline("document_text")
    pipeline_to_save.fit_transform(simple_vocab_df)
    pipeline_to_save.save(tmp_path)

    pipeline_to_load = SparkTextPreprocessingPipeline.load(tmp_path)
    assert pipeline_to_save.get_id_to_word() == pipeline_to_load.get_id_to_word()
