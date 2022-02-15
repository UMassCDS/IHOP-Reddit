"""Supports Spark pipelines for working with text data

.. TODO: set submission timeframe start and end dates
.. TODO: optionally support IDF as last step in the pipeline
.. TODO: SparkRedditCorpus to pandas or numpy functions
.. TODO: Does this actually need to be a script - can argparse options get moved over to clustering module?
"""
import argparse
import logging
import os

import pyspark.sql.functions as fn
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import CountVectorizer, CountVectorizerModel, RegexTokenizer
import pytimeparse

import ihop.import_data

logger = logging.getLogger(__name__)

DEFAULT_DOC_COL_NAME = "document_text"
VECTORIZED_COL_NAME = "vectorized"


class SparkRedditCorpus:
    """Performs the necessary filtering, grouping and concatenation of columns to produce
    a DataFrame of documents used to train topic models.
    Allows for iterating over documents for training models with Gensim.
    """

    def __init__(self, dataframe, document_col_name=DEFAULT_DOC_COL_NAME):
        self.document_dataframe = dataframe
        self.document_col_name = document_col_name

    def iterate_over_documents(self, column_name):
        """Yields the values in a particular column for documents
        :param column_name: str, the column to return for each element
        """
        data_iter = self.document_dataframe.rdd.toLocalIterator()
        for row in data_iter:
            yield row[column_name]

    def iterate_over_doc_vectors(self, column_name=VECTORIZED_COL_NAME):
        """Iterate over the a column containing sparse vector outputs, yielding
        the contents of a vector as list of tuples
        """
        for v in self.iterate_over_documents(column_name):
            yield zip(v.indices, v.values)

    def save(self, output_path):
        """Save the corpus to a parquet file
        """
        self.document_dataframe.write.parquet(output_path)

    @classmethod
    def init_from_joined_dataframe(cls, raw_dataframe, submission_id_col="id",
                                   submission_text_col="selftext", submission_title_col="title",
                                   comments_text_col="body", category_col="subreddit", time_delta_col='time_to_comment_in_seconds',
                                   max_time_delta=None, min_time_delta=None):
        """Instantiate a SparkRedditCorpus using the a DataFrame of submission, comment pairs

        :param raw_dataframe: Spark DataFrame each row represents a submission, comment pair
        :param submission_id_col: str, submission id column to group by for aggregating documents
        :param submission_text_col: str, text on the submission appears in this column
        :param submission_title_col: str, column containing submission title text
        :param comments_text_col: str, column containing comment text
        :param category_col:, str, column for categorical grouping of documents, useful to keep around for entropy metrics,etc..
        :param time_delta_col: str, the column storing the time a comment occured after a submission in
        :param max_time_delta: int, the maximum time (typically in seconds) a comment is allowed to occur after a submission
        :param time_time_delta: int, the minimum time (typically in seconds) a comment is allowed to occur after a submission
        """
        filtered_df = ihop.import_data.filter_by_time_between_submission_and_comment(
            raw_dataframe, max_time_delta, min_time_delta, time_delta_col)

        grouped_submissions = filtered_df.\
            orderBy(time_delta_col).\
            groupBy(submission_id_col).\
            agg(
                fn.first(category_col).alias(category_col),
                fn.first(submission_text_col).alias(submission_text_col),
                fn.first(submission_title_col).alias(submission_title_col),
                fn.concat_ws(" ", fn.collect_list(comments_text_col)).alias(
                    DEFAULT_DOC_COL_NAME)
            )

        document_dataframe = grouped_submissions.select(
            grouped_submissions[submission_id_col],
            fn.concat_ws(" ",
                         grouped_submissions[submission_title_col],
                         grouped_submissions[submission_text_col],
                         grouped_submissions[DEFAULT_DOC_COL_NAME]).
            alias(DEFAULT_DOC_COL_NAME)
        )

        return cls(document_dataframe)

    @classmethod
    def load(cls, spark, df_path, document_col=DEFAULT_DOC_COL_NAME, format='parquet', **kwargs):
        """Returns a SparkRedditCorpus from the given data path.
        The data can be in any format readable by spark.

        :param spark: SparkSession
        :param df_path: str, path to the data file
        :param document_col: str, the column storing document text, where each row in the dataframe is a document
        :param format: str, data format option passed to Spark
        :param kwargs: any other options to pass to Spark when reading data
        """
        cls(spark.read.load(df_path, format=format, **kwargs), document_col)


class SparkTextPreprocessingPipeline:
    """A text pre-processing pipeline that prepares text data for topic modeling
    """
    PIPELINE_OUTPUT_NAME = "SparkTextProcessingPipeline"
    MODEL_OUTPUT_NAME = "SparkTextProcessingModel"

    def __init__(self, input_col, output_col=VECTORIZED_COL_NAME, tokens_col="tokenized", tokenization_pattern="([\p{L}\p{N}#@][\p{L}\p{N}\p{Pd}\p{Pc}\p{S}\p{P}]*[\p{L}\p{N}])|[\p{L}\p{N}]", match_gaps=False, **kwargs):
        """Initializes a text preprocessing pipeline with Spark

        :param input_col: str, the name of the column to be input to the pipeline
        :param output_col: str, the name of the column to be output by the pipeline
        :param tokens_col: str, the name for the intermediate column
        :param tokenization_pattern: regex pattern passed to tokenizer
        :param match_gaps: boolean, True if your regex matches gaps between words, False to match tokens
        :param **kwargs: Any arguments to be passed to Spark transformers in the pipeline
        """
        tokenizer = RegexTokenizer(inputCol=input_col, outputCol=tokens_col, **kwargs).\
            setPattern(tokenization_pattern).\
            setGaps(match_gaps)
        count_vectorizer = CountVectorizer(
            inputCol=tokens_col, outputCol=output_col, **kwargs)
        self.pipeline = Pipeline(
            stages=[tokenizer, count_vectorizer])

        self.model = None

    def fit_transform(self, docs_dataframe):
        """Fit the pipeline, then return results of the running transform on the docs_dataframe
        :param docs_dataframe: Spark DataFrame
        """
        self.model = self.pipeline.fit(docs_dataframe)
        return self.model.transform(docs_dataframe)

    def get_id_to_word(self):
        vocab = {}
        if self.model is not None:
            vectorizers = [s for s in self.model.stages if isinstance(
                s, CountVectorizerModel)]
            vocab = {i: word for i,
                     word in enumerate(vectorizers[0].vocabulary)}
        return vocab

    def get_word_to_id(self):
        return {v: k for k, v in self.get_id_to_word().items()}

    def save(self, save_dir):
        """Saves the model and pipeline to the specified directory

        :param save_dir: Directory to save the model and pipeline
        """
        os.makedirs(save_dir, exist_ok=True)
        self.pipeline.save(os.path.join(save_dir, self.PIPELINE_OUTPUT_NAME))
        if self.model is not None:
            self.model.save(os.path.join(save_dir, self.MODEL_OUTPUT_NAME))

    @classmethod
    def load(cls, load_dir):
        """Loads a SparkTextPreprocessingPipeline from the specified directory
        :param load_dir: Directory to load the model and pipeline from
        """
        result = cls("inplaceholder", "outplaceholder")
        result.pipeline = Pipeline.load(
            os.path.join(load_dir, cls.PIPELINE_OUTPUT_NAME))
        result.model = PipelineModel.load(
            os.path.join(load_dir, cls.MODEL_OUTPUT_NAME))

        return result


parser = argparse.ArgumentParser(
    description="Pre-process text and train document-based topic or cluster models from Reddit threads")
parser.add_argument("input", nargs='+',
                    help="Path to the dataset output by 'ihop.import_data bow'")
parser.add_argument("--model_dir", required=True,
                    help="Path to serialize the trained model to")
parser.add_argument("--min_term_frequency", default=0,
                    help="Minimum term frequency for terms in each document")
parser.add_argument("--min_doc_frequency", default=0.05,
                    type=float, help="Minimum document frequency")
parser.add_argument("--max_doc_frequency", type=float,
                    default=0.90, help="Maximum document frequency")
parser.add_argument("--max_time_delta", "-x", type=pytimeparse.parse,
                    help="Specify a maximum allowed time between the creation time of a submission creation and when a comment is added. Can be formatted like '1d2h30m2s' or '26:30:02'. If this is not used, all comments are kept for every submission.")
parser.add_argument("--min_time_delta", "-m", type=pytimeparse.parse,
                    help="Optionally specify a minimum allowed time between the creation time of a submission creation and when a comment is added. Can be formatted like '1d2h30m2s' or '26:30:02'. If this is not used, all comments are kept for every submission.")
