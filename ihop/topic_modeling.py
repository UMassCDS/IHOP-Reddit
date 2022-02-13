"""Supports training topic models based on the text of Reddit submissions and comments,
such as LDA and clusters of documents based on tf-idf

.. TODO: Implement training of topic models on text: tf-idf-> KMeans, LDA, Hierarchical Dirichlet Processes
.. TODO: Base topic model interface/abstract class defining necessary behaviours
"""
import argparse

import pyspark.sql.functions as fn
from pyspark.ml import Pipeline
from pyspark.ml.feature import CountVectorizer, RegexTokenizer
import pytimeparse

import ihop.import_data

class SparkRedditCorpus:
    """Performs the necessary filtering, grouping and concatenation of columns to produce
    a DataFrame of documents used to train topic models.
    Allows for iterating over documents for training models with Gensim.
    """
    DOC_COL_NAME="document_text"

    def __init__(self, joined_dataset_path, spark, submission_id_col="id",
        submission_text_col="selftext", comments_text_col="body", time_delta_col='time_to_comment_in_seconds',
        max_time_delta=None, min_time_delta=None):
        """
        :param joined_dataset_path: str, path to read csv format dataset with Spark
        :param spark: Spark Session

        """
        raw_dataframe = spark.read(joined_dataset_path)

        filtered_df = ihop.import_data.filter_by_time_between_submission_and_comment(raw_dataframe, max_time_delta, min_time_delta, time_delta_col)

        grouped_submissions = filtered_df.groupBy(submission_id_col).agg(
            fn.first(submission_text_col).alias(submission_text_col),
            fn.concat_ws(" ", comments_text_col).alias(SparkRedditCorpus.DOC_COL_NAME)
        )

        self.document_dataframe = grouped_submissions.select(
            grouped_submissions[submission_id_col],
            fn.concat_ws(" ", grouped_submissions[submission_text_col], SparkRedditCorpus.DOC_COL_NAME).alias(SparkRedditCorpus.DOC_COL_NAME)
        )

    def iterate_over_documents(self, column_name):
        """Yields the values in a particular column for documents
        :param column_name: str, the column to return for each element
        """
        data_iter = self.document_dataframe.rdd.toLocalIterator()
        for row in data_iter:
            yield row[column_name]

    def iterate_over_doc_vectors(self, column_name):
        """Iterate over the a column containing sparse vector outputs, yielding
        the contents of a vector as list of tuples
        """
        for v in self.iterate_over_documents(column_name):
            yield zip(v.indices, v.values)


class SparkTextPreprocessingPipeline:
    """A text pre-processing pipeline that prepares text data for topic modeling
    """

    def __init__(self, input_col, output_col, tokens_col="tokenized", tokenization_pattern="([\p{L}\p{N}#][\p{L}\p{N}\p{Pd}\p{Pc}\p{Pd}.:@]*[\p{L}\p{N}])|[\p{L}\p{N}]", match_gaps=False, **kwargs):
        """Initializes a text preprocessing pipeline with Spark

        :param input_col: str, the name of the column to be input to the pipeline
        :param output_col: str, the name of the column to be output by the pipeline
        :param tokens_col: str, the name for the intermediate column
        :param tokenization_pattern: regex pattern passed to tokenizer
        :param match_gaps: boolean, True if your regex matches gaps between words, False to match tokens
        :param **kwargs: Any arguments to be passed to Spark transformers in the pipeline
        """
        self.tokenizer = RegexTokenizer(inputCol=input_col, outputCol=tokens_col, **kwargs).\
                            setPattern(tokenization_pattern).\
                            setGaps(match_gaps)
        self.count_vectorizer = CountVectorizer(inputCol=tokens_col, outputCol=output_col, **kwargs)
        self.pipeline = Pipeline(stages = [self.tokenizer, self.count_vectorizer])

        self.model = None

    def fit_transform(self, docs_dataframe):
        """Fit the pipeline, then return results of the running transform on the docs_dataframe
        :param docs_dataframe: Spark DataFrame
        """
        self.model = self.pipeline.fit(docs_dataframe)
        return self.model.transform(docs_dataframe)


    def get_id_to_word(self):
        """Returns dictionary mapping indices to word types
        """
        return {i: word for i, word in enumerate(self.count_vectorizer.vocabulary)}


parser = argparse.ArgumentParser(description="Train Gensim topic models from Reddit submission and comment data")
parser.add_argument("input", nargs='+', help="Path to the dataset output by 'ihop.import_data bow'")
parser.add_argument("--model_dir", required=True, help="Path to serialize the trained model to" )
parser.add_argument("--min_term_frequency", "-mtf", default=0, help="Minimum term frequency for terms in each document")
parser.add_argument("--min_doc_frequency", "-mdf", default=0.05, type=float, help="Minimum document frequency")
parser.add_argument("--max_doc_frequency", "-mdf", type=float, default=0.90, help="Maximum document frequency")
parser.add_argument("--max_time_delta", "-d", type=pytimeparse.parse, help="Specify a maximum allowed time between the creation time of a submission creation and when a comment is added. Can be formatted like '1d2h30m2s' or '26:30:02'. If this is not used, all comments are kept for every submission.")