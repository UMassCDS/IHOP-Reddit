"""Supports Spark pipelines for working with text data

.. TODO: set submission timeframe start and end dates
.. TODO: optionally support IDF as last step in the pipeline
.. TODO: SparkRedditCorpus to pandas or numpy functions
.. TODO: Does this actually need to be a script - can argparse options get moved over to clustering module?
"""
import logging
import os

import pyspark.sql.functions as fn
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import CountVectorizer, CountVectorizerModel, RegexTokenizer, IDF


import ihop.import_data

logger = logging.getLogger(__name__)

DEFAULT_DOC_COL_NAME = "document_text"
VECTORIZED_COL_NAME = "vectorized"


class SparkCorpus:
    """Performs the necessary filtering, grouping and concatenation of columns to produce
    a DataFrame of documents used to train topic models.
    Allows for iterating over documents for training models with Gensim.
    """

    def __init__(self, dataframe, document_col_name=DEFAULT_DOC_COL_NAME, id_col="id"):
        self.document_dataframe = dataframe
        self.document_col_name = document_col_name
        self.id_col = id_col

    def get_column_iterator(self, column_name, use_id_col=False):
        """Returns an iterator over data in a particular column of the corpus that contains text or numerical data
        :param column_name: str, name of column
        :param use_id_col: boolean, true to also return the id col during iteration
        """
        return SparkCorpusIterator(self.document_dataframe, column_name, id_col=self.id_col, is_return_id=use_id_col)

    def get_vectorized_column_iterator(self, column_name=VECTORIZED_COL_NAME, use_id_col=False):
        """Returns an iterator over data in a particular column of the corpus
        that contains vector data as zipped tuples
        :param column_name: str, name of column
        :param use_id_col: boolean, true to also return the id col during iteration
        """
        return SparkCorpusIterator(self.document_dataframe, column_name, is_vectorized=True, is_return_id=use_id_col)

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
            grouped_submissions[category_col],
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


class SparkCorpusIterator:
    """An iterator object over a particular column of a SparkCorpus.
    This is required for Gensim models such as LDA, which need iterator objects,
    not generator functions.
    """

    def __init__(self, corpus_df, column_name, is_vectorized=False, is_return_id=False, id_col="id"):
        """
        :param corpus: A SparkDataframe
        :param column_name: The name of the column to retrieve from the corpus
        :param is_vectorized: Set to true if the column stores vectorized documents as opposed to text or numerical data
        """
        self.corpus_df = corpus_df
        self.column_name = column_name
        self.is_vectorized = is_vectorized
        self.id_col = id_col
        self.is_return_id = is_return_id

    def __len__(self):
        return self.corpus_df.count()

    def __iter__(self):
        # Reset iteration each time
        spark_rdd_iter = self.corpus_df.rdd.toLocalIterator()
        for row in spark_rdd_iter:
            data = row[self.column_name]
            if self.is_vectorized:
                result = list(zip(data.indices, data.values))
            else:
                result = data

            if self.is_return_id:
                yield row[self.id_col], result
            else:
                yield result


class SparkTextPreprocessingPipeline:
    """A text pre-processing pipeline that prepares text data for topic modeling
    """
    PIPELINE_OUTPUT_NAME = "SparkTextProcessingPipeline"
    MODEL_OUTPUT_NAME = "SparkTextProcessingModel"

    def __init__(self, input_col=DEFAULT_DOC_COL_NAME, output_col=VECTORIZED_COL_NAME, tokens_col="tokenized",
                 tokenization_pattern="([\p{L}\p{N}#@][\p{L}\p{N}\p{Pd}\p{Pc}\p{S}\p{P}]*[\p{L}\p{N}])|[\p{L}\p{N}]",
                 match_gaps=False, toLowercase=True,
                 maxDF=0.95, minDF=0.5, minTF=0.0, binary=False, useIDF=False):
        """Initializes a text preprocessing pipeline with Spark.
        Note: The tokenization pattern throws away punctuation pretty aggresively, is probably throwing away emojis

        :param input_col: str, the name of the column to be input to the pipeline
        :param output_col: str, the name of the column to be output by the pipeline
        :param tokens_col: str, the name for the intermediate column
        :param tokenization_pattern: regex pattern passed to tokenizer
        :param match_gaps: boolean, True if your regex matches gaps between words, False to match tokens
        :param toLowercase: boolean, True to covert characters to lowercase before tokenizing
        :param maxDF: int or float, maximum document frequency expressed as a float percentage of documents in the corpus or a integer number of documents. Throw away terms that occur in more than that number of documents.
        :param minDF: int or float, minimum number of documents a term must be in as a percentage of documents in the corpus or an integer number of documents. Throw away terms that occur in fewer than that number of docs.
        :param minTF: int or float, ignore terms with frequency (float, fraction of document's token count) or count less than the given value for each document (affects transform only, not fitting)
        :param binary: boolean, Set to True for binary term document flags, rather than term frequency counts
        :param useIDF: boolean, set to True to use inverse document frequency smoothing of counts.
        """
        tokenizer = RegexTokenizer(inputCol=input_col, outputCol=tokens_col, toLowercase=toLowercase).\
            setPattern(tokenization_pattern).\
            setGaps(match_gaps)

        count_vectorizer = CountVectorizer(
            inputCol=tokens_col, outputCol=output_col,
            maxDF=maxDF, minDF=minDF, minTF=minTF, binary=binary)
        pipeline_stages = [tokenizer, count_vectorizer]

        if useIDF:
            count_vectorized_col = "count_vectorized"
            count_vectorizer.setOutputCol(count_vectorized_col)
            idf_stage = IDF(inputCol=count_vectorized_col,
                            outputCol=output_col)
            pipeline_stages.append(idf_stage)

        self.pipeline = Pipeline(stages=pipeline_stages)

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
