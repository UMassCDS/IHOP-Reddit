"""Trains community2vec models (word2vec where users are context and words are subreddits)

Inputs:
1) CSV of subreddits (other columns optional) to use as vocabulary

2) Multiple CSVs with user,words used as document contexts

.. TODO Implement community2vec as described in https://www.tensorflow.org/tutorials/text/word2vec with negative sampling or using gensim

.. TODO Account for downsampling of skipgrams as described in https://github.com/BIU-NLP/word2vecf/blob/master/word2vecf.c#L421 (I think this can be done in tensorflow with the sampling_table) or with the sample kw option in skipgrams

# TODO Considerations for cross validation
"""
import csv
import logging

import gensim
import pyspark.sql.functions as fn

# TODO Logging should be configurable, but for now just turn it on for Gensim
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def get_vocabulary(vocabulary_csv, has_header=True, token_index=0, count_index=1):
    """Return vocabulary as dictionary str->int of frequency counts

    :param vocabulary_csv: path to csv vocabulary
    """
    vocab = {}
    with open(vocabulary_csv) as vocab_in:
        vocab_reader = csv.reader(vocab_in)
        if has_header:
            next(vocab_reader)
        for row in vocab_reader:
            vocab[row[token_index]] = int(row[count_index])

    return vocab


class GensimCommunity2Vec:
    """Implements Community2Vec Skip-gram with negative sampling (SGNS) using the gensim Word2Vec model.
    Determines the appropriate window-size and sets vocabulary according to the
    filtered subreddits.
    """

    def __init__(self, spark, vocab_dict, contexts_path, vector_size=150, negative=20, sample=0, alpha=0.025, min_alpha=0.0001, seed=1, epochs=5, batch_words=10000, workers=3):
        """
        Instantiates a gensim Word2Vec model for Community2Vec
        :param spark: SparkSession, needed for determining window size and number of users for learning rate decay. Most default values match gensim Word2Vec.

        :param vocab_dict: dict, str->int storing frequency counts of the vocab elements.
        :param contexts_path: Path to a text file storing the subreddits a user commented on, one user per line. Can be compressed as a bzip2 or gzip.
        :param vector_size: int, embedding size, passed to gensim Word2Vec
        :param negative: int, how many 'noise words' should be drawn, passed to gensim Word2Vec
        :param sample: float, the threshold for configuring which higher-frequency words are randomly downsampled, useful range is (0, 1e-5), passed to gensim Word2Vec, defaults to 0
        :param alpha: float, initial learning rate, passed to gensim Word2Vec
        :param min_alpha: float, minimum for learning rate decay
        :param seed: int, randomSeed for initializing embeddings, passed to gensim Word2Vec
        :param epochs: int, Number of iterations over the corpus, passed to gensim Word2Vec
        :param batch_words: int, Target size (in words) for batches of examples passed to worker threads
        :param workers: int, number of worker threads for training
        """
        self.contexts_path = contexts_path
        self.context_df = spark.read.csv(contexts_path)
        self.max_comments = self.__get_context_window()
        self.num_users = self.__get_num_users()
        self.epochs = epochs
        self.w2v_model = gensim.models.word2vec.Word2Vec(vector_size=vector_size, min_count=0, window=self.max_comments, sg=1, hs=0, negative=negative, sample=sample, seed=seed, alpha=alpha, epochs=epochs, batch_words=batch_words, min_alpha=min_alpha, workers=workers)
        self.w2v_model.build_vocab_from_freq(vocab_dict)

    def __get_context_window(self):
        """Return the maximum number of comments a user had in the dataset.
        This will become the window size for the Word2Vec model.
        """
        max_agg_df = self.context_df.select(fn.split("_c0", " ").alias("subreddit_list")).select("subreddit_list", fn.size("subreddit_list").alias("num_comments")).agg(fn.max("num_comments"))
        return max_agg_df.head()[0]

    def __get_num_users(self):
        """Returns the number of users in the data, AKA number of contexts/sentences for W2V
        """
        return self.context_df.count()

    def train(self, **kwargs):
        """Trains the word2vec model. Returns the result from gensim.
        :param **kwargs: passed to gensim Word2Vec.train()
        """
        train_result = self.w2v_model.train(gensim.models.word2vec.PathLineSentences(self.contexts_path), total_examples=self.num_users, epochs=self.epochs, **kwargs)
        return train_result


