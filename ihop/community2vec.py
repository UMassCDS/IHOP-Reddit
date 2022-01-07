"""Trains community2vec models (word2vec where users are context and words are subreddits)

Inputs:
1) CSV of subreddits (other columns optional) to use as vocabulary

2) Multiple CSVs with user,words used as document contexts

.. TODO Implement community2vec as described in https://www.tensorflow.org/tutorials/text/word2vec with negative sampling or using gensim

.. TODO Account for downsampling of skipgrams as described in https://github.com/BIU-NLP/word2vecf/blob/master/word2vecf.c#L421 (I think this can be done in tensorflow with the sampling_table) or with the sample kw option in skipgrams

# TODO Considerations for cross validation
"""
import csv


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

