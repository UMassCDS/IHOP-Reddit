"""Trains community2vec models (word2vec where users are context and words are subreddits)

Inputs:
1) CSV of subreddits (other columns optional) to use as vocabulary

2) Multiple CSVs with user,words used as document contexts

.. TODO Implement community2vec as described in https://www.tensorflow.org/tutorials/text/word2vec with negative sampleing

.. TODO Account for downsampling of skipgrams as described in https://github.com/BIU-NLP/word2vecf/blob/master/word2vecf.c#L421 (I think this can be done in tensorflow with the sampling_table)

.. TODO Incrementally read in data from CSV for efficient memory usage, see https://www.tensorflow.org/guide/data
"""