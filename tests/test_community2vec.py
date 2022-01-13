import os

import gensim
import numpy as np

import ihop.community2vec as c2v

FIXTURE_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'test_files')

VOCAB_CSV = os.path.join(FIXTURE_DIR, "vocab.csv")
SAMPLE_SENTENCES = os.path.join(FIXTURE_DIR, "community2vec_sentences.txt")
SAMPLE_COMPRESSED = os.path.join(FIXTURE_DIR, "test_import_data_output.bz2")

def test_get_vocab():
    expected_vocab = {"AskReddit":4, "news":2, "politics":2, "The_Donald":1, "nba":3,
                      "worldnews":1, "funny":4, "teenagers":3, "pics":3, "hockey":3}
    vocab = c2v.get_vocabulary(VOCAB_CSV)
    assert vocab == expected_vocab

def test_init_gensim_community2vec(spark):
    c2v_model = c2v.GensimCommunity2Vec.init_with_spark(spark, c2v.get_vocabulary(VOCAB_CSV), SAMPLE_SENTENCES, vector_size = 25, epochs=2, batch_words=100, alpha=0.04)
    assert c2v_model.num_users == 4
    assert c2v_model.max_comments == 9
    assert c2v_model.epochs == 2
    assert c2v_model.contexts_path == SAMPLE_SENTENCES
    assert c2v_model.w2v_model.epochs == 2
    assert c2v_model.w2v_model.batch_words == 100
    assert c2v_model.w2v_model.alpha == 0.04
    assert len(c2v_model.w2v_model.wv["AskReddit"]) == 25


def test_gensim_community2vec_train_no_errors():
    c2v_model = c2v.GensimCommunity2Vec(c2v.get_vocabulary(VOCAB_CSV), SAMPLE_SENTENCES, 9, 4, vector_size=25, epochs=2)
    train_result = c2v_model.train()
    assert type(train_result) == tuple


def test_gensim_community2vec_compressed_sentences():
    c2v_model = c2v.GensimCommunity2Vec({"dndnext":1, "NBA2k":1}, SAMPLE_COMPRESSED, 1, 2, epochs=2)
    assert c2v_model.num_users == 2
    assert c2v_model.max_comments == 1
    train_result = c2v_model.train()
    assert type(train_result) == tuple


def test_save_load_trained_community2vec_model(tmp_path):
    save_path = str(tmp_path)
    c2v_model = c2v.GensimCommunity2Vec(c2v.get_vocabulary(VOCAB_CSV), SAMPLE_SENTENCES, 9, 4, epochs=2, alpha=0.05)
    c2v_model.train()
    vector = c2v_model.w2v_model.wv['AskReddit']
    c2v_model.save(save_path)

    loaded_model = c2v.GensimCommunity2Vec.load(save_path)
    assert loaded_model.num_users == 4
    assert loaded_model.max_comments == 9
    assert loaded_model.epochs == 2
    assert loaded_model.w2v_model.alpha == 0.05
    assert np.all(loaded_model.w2v_model.wv["AskReddit"] == vector)


def test_save_vectors(tmp_path):
    save_path = str(tmp_path /"vectors.gz")
    c2v_model = c2v.GensimCommunity2Vec(c2v.get_vocabulary(VOCAB_CSV), SAMPLE_SENTENCES,  9, 4, epochs=2, alpha=0.05, vector_size=25)
    random_vector = c2v_model.w2v_model.wv["AskReddit"]

    c2v_model.save_vectors(save_path)

    loaded_vectors = gensim.models.KeyedVectors.load(save_path)

    assert loaded_vectors.vector_size == 25
    assert len(loaded_vectors.key_to_index) == 10
    assert np.all(loaded_vectors["AskReddit"] == random_vector)
