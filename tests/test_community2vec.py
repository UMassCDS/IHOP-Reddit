import os

import gensim
import numpy as np
import pytest

import ihop.community2vec as c2v


@pytest.fixture
def vocab_csv(fixture_dir):
    return os.path.join(fixture_dir, "vocab.csv")


@pytest.fixture
def sample_sentences(fixture_dir):
    return os.path.join(fixture_dir, "community2vec_sentences.txt")


@pytest.fixture
def sample_compressed(fixture_dir):
    return os.path.join(fixture_dir, "test_import_data_output.bz2")


def test_get_vocab(vocab_csv):
    expected_vocab = {
        "AskReddit": 4,
        "news": 2,
        "politics": 2,
        "The_Donald": 1,
        "nba": 3,
        "worldnews": 1,
        "funny": 4,
        "teenagers": 3,
        "pics": 3,
        "hockey": 3,
    }
    vocab = c2v.get_vocabulary(vocab_csv)
    assert vocab == expected_vocab


def test_init_gensim_community2vec(spark, vocab_csv, sample_sentences):
    c2v_model = c2v.GensimCommunity2Vec.init_with_spark(
        spark,
        c2v.get_vocabulary(vocab_csv),
        sample_sentences,
        vector_size=25,
        epochs=2,
        batch_words=100,
        alpha=0.04,
    )
    assert c2v_model.num_users == 4
    assert c2v_model.max_comments == 9
    assert c2v_model.epochs == 2
    assert c2v_model.contexts_path == sample_sentences
    assert c2v_model.w2v_model.epochs == 2
    assert c2v_model.w2v_model.batch_words == 100
    assert c2v_model.w2v_model.alpha == 0.04
    assert len(c2v_model.w2v_model.wv["AskReddit"]) == 25
    expected_params = {
        "num_users": 4,
        "max_comments": 9,
        "contexts_path": sample_sentences,
        "epochs": 2,
        "vector_size": 25,
        "skip_gram": 1,
        "hierarchical_softmax": 0,
        "negative": 20,
        "ns_exponent": 0.75,
        "alpha": 0.04,
        "min_alpha": 0.0001,
        "seed": 1,
        "batch_words": 100,
        "sample": 0,
    }
    assert c2v_model.get_params_as_dict() == expected_params


def test_gensim_community2vec_train_no_errors(vocab_csv, sample_sentences):
    # Test that training doesn't error out
    c2v_model = c2v.GensimCommunity2Vec(
        c2v.get_vocabulary(vocab_csv), sample_sentences, 9, 4, vector_size=25, epochs=2
    )

    train_result = c2v_model.train()

    assert type(train_result) == tuple

    # We should also be able to get some reasonable nearest neighbor results
    hockey_neighbors = c2v_model.get_nearest_neighbors("hockey", 5)
    assert len(hockey_neighbors) == 5
    assert "nba" in hockey_neighbors
    assert "funny" in hockey_neighbors



def test_gensim_community2vec_compressed_sentences(sample_compressed):
    c2v_model = c2v.GensimCommunity2Vec(
        {"dndnext": 1, "NBA2k": 1}, sample_compressed, 1, 2, epochs=2
    )
    assert c2v_model.num_users == 2
    assert c2v_model.max_comments == 1
    train_result = c2v_model.train()
    assert type(train_result) == tuple


def test_save_load_trained_community2vec_model(tmp_path, vocab_csv, sample_sentences):
    save_path = str(tmp_path)
    c2v_model = c2v.GensimCommunity2Vec(
        c2v.get_vocabulary(vocab_csv), sample_sentences, 9, 4, epochs=2, alpha=0.05
    )
    c2v_model.train()
    vector = c2v_model.w2v_model.wv["AskReddit"]
    c2v_model.save(save_path)

    loaded_model = c2v.GensimCommunity2Vec.load(save_path)
    assert loaded_model.num_users == 4
    assert loaded_model.max_comments == 9
    assert loaded_model.epochs == 2
    assert loaded_model.w2v_model.alpha == 0.05
    assert np.all(loaded_model.w2v_model.wv["AskReddit"] == vector)


def test_save_vectors(tmp_path, vocab_csv, sample_sentences):
    save_path = str(tmp_path / "vectors.gz")
    c2v_model = c2v.GensimCommunity2Vec(
        c2v.get_vocabulary(vocab_csv),
        sample_sentences,
        9,
        4,
        epochs=2,
        alpha=0.05,
        vector_size=25,
    )
    random_vector = c2v_model.w2v_model.wv["AskReddit"]

    c2v_model.save_vectors(save_path)

    loaded_vectors = gensim.models.KeyedVectors.load(save_path)

    assert loaded_vectors.vector_size == 25
    assert len(loaded_vectors.key_to_index) == 10
    assert np.all(loaded_vectors["AskReddit"] == random_vector)


def test_grid_search_init(vocab_csv):
    grid_trainer = c2v.GridSearchTrainer(
        vocab_csv,
        "dummy_context_path",
        2,
        10,
        "dummy_model_out",
        {
            "alpha": [0.02, 0.05],
            "vector_size": [150, 200],
            "negative": [20, 40],
            "sample": [0.002],
        },
    )
    assert grid_trainer.num_models == 8


def test_empty_grid_init(vocab_csv):
    grid_trainer = c2v.GridSearchTrainer(
        vocab_csv, "dummy_context_path", 2, 10, "dummy_model_out", {}
    )
    assert grid_trainer.num_models == 1
    assert grid_trainer.param_grid == c2v.GridSearchTrainer.DEFAULT_PARAM_GRID


def test_grid_search_model_id(vocab_csv):
    grid_trainer = c2v.GridSearchTrainer(
        vocab_csv, "dummy_context_path", 2, 10, "dummy_model_out", {}
    )
    model_id = grid_trainer.get_model_id(
        {"alpha": 0.02, "vector_size": 25, "sample": 0.002}
    )
    assert model_id == "alpha0.02_sample0.002_vectorSize25"


def test_grid_search_expand_param_grid_to_list(vocab_csv):
    grid_trainer = c2v.GridSearchTrainer(
        vocab_csv,
        "dummy_context_path",
        2,
        10,
        "dummy_model_out",
        {
            "alpha": [0.02, 0.05],
            "vector_size": [150, 200],
            "negative": [20, 40],
            "sample": [0.002],
        },
    )

    model_params = grid_trainer.expand_param_grid_to_list()
    assert len(model_params) == 8
    for param_dict in model_params:
        assert param_dict.keys() == set(["alpha", "vector_size", "negative", "sample"])
        assert param_dict["alpha"] in [0.02, 0.05]
        assert param_dict["vector_size"] in [150, 200]
        assert param_dict["negative"] in [20, 40]
        assert param_dict["sample"] == 0.002


def test_grid_search_train(tmp_path, vocab_csv, sample_sentences):
    model_dir = str(tmp_path / "models")
    grid_trainer = c2v.GridSearchTrainer(
        vocab_csv,
        sample_sentences,
        4,
        9,
        model_dir,
        {"alpha": [0.02], "vector_size": [25], "negative": [20, 40]},
        keep_all=True,
    )
    best_acc, best_model = grid_trainer.train(epochs=1)

    assert best_model is not None
    output_models = os.listdir(model_dir)
    assert len(output_models) == 3

    # Check best model folder has all necessary outputs
    best_model_dir = tmp_path / "models" / "best_model"
    assert best_model_dir.exists() and best_model_dir.is_dir()
    assert (best_model_dir / "metrics.json").exists()
    assert (best_model_dir / "keyedVectors").exists()
    assert (best_model_dir / "word2vec.pickle").exists()
    assert (best_model_dir / "parameters.json").exists()

    model_df = grid_trainer.model_analogy_results_as_dataframe()
    assert model_df.shape == (2, 17)
