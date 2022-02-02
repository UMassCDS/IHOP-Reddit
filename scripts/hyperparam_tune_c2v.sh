#!/bin/bash
# Perform hyperparameter tuning to train community2vec models

DATA_ROOT="data/community2vec/sample_data"
CONTEXTS="${DATA_ROOT}/user_contexts"
VOCAB="${DATA_ROOT}/subreddit_counts.csv"
MODEL_DIR="${DATA_ROOT}/models"


python -m ihop.community2vec --contexts $CONTEXTS --vocab_csv $VOCAB --output_dir $MODEL_DIR --param_grid '{"alpha": [0.05, 0.01], "vector_size":[150], "sample":[0.005, 0.05], "negative":[20,40]}'