#!/bin/bash
# Runs data transformation required for LDA from import data to final LDA model training
COMMENTS_IN=data/raw_data/comments/RC_2021-*.bz2
SUBMISSIONS_IN=data/raw_data/submissions/RS_2021-05.bz2

DATESTAMP="03212022"

JOINED_DATA="data/bagOfWords/2021-05_joined_submissions_comments_5percentTopUsersExcludedFromComments_${DATESTAMP}.parquet"

VECTORIZED_DATA="data/2021-05_reddit_threads_${DATESTAMP}"

MIN_DF=0.001
MAX_DF=0.65
MIN_TIME_DELTA="3s"
MAX_TIME_DELTA="72h"


echo "Joining submissions and comments"
python -m ihop.import_data bow $JOINED_DATA -p 0.05 --comments $COMMENTS_IN --submissions $SUBMISSIONS_IN

echo "Preprocessing text"
python -m ihop.text_processing $JOINED_DATA -o $VECTORIZED_DATA --min_doc_freq $MIN_DF --max_doc_freq $MAX_DF -x $MAX_TIME_DELTA -m $MIN_TIME_DELTA

echo "Training Spark LDA model"
python -m ihop.clustering $VECTORIZED_DATA -o ${VECTORIZED_DATA}/sparklda_$DATESTAMP -d SparkVectorized -c sparklda -p '{"maxIter":50, "optimizeDocConcentration": true}'

echo "Training Gensim LDA model"
python -m ihop.clustering $VECTORIZED_DATA -o ${VECTORIZED_DATA}/gensimlda_$DATESTAMP -d SparkVectorized -c gensimlda