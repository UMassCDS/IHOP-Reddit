#!/bin/bash
# Reprocesses data and retrains community2vec and all later dependencies
# without downloading the Reddit zip files from Pushshift again

months=(2021-09 2021-10 2021-11 2021-12 2022-01 2022-02 2022-03)

for m in "${months[@]}"
do
    echo $m
    dvc repro --downstream --force prep_community2vec_data@$m
done