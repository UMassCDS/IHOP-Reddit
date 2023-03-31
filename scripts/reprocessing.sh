#!/bin/bash
# Reprocesses data and retrains community2vec and Kmeans models for each month using DVC without
# downloading the Reddit zip files from Pushshift again

months=(2022-12 2023-01 2023-02)

for m in "${months[@]}"
do
    echo $m
    #dvc repro --downstream --force prep_community2vec_data@$m
    dvc repro prep_community2vec_data@$m
    dvc repro community2vec_models@$m
    dvc repro kmeans_cluster_models@$m
    dvc repro tsne_visualizations@$m
done
