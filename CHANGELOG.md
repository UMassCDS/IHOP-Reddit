# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Changed

### Fixed

### Added

### Removed
- Removed Unity documentation

## [2.0.0]
### Changed
- Removed prefilled anti-immigrant subreddits selected in subreddit clustering app dropdown. Now the dropdown is initially empty.
- Package installation switched to use setup.cfg and pyproject.toml rather than setup.py
- Slurm python version changed from 3.8 to 3.9
- Separate Dash app requirements in requirements.txt from ihop package requirments in setup.py
- T-SNE visualiations for each community2vec model are generated using a DVC pipeline step and written to CSV in order to speed up interaction times and avoid re-generating the projection each time a new month is selected
- Pandas version updated from 1.3.4 to 1.3.5
- Notebooks with many changes to support visualizations and analysis for ICWSM paper

### Fixed
- Fixed bug in app where wrong clusters are highlighted after changing selected subreddits and using the "Highlight selected clusters" button
- ihop.clustering.ClusteringModel predict method now works for sklearn.cluster.AgglomerativeClustering, which doesn't have the same predict API as KMeans and AffinityPropagation
- Filter out subreddits that are actually user profile pages, denoted by starting with "u_", in both community2vec and bow pre-processing steps in ihop.import_data

### Added
- Feature for selecting community2vec models from different time frames in subreddit clustering app using a dropdown
- Added notebook demonstrating automatic subreddit or cluster labeling using differential text analysis
- Added vocabSize option to Spark pipeline in text_preprocessing.py
- Added pyproject.toml in switch to setup.cfg
- Bash script for running community2vec experiments using DVC
- DVC stages for downloading data and training community2vec models
- Output analogy accuracy metrics and community2vec model parameters for tracking experiments with DVC
- Github workflow for running tests on pull requests to the main branch
- MANIFEST.in to include analogy and subreddit collection resource files in package
- Annotation results from coherence task added in data/kmeans_annotation_task_data and data/average_agglomerative_annoation_task
- Notebook for computing inter-annotator agreement in notebooks/inter_rater_agreements.ipynb
- Functions that computing metrics to compare clusterings/partitions of data points added to ihop/clustering.py and unit tests
- Function for computing contingency tables between clusters/partitions of data points added to ihop/clustering.py and unit tests
- Maximum match algorithm to align two cluster across two partitions of the same datapoints added to ihop/clustering.py and unit tests
- Method to easy get nearest neighbors for data points from community2vec models added to ihop.community2vec.GensimCommunity2Vec
- Added notebook to demonstrating comparisons of community2vec models and clustering stability over time, notebooks/clustering_stability_metrics_and_visualizations.ipynb
- DVC stage for producing agglomerative clustering models for annotation
- MIT License

### Removed
- community2vec hyperparameter tuning bash script
- Removed requirements.txt, environment.yml in switch to setup.cfg

## [1.0.0] - 2022-05-24
### Changed
- Restructured analogies resources, moving from Community2Vec module
- t-sne projection of embeddings can have more than 2 components
- Parsing configuration file returns the spark config, logging config and full configuration dictionary
- Fixed a bug around reading configuration file using argparse

### Added
- Dash app to browse and visualize community2vec clusters
- ihop/visualization.py for supporting the Dash app
- Resource for collections of subreddits

## [0.2.0] - 2022-03-31
### Changed
- SparkSession configured via dictionary in ihop/utils.py
- Logging for ihop package set up via config file (with a default fallback) rather than the logging basicConfig function
- Config file reader in ihop/utils.py to configure logging and the SparkSession
- Object oriented approach to clustering and topic models in ihop/clustering.py
- Used black code formatter to make code more readable
- Updated README to reflect new module additions

### Added
- ihop/text_processing.py which vectorizes documents using Spark pipelines and outputs suitable formats for Gensim
- SparkLDAModel class for training LDA models using spark in ihop/clustering.py
- Class in ihop/utils.py for serializing numpy float32 to json
- Option for importing data to bag-of-words format in ihop/import_data.py
- Created ihop/text_processing.py for tokenization and tf-idf options when working with text data
- GensimLDAModel class for training LDAMulticore models in ihop/clustering.py
- In ihop/clustering.py added main method with argparse options for training clustering and topic models
- Unit tests for ihop/text_processing.py and ihop/clustering.py
- Serialization for all models in ihop/clustering.py supported

## [0.1.0] - 2022-02-14
### Added
- This changelog
- Import Reddit data to a Gensim Word2Vec compatible CSV format
- Full implementation of training community2vec models on Reddit data with hyperparameter tuning
- Analogies for evaluating community2vec
