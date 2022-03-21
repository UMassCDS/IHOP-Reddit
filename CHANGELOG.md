# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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