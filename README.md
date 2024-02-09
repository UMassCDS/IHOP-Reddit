[![DOI](https://zenodo.org/badge/436675434.svg)](https://zenodo.org/doi/10.5281/zenodo.10641986)
# RedditMap.Social Modeling
The Center for Data Science repository for working with Reddit data as part of the International Hate Observatory Project, creating the models behind [RedditMap.social](https://redditmap.social).
The `ihop` directory is a python module with submodules that can also be run as command line programs:
- `ihop.import_data`: Uses Spark to import Reddit data from the Pushshift json dumps to formats more easily used for NLP modeling. Run `python -m ihop.import_data --help` for details
- `ihop.community2vec`: Wrappers for training and tuning word2vec to implement community2vec on the Reddit datasets. Run `python -m ihop.community2vec --help` to see options for training community2vec with hyperparameter tuning for best accuracy on the subreddit analogy task.
- `ihop.clustering`: Use to fit sklearn cluster modules with subreddit embeddings or fit Gensim LDA modules on text data.  Run `python -m ihop.clustering --help` to see options.
- `ihop.text_processing`: Text preprocessing utilities for tokenization and vectorizing documents. No script support.
- `ihop.visualizations`: Visualization utilities to create T-SNE projections used the in the cluster viewer applications
- `ihop.utils`: Options to configure logging and Spark environment
- `ihop.resources`: Data resources
	- `ihop.resources.analogies`: Subreddit algebra analogies for tuning community2vec, taken from [social-dimensions](https://github.com/CSSLab/social-dimensions) with minor updates
    - `ihop.resources.collections`: Pre-defined collections of subreddits from the Media Cloud team.

# External Dependencies
- Python >= 3.7
- [Java](https://docs.oracle.com/en/java/javase/17/install/overview-jdk-installation.html) or [OpenJDK](https://openjdk.java.net/install/) (at least version 8). Make sure you have `JAVA_HOME` set appropriately
- (Optional to support faster compression & customize Hadoop config for Spark) [Hadoop](https://hadoop.apache.org) at least version 3.3 is needed for Pyspark to properly decompress the Reddit zst files (see [this issue](https://stackoverflow.com/questions/64607248/configure-spark-on-yarn-to-use-hadoop-native-libraries) or [this one](https://stackoverflow.com/questions/67099204/reading-a-zst-archive-in-scala-spark-native-zstandard-library-not-available)). Install Hadoop and configure the environment variables using [these instructions](https://phoenixnap.com/kb/install-hadoop-ubuntu).
- [unzstd](http://manpages.ubuntu.com/manpages/bionic/man1/unzstd.1.html) and [bzip2](https://www.sourceware.org/bzip2/) are used for recompressing the monthly Reddit dumps to bzip2 format, which Spark and Gensim are more readily able to handle than the zst files.


# Setup and Installation
Note that [Pyspark](https://spark.apache.org/docs/latest/api/python/getting_started/install.html#dependencies) is used for training embeddings and LDA models, so you must have Java installed (at least version 8) and `JAVA_HOME` set appropriately.

Use [Anaconda](https://docs.anaconda.com/anaconda/install/index.html) to create the `ihop` environment by running `conda create -n ihop python=3.9`, install the package and its dependencies using `pip install .`. Note that if you are using a Mac with an M1 or M2 chip, install sklearn and numpy *first* using conda, `conda install numpy==1.21.2 scikit-learn==1.0.1`.  This installs everything needed for just for preprocessing data and training models. If you're going to run the Dash app, install using `pip install .[app]`.

For testing and development tools, install the `ihop` package to be importable for testing, install using `pip install -e .[test,dev]`

# Testing
Unit tests can be run with [`python -m pytest`](https://docs.pytest.org/en/6.2.x/).

# Logging
Logging is configured in `config.json` using the `"logger": {options}` fields. See [Python's logging configuration documentation](https://docs.python.org/3/library/logging.config.html) for details or refer to the example in `ihop/utils.py`.


# Data Processing and Data Version Control
The data, experiments and metrics for this project are tracked using [Data Version Control](https://dvc.org/)(DVC) and backed up in the s3 bucket `s3://ihopmeag`.

Some useful commands are:
- `dvc dag`: View the pipeline stages used to preprocess data and train models
- `dvc metrics show`: See the community2vec model accuracy for each month
- `dvc pull community2vec_models`: Download only the best trained community2vec model for each month of Reddit data. Useful for when you don't need the raw Reddit, such as when you are deploying the subreddit cluster viewer app.

See the DVC documentation for more details.

## Tips for Processing Data
- To reprocess the data without downloading the comments and submissions from Pushshift again, you can use the `dvc repro --downstream` option. There is an example in `scripts/reprocessing.sh`
- The `ihop.import_data` script uses Apache Spark under the hood, so that tasks can be distributed across resources, which will allow customizing processing to whatever resources are on hand. The Spark configuration options are described in the [Spark Documentation](https://spark.apache.org/docs/latest/configuration.html) and can be easily customized by adding a `"spark": {"option_name":"option_value"}` field in the `config.json` file. The default is to use 4G of memory for both the driver and executor, done in `ihop/utils.py`.
- While the `ihop.import_data` script is running locally, you can go to <http://localhost:4040/jobs/> to monitor the Spark jobs progress.


# Subreddit Clustering Application
This code includes a prototype predating the RedditMap.social website. The `app.py` program is a [Dash](https://plotly.com/dash/) app that allows for filtering and visualizing subreddit clusters using the community2vec embeddings.
It expects a JSON configuration file with paths to trained community2vec models, for example:
```
{
    "logger": {<log config, refer to ihop/utils.py for examples>},
    "model_paths": {
        "Model ID displayed in UI dropdown, typically month time frame": "<path to single model output of ihop.community2vec.py model training>",
        "May 2021": ""data/community2vec/RC_2021-05/best_model"
    }
}
```
Run `python app.py --config config.json` to start the application on port 8050, you will be able to navigate to http://localhost:8050/ to see the app running. You can also run using the `--debug` flag to have the application dynamically relaunch on code changes.

The committed `config.json` is configured to load in the best models for each month over a year, from April 2021 through March 2022. To pull the models, run `dvc pull community2vec_models`, assuming you have access to the `s3://ihopmeag` bucket on AWS. See more details on DVC above.

# Citation
If you use this code, please cite [Here Be Livestreams: Trade-offs in Creating Temporal Maps of Reddit](https://arxiv.org/abs/2309.14259) as
```
@misc{partridge2023livestreams,
      title={Here Be Livestreams: Trade-offs in Creating Temporal Maps of Reddit}, 
      author={Virginia Partridge and Jasmine Mangat and Rebecca Curran and Ryan McGrady and Ethan Zuckerman},
      year={2023},
      eprint={2309.14259},
      archivePrefix={arXiv},
      primaryClass={cs.SI}
}
```
This paper was also accepted at [WebSci24](https://websci24.org) with details forthcoming. 


# Known Issues
- Spark can't read the origial zst compressed files from Pushshift, due to the window size being larger than 27 and I didn't know how to change the Spark/Hadoop settings to fix this (see note in [zstd man page](https://manpages.debian.org/unstable/zstd/zstd.1.en.html) and [Stackoverflow: Read zst to pandas](https://stackoverflow.com/questions/61067762/how-to-extract-zst-files-into-a-pandas-dataframe))). Moreover, if you try to read in large .zst files in Spark, you are limited by memory and if there's not enough, the dataframe just gets filled with `null`. The workaround is re-compress the file as a bzip2 before running `ihop.import_data.py`. This takes a long time, but is simple on the command line and `scripts/export_c2v.sh` is provided as a wrapper for the import.
- If you see an error about missing linear algebra acceleration from Spark (`Failed to load implementation from: com.github.fommil.netlib.NativeSystemBLAS`) when running locally, check this [Spark Doc page](https://spark.apache.org/docs/latest/ml-linalg-guide.html) or the [netlib-java Github page](https://github.com/fommil/netlib-java/) for library installation instructions. You can also safely ignore this warning, it just makes Spark a bit slower.
- It would be ideal to keep data in the remote (S3 bucket) and read directly from remote storage using Spark, to avoid keeping the huge Reddit files locally. However, it's difficult to resolve the correct Hadoop dependencies for accessing AWS S3 buckets directly, so I'm punting on this.
- Subreddit counts CSV output from the pre-processing step still counts comments that were deleted/removed. This means it doesn't give accurate information on the number of comments used to train community2vec models. However, the correct counts are logged. Maybe we should take the top n subreddits AFTER filtering out deleted/removed authors?
