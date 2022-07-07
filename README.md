# IHOP
The Center for Data Science repository for the International Hate Observatory Project.
The `ihop` directory is a python module with submodules that can also be run as command line programs:
- `ihop.import_data`: Uses Spark to import Reddit data from the Pushshift json dumps to formats more easily used for topic modeling. Run `python -m ihop.import_data --help` for details
- `ihop.community2vec`: Wrappers for training and tuning word2vec to implement community2vec on the Reddit datasets. Run `python -m ihop.community2vec --help` to see options for training community2vec with hyperparameter tuning for best accuracy on the subreddit analogy task.
- `ihop.clustering`: Use to fit sklearn cluster modules with subreddit embeddings or fit Gensim LDA modules on text data.  Run `python -m ihop.clustering --help` to see options.
- `ihop.text_processing`: Text preprocessing utilities for tokenization and vectorizing documents. No script support.
- `ihop.resources`: Data resources
	- `ihop.resources.analogies`: Subreddit algebra analogies for tuning community2vec, taken from [social-dimensions](https://github.com/CSSLab/social-dimensions) with minor updates

## Subreddit Clustering Application
The `app.py` program is a [Dash](https://plotly.com/dash/) app that allows for filtering and visualizing subreddit clusters using the community2vec embeddings.
It expects a JSON configuration file with paths to trained community2vec models, for example:
```
{
    "logger": {<log config, refer to ihop.utils.py for examples>},
    "model_paths": {
        "Model ID displayed in UI dropdown, typically month time frame": "<path to single model output of ihop.community2vec.py model training>",
        "May 2021": ""data/community2vec/RC_2021-05/best_model"
    }
}
```
Run `python app.py --config config.json` to start the application on port 8050, you will be able to navigate to http://localhost:8050/ to see the app running. You can also run using the `--debug` flag to have the application dynamically relaunch on code changes.

The committed `config.json` is configured to load in the best models for each month over a year, from April 2021 through March 2022. To pull the models, run `dvc pull community2vec_models`, assuming you have access to the `s3://ihopmeag` bucket on AWS. See more details on DVC below.


# External Dependencies
- Python >= 3.7
- [Java](https://docs.oracle.com/en/java/javase/17/install/overview-jdk-installation.html) or [OpenJDK](https://openjdk.java.net/install/) (at least version 8). Make sure you have `JAVA_HOME` set appropriately
- (Optional to support faster compression & customize Hadoop config for Spark) [Hadoop](https://hadoop.apache.org) at least version 3.3 is needed for Pyspark to properly decompress the Reddit zst files (see [this issue](https://stackoverflow.com/questions/64607248/configure-spark-on-yarn-to-use-hadoop-native-libraries) or [this one](https://stackoverflow.com/questions/67099204/reading-a-zst-archive-in-scala-spark-native-zstandard-library-not-available)). Install Hadoop and configure the environment variables using [these instructions](https://phoenixnap.com/kb/install-hadoop-ubuntu).
- [unzstd](http://manpages.ubuntu.com/manpages/bionic/man1/unzstd.1.html) and [bzip2](https://www.sourceware.org/bzip2/) are used for recompressing the monthly Reddit dumps to bzip2 format, which Spark and Gensim are more readily able to handle than the zst files.


# Setup and Installation
Note that [Pyspark](https://spark.apache.org/docs/latest/api/python/getting_started/install.html#dependencies) is used for training embeddings and LDA models, so you must have Java installed (at least version 8) and `JAVA_HOME` set appropriately.

Use [Anaconda](https://docs.anaconda.com/anaconda/install/index.html) to create the `ihop` environment by running `conda create -n ihop python=3.9`, install the package and its dependencies using `pip install .`. This installs everything needed for just for preprocessing data and training models. If you're going to run the Dash app, install using `pip install .[app]`.

For testing and development tools, install the `ihop` package to be importable for testing, install using `pip install -e .[test]`

## Unity
To install the packages on [Unity](https://unity.rc.umass.edu/docs/#modules/using/), you will need to load the python and miniconda modules, then install the package as usual with conda or pip:
```
module load miniconda/4.8.3

conda create --yes --name ihop python=3.8
conda activate ihop
pip install .
```

In order to use the packages on Unity, you will load the miniconda and java modules, then activate the conda environment you created earlier:
```
module load miniconda/4.8.3
module load java/11.0.2
conda activate ihop
```

An example of using these modules to submit community2vec jobs to slurm on Unity is given in `scripts/hyperparam_tune_c2v_slurm.sh`

# Data Verion Control
The data, experiments and metrics for this project are tracked using [Data Version Control](https://dvc.org/)(DVC) and backed up in the s3 bucket `s3://ihopmeag`.

Some useful commands are:
- `dvc dag`: View the pipeline stages used to preprocess data and train models
- `dvc metrics show`: See the community2vec model accuracy for each month
- `dvc pull community2vec_models`: Download only the best trained community2vec model for each month of Reddit data. Useful for when you don't need the raw Reddit, such as when you are deploying the subreddit cluster viewer app.

See the DVC documentation for more details.

# Testing
Unit tests can be run with [`python -m pytest`](https://docs.pytest.org/en/6.2.x/).

# Known Issues
- Spark can't read the origial zst compressed files from Pushshift, due to the window size being larger than 27 and I didn't know how to change the Spark/Hadoop settings to fix this (see note in [zstd man page](https://manpages.debian.org/unstable/zstd/zstd.1.en.html) and [Stackoverflow: Read zst to pandas](https://stackoverflow.com/questions/61067762/how-to-extract-zst-files-into-a-pandas-dataframe))). Moreover, if you try to read in large .zst files in Spark, you are limited by memory and if there's not enough, the dataframe just gets filled with `null`. The workaround is re-compress the file as a bzip2 before running `ihop.import_data.py`. This takes a long time, but is simple on the command line and `scripts/export_c2v.sh` is provided as a wrapper for the import.
- Sports analogies in `ihop/resources/analogies` only contains sports leagues & teams from North America
- `uni_to_city.csv` only contains universities in English-speaking countries and French Canada
- If you see an error about missing linear algebra acceleration from Spark (`Failed to load implementation from: com.github.fommil.netlib.NativeSystemBLAS`) when running locally, check this [Spark Doc page](https://spark.apache.org/docs/latest/ml-linalg-guide.html) or the [netlib-java Github page](https://github.com/fommil/netlib-java/) for library installation instructions. You can also safely ignore this warning, it just makes Spark a bit slower.
- It would be ideal to keep data in the remote (S3 bucket) and read directly from remote storage using Spark, to avoid keeping the huge Reddit files locally. However, it's difficult to resolve the correct Hadoop dependencies for accessing AWS S3 buckets directly, so I'm punting on this.
- TSNE visualizations are slow to train and slow down the subreddit cluster viewer app significantly. It would be good to produce the TSNE visualizations as an additional output of training the community2vec model or as its own pipeline step.

