# IHOP
The Center for Data Science repository for the International Hate Observatory Project.
The `ihop` directory is a python module with submodules that can also be run as command line programs:
- `ihop.import_data`: Uses Spark to import Reddit data from the Pushshift json dumps to formats more easily used for topic modeling. Run `python -m ihop.import_data --help` for details
- `ihop.community2vec`: Wrappers for training and tuning word2vec to implement community2vec on the Reddit datasets.


# External dependencies:
- [Java](https://docs.oracle.com/en/java/javase/17/install/overview-jdk-installation.html) or [OpenJDK](https://openjdk.java.net/install/) (at least version 8). Make sure you have `JAVA_HOME` set appropriately
- (Optional to support faster compression & customize Hadoop config for Spark) [Hadoop](https://hadoop.apache.org) at least version 3.3 is needed for Pyspark to properly decompress the Reddit zst files (see [this issue](https://stackoverflow.com/questions/64607248/configure-spark-on-yarn-to-use-hadoop-native-libraries) or [this one](https://stackoverflow.com/questions/67099204/reading-a-zst-archive-in-scala-spark-native-zstandard-library-not-available)). Install Hadoop and configure the environment variables using [these instructions](https://phoenixnap.com/kb/install-hadoop-ubuntu).

# Setup and Installation
Note that [Pyspark](https://spark.apache.org/docs/latest/api/python/getting_started/install.html#dependencies) is used for training embeddings, so you must have Java installed (at least version 8) and `JAVA_HOME` set appropriately.

Use [Anaconda](https://docs.anaconda.com/anaconda/install/index.html) to create the `ihop` environment by running `conda env create -f environment.yml` or install the dependencies using `pip install -r requirements.txt`. These environments include dependencies for running jupyter notebooks.

To install the `ihop` package as importable for testing and development, run `pip install -e .`. See [this article](https://godatadriven.com/blog/a-practical-guide-to-using-setup-py/) for more details on using `setup.py`.

# Testing
Unit tests can be run with [`python -m pytest`](https://docs.pytest.org/en/6.2.x/).

# Known Issues
- Spark can't read the origial zst compressed files from Pushshift, due to the window size being larger than 27 and I didn't know how to change the Spark/Hadoop settings to fix this (see note in [zstd man page](https://manpages.debian.org/unstable/zstd/zstd.1.en.html and [Stackoverflow: Read zst to pandas](https://stackoverflow.com/questions/61067762/how-to-extract-zst-files-into-a-pandas-dataframe))). Moreover, if you try to read in large .zst files in Spark, you are limited by memory and if there's not enough, the dataframe just gets filled with `null`. The workaround is re-compress the file as a bzip2 before running `ihop.import_data.py`. This takes a long time, but is simple on the command line and `scripts/export_c2v.sh` is provided as a wrapper for the import.

# TODOs
- Configure logging, replace print statements with logging