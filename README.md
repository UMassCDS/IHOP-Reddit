# IHOP
The Center for Data Science repository for the International Hate Observatory Project.

# Setup and Installation
Note that [Pyspark](https://spark.apache.org/docs/latest/api/python/getting_started/install.html#dependencies) is used for training embeddings, so you must have Java installed (at least version 8) and `JAVA_HOME` set appropriately.

Use [Anaconda](https://docs.anaconda.com/anaconda/install/index.html) to create the `ihop` environment by running `conda env create -f environment.yml` or install the dependencies using `pip install -r requirements.txt`. These environments include dependencies for running jupyter notebooks.

To install the `ihop` package as importable for testing and development, run `pip install -e .`. See [this article](https://godatadriven.com/blog/a-practical-guide-to-using-setup-py/) for more details on using `setup.py`.

# Testing
Unit tests can be run with [`pytest`](https://docs.pytest.org/en/6.2.x/).