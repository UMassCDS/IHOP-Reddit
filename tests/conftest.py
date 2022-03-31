import os

import pytest
from pyspark.sql import SparkSession


@pytest.fixture(scope="session")
def fixture_dir():
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_files")


@pytest.fixture(scope="session")
def spark():
    """A Spark Session fixture to use for all tests
    """
    return SparkSession.builder.appName("IHOP tests").getOrCreate()
