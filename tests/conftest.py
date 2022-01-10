import pytest
from pyspark.sql import SparkSession


@pytest.fixture(scope='session')
def spark():
    """A Spark Session fixture to use for all tests
    """
    return SparkSession.builder.appName("IHOP tests").getOrCreate()