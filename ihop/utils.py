"""Utilities for working with Spark.

# TODO Eventually, we may need to read Spark config from a file, especially if we want to submit jobs to a cluster.
"""
import os
import logging

from pyspark.sql import SparkSession

logger = logging.getLogger(__name__)

HADOOP_ENV = "HADOOP_HOME"


def get_spark_session(name, driver_mem="8G", quiet=False):
    """Return a SparkSession configured with checking HADOOP_HOME for additional library support.

    :param name: str, application name to pass to Spark
    :param driver_mem, str, Spark configuration value for spark.driver.memory, defaults to '8G'. Make this large to prevent OOM errors from JVM
    :param quiet: True to print session configuration
    """

    if HADOOP_ENV in os.environ:
        hadoop_lib_path = os.path.join(os.environ[HADOOP_ENV], "lib", "native")
        spark = SparkSession.builder \
                            .config("spark.driver.extraLibraryPath", hadoop_lib_path) \
                            .config("spark.executor.extraLibraryPath", hadoop_lib_path) \
                            .config("spark.driver.memory", driver_mem) \
                            .appName(name).getOrCreate()
    else:
        print("WARNING: No HADOOP_HOME variable found, zstd decompression may not be available")
        spark = SparkSession.builder \
                            .config("spark.driver.memory", driver_mem) \
                            .appName(name).getOrCreate()

    if not quiet:
        print("Spark configuration:")
        print(spark.sparkContext.getConf().getAll())

    return spark
