"""Utilities for working with Spark and various file and path manipulations.

# TODO Eventually, we may need to be more careful with Spark configuration, especially if we want to submit jobs to a cluster.
"""
import os
from pyspark.sql import SparkSession

HADOOP_ENV = "HADOOP_HOME"

def get_spark_session(name, driver_mem="8G", quiet=False):
    """Return a SparkSession configured with checking HADOOP_HOME for additional library support.

    :param name: str, application name to pass to Spark
    :param driver_mem, str, Spark configuration value for spark.driver.memory, defaults to '8G'. Make this large to prevent OOM errors from JVM
    :param quiet: True to print session configuration
    """

    spark_builder = SparkSession.builder
    if HADOOP_ENV in os.environ:
        hadoop_lib_path = os.path.join(os.environ[HADOOP_ENV], "lib", "native")
        spark_builder = spark_builder.config("spark.driver.extraLibraryPath", hadoop_lib_path).config("spark.executor.extraLibraryPath", hadoop_lib_path)
    else:
        print("WARNING: No HADOOP_HOME variable found, zstd decompression may not be available")

    spark =  spark_builder.config("spark.driver.memory", "8G").appName(name).getOrCreate()

    if not quiet:
        print("Spark configuration:")
        print(spark.sparkContext.getConf().getAll())

    return spark