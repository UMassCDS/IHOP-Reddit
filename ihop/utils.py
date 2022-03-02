"""Utilities for working with Spark, numpy, file utils and other misc. operations

# TODO Eventually, we may need to read Spark config from a file, especially if we want to submit jobs to a cluster.
"""
import json
import logging
import os

import numpy as np
from pyspark.sql import SparkSession
import pyspark.sql.functions as fn

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


def get_start_end_timeframes(spark_dataframe, utc_time_col="created_utc"):
    """Returns a dataframe detailing the start and end times for input dataframe in both UTC time and a human readable format

    :param spark_dataframe: Spark DataFrame
    :param utc_time_col: column containing timestamps
    """
    timeframes = spark_dataframe.select(
        fn.max(utc_time_col).alias('end_timeframe'),
        fn.min(utc_time_col).alias('start_timeframe')
    )
    timeframes = timeframes.withColumn('human_readable_start', fn.from_unixtime(
        'start_timeframe')).withColumn('human_readable_end', fn.from_unixtime('end_timeframe'))
    return timeframes


class NumpyFloatEncoder(json.JSONEncoder):
    """Avoids issues serializing numpy float32 to json
    """

    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        return json.JSONEncoder.default(self, obj)
