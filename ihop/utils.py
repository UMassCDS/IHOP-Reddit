"""Utilities for working with Spark, numpy, file utils and other misc. operations

# TODO Eventually, we may need to read Spark config from a file, especially if we want to submit jobs to a cluster.
"""
import json
import logging
import logging.config
import os

import numpy as np
import pyspark
from pyspark.sql import SparkSession
import pyspark.sql.functions as fn

logger = logging.getLogger(__name__)

HADOOP_ENV = "HADOOP_HOME"

DEFAULT_LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default_formatter": {
            "format": "%(name)s : %(asctime)s : %(levelname)s : %(message)s"
        },
    },
    "handlers": {
        "stream_handler": {
            "class": "logging.StreamHandler",
            "formatter": "default_formatter",
        },
        "file_handler": {
            "class": "logging.FileHandler",
            "filename": "ihop.log",
            "formatter": "default_formatter",
        },
    },
    "root": {"handlers": ["stream_handler", "file_handler"], "level": logging.DEBUG,},
    "loggers": {
        "py4j": {
            "handler": ["stream_handler", "file_handler"],
            "level": logging.WARNING,
        }
    },
}

DEFAULT_SPARK_CONFIG = {
    "spark.driver.memory": "4G",
    "spark.executor.memory": "4G",
}


def parse_config_file(config_file):
    """Reads a config file from JSON, optionally expecting 'spark' and 'logger' keys. If a key isn't present, None is returned for the key.
    Returns (spark_config, logger_config, full_config)
    """
    if not os.path.exists(config_file):
        return None, None, None

    with open(config_file, "r") as conf:
        conf_dict = json.load(conf)
        spark_conf = conf_dict.get("spark")
        logger_conf = conf_dict.get("logger")

    return spark_conf, logger_conf, conf_dict


def configure_logging(log_dict=None):
    """Configures the root logger for the application.

    :param log_dict: dict, specify logging configurations to override the defaults
    """
    if log_dict is None:
        logging.config.dictConfig(DEFAULT_LOGGING_CONFIG)
    else:
        logging.config.dictConfig(log_dict)


def get_spark_session(name, config=None):
    """Return a SparkSession configured with checking HADOOP_HOME for additional library support.

    :param name: str, application name to pass to Spark
    :param config: dict, dictionary where keys are Spark properites, see https://spark.apache.org/docs/latest/configuration.html
    """
    use_config = {}
    if config is None:
        use_config.update(DEFAULT_SPARK_CONFIG)
    else:
        use_config.update(config)

    if HADOOP_ENV in os.environ:
        hadoop_lib_path = os.path.join(os.environ[HADOOP_ENV], "lib", "native")
        lib_path_keys = [
            "spark.driver.extraLibraryPath",
            "spark.executor.extraLibraryPath",
        ]
        for k in lib_path_keys:
            use_config[k] = hadoop_lib_path

    else:
        logger.warning(
            "WARNING: No HADOOP_HOME variable found, zstd decompression may not be available"
        )
    conf = pyspark.SparkConf().setAll(list(use_config.items()))
    spark = SparkSession.builder.appName(name).config(conf=conf).getOrCreate()
    logger.info("Spark configuration: %s", spark.sparkContext.getConf().getAll())

    return spark


def get_start_end_timeframes(spark_dataframe, utc_time_col="created_utc"):
    """Returns a dataframe detailing the start and end times for input dataframe in both UTC time and a human readable format

    :param spark_dataframe: Spark DataFrame
    :param utc_time_col: column containing timestamps
    """
    timeframes = spark_dataframe.select(
        fn.max(utc_time_col).alias("end_timeframe"),
        fn.min(utc_time_col).alias("start_timeframe"),
    )
    timeframes = timeframes.withColumn(
        "human_readable_start", fn.from_unixtime("start_timeframe")
    ).withColumn("human_readable_end", fn.from_unixtime("end_timeframe"))
    return timeframes


class NumpyFloatEncoder(json.JSONEncoder):
    """Avoids issues serializing numpy float32 to json
    """

    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        return json.JSONEncoder.default(self, obj)
