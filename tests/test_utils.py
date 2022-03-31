"""Unit tests for ihop.utils"""
import logging
import os

import pytest

import ihop.utils


@pytest.fixture
def config_json(fixture_dir):
    return os.path.join(fixture_dir, "test_config.json")


def test_parse_config(config_json):
    spark_conf, logger_conf = ihop.utils.parse_config_file(config_json)
    assert set(spark_conf.items()) == set(
        [("spark.app.name", "utils test"), ("spark.driver.memory", "1G")]
    )
    assert set(logger_conf.keys()) == set(["handlers", "root"])
    assert logger_conf["handlers"] == {
        "stream_handler": {"class": "logging.StreamHandler"}
    }
    assert logger_conf["root"] == {
        "handler": ["stream_handler"],
        "level": logging.WARNING,
    }

