"""Unit tests for ihop.import_data.py
"""
import os
import pytest

from ihop.import_data import get_spark_dataframe

FIXTURE_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'test_files')

@pytest.mark.datafiles(
    os.path.join(FIXTURE_DIR, 'comments1.json'),
    os.path.join(FIXTURE_DIR, 'comments2.json')
)
def test_get_spark_dataframe_comments(datafiles, spark):
    files = [str(f) for f in datafiles.listdir()]
    spark_df = get_spark_dataframe(files, spark, "comments")
    spark_df.show()
    assert len(spark_df.columns) == 14
    assert spark_df.count() == 3


def test_get_spark_dataframe_submissions(datafiles, spark):
    # TODO
    assert False



