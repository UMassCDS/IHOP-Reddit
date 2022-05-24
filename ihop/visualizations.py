"""Functionality related to manipulating visualizations of clusterings using pandas
dataframes and seaborn or plotly libraries
"""
import pandas as pd


def jsonify_stored_df(dataframe):
    """To keep dataframes in local cache, they must be in JSON, see https://dash.plotly.com/sharing-data-between-callbacks.
    Returns the dataframe in json format

    :param dataframe: pandas DataFrame
    """
    return dataframe.to_json(orient="split")


def unjsonify_stored_df(dataframe_as_json, categorical_columns=None):
    """Deserialize pandas from JSON

    :param dataframe_as_json: json mapping storing a dataframe
    :param categorical_columns: list of columns to be recast as categorical columns
    """
    result = pd.read_json(dataframe_as_json, orient="split")
    if categorical_columns is not None:
        result[categorical_columns] = result[categorical_columns].astype("category")
    return result
