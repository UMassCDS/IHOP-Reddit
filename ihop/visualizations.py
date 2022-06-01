"""Functionality related to manipulating visualizations of clusterings using the Dash app, pandas DataFrames and seaborn or plotly libraries.
"""
import json
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
    json_data = json.loads(dataframe_as_json)
    result = pd.DataFrame(
        json_data["data"], index=json_data["index"], columns=json_data["columns"]
    )
    if categorical_columns is not None:
        result[categorical_columns] = result[categorical_columns].astype("category")
    return result


def assign_other_category_column(
    dataframe, input_col, output_col, keep_values, other_value
):
    """Adds a new column output_col in place to dataframe with the same
    values as in the input_col, but any values appear in the keep_values
    are replaced with other_value. Useful for reducing the information
    present in visualizations.

    :param dataframe: pandas DataFrame to add the column to
    :param input_col: str, name of column where original values are coming from
    :param output_col:
    :param keep_values: set or list of values from input_col that should also appear in output_col
    :param other_value: the value will appear in output_col instead of the original input_col value if the original value isn't in keep_values
    """
    # TODO
    pass
