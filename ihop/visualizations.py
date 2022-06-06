"""Functionality related to manipulating visualizations of clusterings using the Dash app, pandas DataFrames and seaborn or plotly libraries.
"""
import json
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


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
    """Returns a copy of the dataframe, but with an additional output_col storing the same values values as input_col, but any values appear in the keep_values
    are replaced with other_value. Useful for reducing the information
    present in visualizations. When no keep values are specified, output_col is just a copy of input_col.

    :param dataframe: pandas DataFrame to add the column to
    :param input_col: str, name of Categorical column where original values are coming from
    :param output_col: str, name of desired new Categorical column
    :param keep_values: set or list of values from input_col that should also appear in output_col
    :param other_value: the value will appear in output_col instead of the original input_col value if the original value isn't in keep_values
    """
    logger.debug("Cateogries to keep in new column: %s", keep_values)
    logger.debug("Other categories will be replaced with '%s'", other_value)
    new_dataframe = dataframe.copy()

    if other_value in new_dataframe[input_col]:
        raise ValueError(
            f"Replacement value '{other_value}' is already a value in column {input_col}"
        )

    if len(keep_values) == 0:
        new_dataframe[output_col] = new_dataframe[input_col].copy()
    else:
        # Fill the output column with keep values or other
        new_dataframe[output_col] = np.where(
            new_dataframe[input_col].isin(keep_values),
            new_dataframe[input_col],
            "other",
        )

    new_dataframe[output_col] = pd.Categorical(new_dataframe[output_col])
    return new_dataframe
