"""Functionality related to manipulating visualizations of clusterings using the Dash app, pandas DataFrames and seaborn or plotly libraries.

.. TODO: TSNE parameters are not being carefully tracked right now, but this may be necessary in the future
"""
import argparse
import json
import logging
import pathlib

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

from ihop.community2vec import GensimCommunity2Vec
import ihop.utils

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


def generate_tsne_dataframe(
    c2v_path, key_col="subreddit", n_components=2, random_state=77, **kwargs
):
    """Fits a TSNE representation of the dataframe.
    Returns the result asPandas dataframe

    :param c2v_path: str, path to a trained GensimCommunity2Vec model saved to disk
    :param key_col: str, column name for indexed values
    :param n_components: int, usually 2 or 3 dimensions, since the purpose of this is for creating visualizations
    :param kwargs: dict params passed to sklearn's TNSE model
    """
    logger.debug("Loading community2vec model: %s", c2v_path)
    c2v_model = GensimCommunity2Vec.load(c2v_path)
    logger.debug("Model %s successfully loaded", c2v_path)
    tsne_fitter = TSNE(
        **kwargs,
        n_components=n_components,
        init="pca",
        metric="cosine",
        learning_rate="auto",
        square_distances=True,
        random_state=random_state,
    )
    tsne_projection = tsne_fitter.fit_transform(c2v_model.get_normed_vectors())
    logger.info("TSNE ran for %s iterations", tsne_fitter.n_iter_)

    dataframe_elements = list()
    for i, vocab_elem in enumerate(c2v_model.get_index_to_key()):
        elem_proj = tsne_projection[i]
        dataframe_elements.append((vocab_elem, *elem_proj))

    # Generate columns for dataframe
    cols = [key_col]
    for i in range(1, n_components + 1):
        cols.append(f"tsne_{i}")

    return pd.DataFrame.from_records(dataframe_elements, columns=cols)


def load_tsne_dataframe(tsne_csv):
    """Loads the TSNE dataframe produced by generate_tsne_dataframe.
    Returns previously saved TSNE coordinates as a Pandas DataFrame

    :param tsne_csv: Path or str
    """
    return pd.read_csv(tsne_csv, header=0)


def main(c2v_path, tsne_csv):
    """Generates a TSNE

    :param c2v_path: str, Path to the trained GensimCommunity2Vec model
    :param tsne_csv: str, Path to the desired dataframe to store t-sne coordinates
    """
    tsne_df = generate_tsne_dataframe(c2v_path)
    logger.debug("Writing TSNE dataframe to %s", tsne_csv)
    tsne_df.to_csv(tsne_csv, index=False)
    logger.debug("TSNE Dataframe successfully written")


parser = argparse.ArgumentParser(
    description="Create 2-dimensional TSNE visualizations that can be used in the cluster visualization app."
)

parser.add_argument(
    "--config",
    type=pathlib.Path,
    help="JSON file used to override default logging and spark configurations",
)

parser.add_argument(
    "c2v_model",
    help="Path to Community2Vec model produced by ihop.community2vec to create TSNE projection of the subreddit embeddings.",
)
parser.add_argument(
    "tsne_csv",
    help="Path to a CSV file to write out TSNE (x,y) coordinates, so they can be later loaded as a DataFrame",
)


if __name__ == "__main__":
    try:
        args = parser.parse_args()
        config = ihop.utils.parse_config_file(args.config)
        ihop.utils.configure_logging(config[1])
        logger.debug("Script arguments: %s", args)
        main(args.c2v_model, args.tsne_csv)
    except Exception:
        logger.error("Fatal error while producing TSNE visualization", exc_info=True)
