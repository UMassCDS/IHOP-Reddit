"""Visualize subreddit clusters
Run using `python app.py` and visit http://127.0.0.1:8050

# TODO vector models should be configurable
"""
import argparse
import logging

from dash import Dash, dcc, html
import plotly.express as px
import pandas as pd

import ihop.utils
import ihop.community2vec

logger = logging.getLogger(__name__)

app = Dash(__name__)

# TODO Config and select from multiple models
c2v = ihop.community2vec.GensimCommunity2Vec.load(
    "data/community2vec/RC_2021-05_5percentTopUsersExcluded_02142022/models/alpha0.05_negative10_sample0.005_vectorSize100"
)


def get_cluster_visualization(vector_model, subreddits=None, clusters=None):
    """Build the plotly visualization for a model

    :param vector_model: Gensim community to vec for clusters
    :param subreddits: list of subreddits to filter clusters from
    :param cluster: list of cluster ids to filter visualization
    """
    df, _ = vector_model.get_tsne_dataframe()
    return px.scatter(
        df, x="tsne_x", y="tsne_y", text="subreddit", hover_data=["subreddit"],
    )


app.layout = html.Div(
    children=[
        html.H1(children="Hello Dash"),
        dcc.Graph(id="test_c2v_clusters", figure=get_cluster_visualization(c2v)),
    ]
)


parser = argparse.ArgumentParser(
    description="Runs a Dash application for browsing subreddit clusters"
)
# TODO Add application confiugration as needed
parser.add_argument(
    "--config",
    default=(ihop.utils.DEFAULT_SPARK_CONFIG, ihop.utils.DEFAULT_LOGGING_CONFIG),
    type=ihop.utils.parse_config_file,
    help="JSON file used to override default logging and spark configurations",
)
parser.add_argument(
    "-d",
    "--debug",
    action="store_true",
    help="Use this flag to launch the application in 'hot-reload' mode",
)

if __name__ == "__main__":
    args = parser.parse_args()
    print("Starting IHOP subreddit visualization application")
    config = args.config
    print("Configuration:", config)
    ihop.utils.configure_logging(config[1])
    logger.info("Logging configured")
    logger.info("Starting app")
    app.run_server(debug=args.debug)

