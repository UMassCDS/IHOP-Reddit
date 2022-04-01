"""Visualize subreddit clusters
Run using `python app.py` and visit http://127.0.0.1:8050

# TODO vector models should be configurable
"""
import argparse
import logging

import dash
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd

import ihop.utils
import ihop.community2vec

logger = logging.getLogger(__name__)

# TODO Config and select from multiple models
c2v = ihop.community2vec.GensimCommunity2Vec.load(
    "data/community2vec/RC_2021-05_5percentTopUsersExcluded_02142022/models/alpha0.05_negative10_sample0.005_vectorSize100"
)
tsne_df, _ = c2v.get_tsne_dataframe()


def get_cluster_visualization(tsne_df, subreddits=None, clusters=None):
    """Build the plotly visualization for a model

    :param vector_model: Gensim community to vec for clusters
    :param subreddits: list of subreddits to filter clusters from
    :param cluster: list of cluster ids to filter visualization
    """
    return px.scatter(
        tsne_df, x="tsne_x", y="tsne_y", text="subreddit", hover_data=["subreddit"],
    )

@app.callback(
    Output(component_id="cluster_plot", component_property="children")

)


KMEANS_PARAM_SECTION = [
    dash.html.Label("Number of clusters"),
    dbc.Input(
        id="n_clusters",
        type="number",
        placeholder="number of clusters",
        min=0,
        debounce=True,
        value=250,
    ),
    dash.html.Label("Random seed"),
    dbc.Input(
        id="random seed",
        type="number",
        placeholder="random seed",
        min=0,
        debounce=True,
        value=10,
    ),
]

BODY = dash.html.Div(
    children=[
        dash.html.H1("Community2Vec Subreddit Clusters"),
        dbc.Row(
            dbc.Col(
                dash.html.Div(
                    children=[
                        dbc.Accordion(
                            dbc.AccordionItem(
                                KMEANS_PARAM_SECTION, title="K-means Cluster Parameters"
                            )
                        )
                    ]
                )
            )
        ),
        dbc.Row(
            dbc.Col(
                dash.html.Div(
                    children=[
                        dash.html.H2("Cluster visualization"),
                        dash.dcc.Graph(
                            id="cluster_visualization",
                        ),
                    ],
                )
            )
        ),
    ]
)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = dash.html.Div(children=dbc.Container([BODY]))

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

