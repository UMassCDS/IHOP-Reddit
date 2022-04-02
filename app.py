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
import ihop.clustering

logger = logging.getLogger(__name__)
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

STARTING_NUM_CLUSTERS = 250
STARTING_RANDOM_SEED = 100


# TODO Config and select from multiple models
c2v = ihop.community2vec.GensimCommunity2Vec.load(
    "data/community2vec/RC_2021-05_5percentTopUsersExcluded_02142022/models/alpha0.05_negative10_sample0.005_vectorSize100"
)
tsne_df, _ = c2v.get_tsne_dataframe()

KMEANS_PARAM_SECTION = [
    dash.html.Div(
        [
            dash.html.Label("Number of clusters"),
            dbc.Input(
                id="n-clusters",
                type="number",
                placeholder="number of clusters",
                min=0,
                debounce=True,
                value=STARTING_NUM_CLUSTERS,
            ),
        ]
    ),
    dash.html.Div(
        [
            dash.html.Label("Random seed"),
            dbc.Input(
                id="random-seed",
                type="number",
                placeholder="random seed",
                min=0,
                debounce=True,
                value=STARTING_RANDOM_SEED,
            ),
        ]
    ),
    dash.html.Button("Train clustering model", id="clustering_button"),
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
                        dash.html.H2(id="model-name"),
                        dash.dcc.Graph(id="cluster-visualization",),
                    ],
                )
            )
        ),
        dash.dcc.Store(id="cluster-assignment"),
    ]
)

app.layout = dash.html.Div(children=dbc.Container([BODY]))

# TODO Add metrics display to output
def get_metrics_display(metrics_dict):
    return dash.html.Article(children = [])

@app.callback(
    dash.Output("cluster-assignment", "data"),
    dash.Output("model-name", "children"),
    dash.Output("cluster-metrics", "children")
    dash.Input("clustering_button", "n_clicks"),
    dash.State("n-clusters", "value"),
    dash.State("random-seed", "value"),
)
def train_clusters(n_clicks, n_clusters, random_seed):
    # TODO: eventually we may want to support different types of models. The ClusteringModelFactory should allow that fairly easily
    model_name = f"Kmeans Cluster Assignment {n_clusters} clusters and random state {random_seed}"
    cluster_model = ihop.clustering.ClusteringModelFactory.init_clustering_model(
        ihop.clustering.ClusteringModelFactory.KMEANS,
        c2v.get_normed_vectors(),
        c2v.get_index_as_dict(),
        model_name=model_name,
        **{"n_clusters": n_clusters, "random_state": random_seed},
    )
    cluster_model.train()
    metrics_dict = cluster_model.get_metrics()

    # TODO Tsne-df currently is constant, but should eventually be something determined from selecting which vector model to use
    return (
        {
            "name": model_name,
            "clusters": cluster_model.get_cluster_results_as_df(
                join_df=tsne_df
            ).to_json(orient="split"),
        },
        model_name,
    )


@app.callback(
    dash.Output("cluster-visualization", "figure"),
    dash.Input("cluster-assignment", "data"),
)
def get_cluster_visualization(cluster_json):
    """Build the plotly visualization for a model
    """
    model_name = cluster_json["name"]
    cluster_df = pd.read_json(cluster_json["clusters"], orient="split")
    cluster_df[model_name] = cluster_df[model_name].astype("category")
    return px.scatter(
        cluster_df,
        x="tsne_x",
        y="tsne_y",
        color=model_name,
        text="subreddit",
        hover_data=["subreddit", model_name],
    )


parser = argparse.ArgumentParser(
    description="Runs a Dash application for browsing subreddit clusters"
)
# TODO Add application confiugration as needed
parser.add_argument(
    "--config",
    default=(None, ihop.utils.DEFAULT_LOGGING_CONFIG),
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
    spark_conf, logging_conf = args.config
    print("Configuration:", args.config)
    ihop.utils.configure_logging(logging_conf)
    logger.info("Logging configured")
    logger.info("Starting app")
    try:
        app.run_server(debug=args.debug)
    except Exception as e:
        logger.error(e)

