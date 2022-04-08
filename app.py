"""Visualize subreddit clusters
Run using `python app.py` and visit http://127.0.0.1:8050

# TODO vector models should be configurable
"""
import argparse
import logging

import dash
import dash_bootstrap_components as dbc
import dash_daq
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

import ihop.visualizations as iv
import ihop.utils
import ihop.community2vec
import ihop.clustering
import ihop.resources.collections

logger = logging.getLogger(__name__)
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

STARTING_NUM_CLUSTERS = 250
STARTING_RANDOM_SEED = 100


# TODO Config and select from multiple models
c2v = ihop.community2vec.GensimCommunity2Vec.load(
    "data/community2vec/RC_2021-05_5percentTopUsersExcluded_02142022/models/alpha0.05_negative10_sample0.005_vectorSize100"
)
model_description = "TODO: add some nice blurb about the model"
tsne_df, _ = c2v.get_tsne_dataframe()
subreddits = tsne_df["subreddit"].sort_values().unique()

KMEANS_PARAM_SECTION = [
    dash.html.Div(
        children=[
            dash.html.H2("Select clustering parameters"),
            dbc.Row(
                children=[
                    dbc.Col(
                        children=[
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
                    dbc.Col(
                        children=[
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
                ]
            ),
            dash.html.Br(),
            dbc.Button("Train clustering model", id="clustering_button"),
        ]
    ),
    dash.html.Br(),
    dash.html.Div(
        children=[
            dash.html.H2("K-means clustering metrics"),
            dash.dcc.Loading(
                id="loading-metrics",
                type="default",
                children=[dash.html.Article(id="cluster-metrics")],
            ),
            dash.html.Br(),
        ]
    ),
]

SUBREDDIT_FILTERING_SECTION = dash.html.Div(
    children=[
        dash.html.H2("Filter by Subreddits and Clusters"),
        dash.html.Label("Select subreddits"),
        dbc.Row(
            children=[
                dbc.Col(
                    children=[
                        dash.dcc.Dropdown(
                            subreddits,
                            # TODO Could allow user to select form list of possible collections
                            ihop.resources.collections.get_collection_members(
                                "Denigrating toward immigrants"
                            ),
                            multi=True,
                            id="subreddit-dropdown",
                        ),
                    ],
                    width=9,
                ),
                dbc.Col(
                    dash_daq.BooleanSwitch(
                        id="show-in-cluster-neighbors",
                        label="Display subreddits in the same clusters",
                    )
                ),
            ]
        ),
        dash.html.Br(),
        dash.html.Label("Select clusters"),
        dash.dcc.Dropdown(multi=True, id="cluster-dropdown"),
        dash.html.Br(),
        dash.dcc.Loading(type="default", id="subreddit-cluster-table"),
    ]
)

MODEL_PLOT_SECTION = dbc.Col(
    dash.html.Div(
        children=[
            dash.html.Div(
                children=[
                    dash.html.H2(id="model-name"),
                    dash.html.Br(),
                    dash.html.P(model_description),
                    dash.html.Br(),
                ]
            ),
            dash.dcc.Loading(
                dash.dcc.Graph(id="cluster-visualization"),
                id="loading-plot",
                type="default",
            ),
        ],
    )
)

BODY = dash.html.Div(
    children=[
        dash.html.H1("Community2Vec Subreddit Clusters"),
        MODEL_PLOT_SECTION,
        dbc.Accordion(
            start_collapsed=False,
            children=[
                dbc.AccordionItem(
                    KMEANS_PARAM_SECTION, title="K-means Cluster Parameters",
                )
            ],
        ),
        dash.html.Br(),
        SUBREDDIT_FILTERING_SECTION,
        dash.html.Br(),
        dash.dcc.Store(id="cluster-assignment"),
    ]
)

app.layout = dash.html.Div(children=dbc.Container([BODY]))


def get_metrics_display(metrics_dict):
    """Returns the html output used for displaying model metrics.

    :param metrics_dict: dict, name of metric -> value
    :return: a list of dash.html objects
    """
    display_output = []
    for metric_name, metric_value in metrics_dict.items():
        display_output.extend([dash.html.H3(metric_name), dash.html.P(metric_value)])
    return display_output


@app.callback(
    dash.Output("cluster-assignment", "data"),
    dash.Output("model-name", "children"),
    dash.Output("cluster-metrics", "children"),
    dash.Input("clustering_button", "n_clicks"),
    dash.State("n-clusters", "value"),
    dash.State("random-seed", "value"),
)
def train_clusters(n_clicks, n_clusters, random_seed):
    """Trains kmeans cluster with given number of clusters and random seed.

    :param n_clicks: int, button click indicator which triggers training the model (value unused)
    :param n_clusters: int, number of clusters to create
    :param random_seed: int, random seed for reproducibility

    :return: Return cluster assignments with a model name as a json {'name': 'model name', 'clusters': json_serialized_pandas_dataframe}
    """
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
            "clusters": iv.jsonify_stored_df(
                cluster_model.get_cluster_results_as_df(join_df=tsne_df)
            ),
        },
        model_name,
        get_metrics_display(metrics_dict),
    )


@app.callback(
    dash.Output("cluster-visualization", "figure"),
    dash.Input("cluster-assignment", "data"),
)
def get_cluster_visualization(cluster_json):
    """Build the plotly visualization for a model
    """
    model_name = cluster_json["name"]
    cluster_df = iv.unjsonify_stored_df(cluster_json["clusters"], [model_name])

    figpx = px.scatter(
        cluster_df,
        x="tsne_1",
        y="tsne_2",
        color=model_name,
        text="subreddit",
        hover_data=["subreddit", model_name],
    )

    layout = go.Layout(
        updatemenus=[
            dict(
                type="buttons",
                showactive=True,
                xanchor="left",
                y=1.3,
                buttons=[
                    dict(
                        method="restyle",
                        label="Toggle subreddit labels",
                        visible=True,
                        args2=["text", {"visible": False}],
                        args=[{"text": [d.customdata[:, 0] for d in figpx.data],}],
                    )
                ],
            )
        ],
        showlegend=True,
        legend_title_text=model_name,
    )

    fig = go.Figure(data=figpx.data, layout=layout)

    return fig


@app.callback(
    dash.Output("cluster-dropdown", "options"), dash.Input("n-clusters", "value")
)
def set_cluster_dropdown(n_clusters):
    """Give the cluster selection options the cluster index numbers to choose from
    """
    return [i for i in range(int(n_clusters))]


@app.callback(
    dash.Output("subreddit-cluster-table", "children"),
    dash.Input("cluster-assignment", "data"),
    dash.Input("subreddit-dropdown", "value"),
    dash.Input("cluster-dropdown", "value"),
    dash.Input("show-in-cluster-neighbors", "on"),
)
def get_display_table(
    cluster_json, selected_subreddits, selected_clusters, is_show_cluster_neighbors
):
    """Builds the DataTable showing all selected subreddits and clusters
    """
    model_name = cluster_json["name"]
    cluster_df = iv.unjsonify_stored_df(cluster_json["clusters"], [model_name])

    selected_subreddits_df = cluster_df[
        cluster_df["subreddit"].isin(selected_subreddits)
    ].copy()

    all_selected_clusters = list()
    if selected_clusters is not None:
        all_selected_clusters += selected_clusters

    if is_show_cluster_neighbors:
        all_selected_clusters.extend(selected_subreddits_df[model_name].unique())

    selected_clusters_df = cluster_df[
        cluster_df[model_name].isin(all_selected_clusters)
    ]
    selected_subreddits_df = pd.concat(
        [selected_subreddits_df, selected_clusters_df]
    ).drop_duplicates()

    tsne_cols = [c for c in selected_subreddits_df.columns if c.startswith("tsne")]
    selected_subreddits_df.drop(columns=tsne_cols, inplace=True)
    return dash.dash_table.DataTable(
        selected_subreddits_df.to_dict("records"),
        [{"name": i, "id": i} for i in selected_subreddits_df.columns],
        sort_action="native",
        export_format="csv",
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

