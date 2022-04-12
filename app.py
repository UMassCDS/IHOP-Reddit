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

CLUSTER_ASSIGNMENT_DISPLAY_NAME = "Cluster Assignment"
UNSELECTED_CLUSTER_KEY = "other"
UNSELECTED_COLOR = "#D3D3D3"


# TODO Config and select from multiple models
c2v = ihop.community2vec.GensimCommunity2Vec.load(
    "data/community2vec/RC_2021-05_5percentTopUsersExcluded_02142022/models/alpha0.05_negative10_sample0.005_vectorSize100"
)

MODEL_DESCRIPTION_MD = """This a [Community2Vec model](https://aclanthology.org/W17-2904/) trained on a month's worth of Reddit comments from the 10,000 most popular subreddits and tuned to perform well on a set of pre-defined analogy tasks, like matching up sports teams with the cities they play in, `r/Nationals - r/washingtondc + r/toronto = r/Torontobluejays`. The model positions subreddit communities in multidimensional space such that subreddits with similar user bases are close together.

This particular community2vec model was trained on May 2021 Reddit comments and achieved a 65% accuracy on the [predetermined analogy set](https://github.com/UMassCDS/IHOP/tree/main/ihop/resources/analogies).

You can use clustering to create groupings of subreddits based on overlapping users.  This strategy can be used to understand social dimensions in Reddit, such as political polarization, as shown by Waller and Anderson in [Quantifying social organization and political polarization in online platforms](https://www.nature.com/articles/s41586-021-04167-x).
"""

KMEANS_METRICS_MD = """The quality of topics produced by clustering the community embeddings can be subjective. However, a human should be able to intuit each cluster's theme based on the dominant topic, such as 'politics', 'music', or 'sports'. Changing the number of clusters should correspond to a change in granularity of the labels, e.g. 'sports' vs 'basketball'.

Generating gold-standard labels for clusters isn't feasible, so instead we can rely on metrics that measure the overlap and dispersion of clusters:

- **Silhouette Coefficient**: Ranges between -1 (clustering is totally incorrect) and 1 (clusters are dense and well separated), scores around 0 indicate overlapping clusters.

- **Calinski-Harabasz Index**: Higher for models with clusters that are dense and well separated.

- **Davies-Bouldin Index**: Measures separation/similarity between clusters by taking ratio of within-cluster distances to between-cluster distances. The minimum is zero, lower scores indicate better separation

These scores can help you compare different groupings of the data using the same number of clusters. The scores for your current model are:
"""

TSNE_DESCRIPTION_MD = """T-distributed Stochastic Neighbor Embedding (t-SNE) is a way of visualizing high-dimensional data in 2D or 3D, so we can actually make sense of it. However, like map projections of the globe into 2D, this can cause distortions, as described in [How to Use t-SNE Effectively](https://distill.pub/2016/misread-tsne/). This visualization should be treated as a guide to help explore subreddit clusters interatively, not a definitive measure of how similar subreddits are to each other.

Each point is a subreddit and the color marks the cluster it gets assigned to. """


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
            dash.dcc.Markdown(KMEANS_METRICS_MD),
            dash.html.Br(),
            dash.dcc.Loading(
                id="loading-metrics",
                type="default",
                children=[dash.html.Article(id="cluster-metrics")],
            ),
            dash.html.Br(),
        ]
    ),
]

SUBREDDIT_DROPDOWN = [
    dash.html.Label("Select subreddits"),
    dash.dcc.Dropdown(
        subreddits,
        # TODO Could allow user to select form list of possible collections
        ihop.resources.collections.get_collection_members(
            "Denigrating toward immigrants"
        ),
        multi=True,
        id="subreddit-dropdown",
    ),
]

CLUSTER_DROPDOWN = [
    dash.html.Label("Select clusters"),
    dash.dcc.Dropdown(multi=True, id="cluster-dropdown"),
]

SUBREDDIT_FILTERING_SECTION = dash.html.Div(
    children=[
        dash.html.H2("Filter by Subreddits and Clusters"),
        dbc.Row(
            children=[
                dbc.Col(children=SUBREDDIT_DROPDOWN, width=8,),
                dbc.Col(
                    dash.html.Div(
                        dbc.Button(
                            "Highlight selected clusters in graph",
                            id="highlight-selected-clusters",
                        ),
                        style={"verticalAlign": "middle"},
                    ),
                    align="center",
                ),
            ]
        ),
        dash.html.Br(),
        dbc.Row(
            children=[
                dbc.Col(children=CLUSTER_DROPDOWN, width=8),
                dbc.Col(
                    dash.html.Div(
                        dash_daq.BooleanSwitch(
                            id="show-in-cluster-neighbors",
                            label="Display entire cluster in table",
                        ),
                        style={"verticalAlign": "middle"},
                    ),
                    align="center",
                ),
            ]
        ),
        dash.html.Br(),
        dash.dcc.Loading(type="default", id="subreddit-cluster-table"),
    ]
)

INTRODUCTION_SECTION = dash.html.Div(
    children=[
        dash.html.H2(id="model-name"),
        dash.html.Br(),
        dash.dcc.Markdown(MODEL_DESCRIPTION_MD),
        dash.html.Br(),
    ]
)

MODEL_PLOT_SECTION = dbc.Col(
    dash.html.Div(
        children=[
            dash.dcc.Markdown(TSNE_DESCRIPTION_MD),
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
        INTRODUCTION_SECTION,
        dbc.Accordion(
            start_collapsed=False,
            always_open=True,
            active_item=["item-0", "item-1"],
            children=[
                dbc.AccordionItem(
                    KMEANS_PARAM_SECTION, title="K-means Cluster Parameters",
                ),
                dbc.AccordionItem(
                    MODEL_PLOT_SECTION, title="t-SNE Visualization of Clusters"
                ),
            ],
        ),
        dash.html.Br(),
        SUBREDDIT_FILTERING_SECTION,
        dash.html.Br(),
        dash.dcc.Store(id="cluster-assignment"),
    ]
)

app.layout = dbc.Container([BODY])


def get_metrics_display(metrics_dict):
    """Returns the html output used for displaying model metrics.

    :param metrics_dict: dict, name of metric -> value
    :return: a list of dash.html objects
    """
    display_output = []
    for metric_name, metric_value in metrics_dict.items():
        display_output.extend([dash.html.H4(metric_name), dash.html.P(metric_value)])
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
    dash.State("subreddit-dropdown", "value"),
    dash.State("cluster-dropdown", "value"),
    dash.Input("highlight-selected-clusters", "n_clicks"),
)
def get_cluster_visualization(
    cluster_json, subreddit_selection, cluster_selection, is_only_highlight_selection
):
    """Build the plotly visualization for a model
    """
    model_name = cluster_json["name"]
    logger.info("Updating graph visualization, model: %s", model_name)
    cluster_df = iv.unjsonify_stored_df(cluster_json["clusters"], [model_name])
    cluster_df[CLUSTER_ASSIGNMENT_DISPLAY_NAME] = cluster_df[model_name]

    if is_only_highlight_selection:
        logger.info("Highlight selected clusters is selected")
        logger.info("Subreddit list given: %s", subreddit_selection)
        logger.info("Cluster list given: %s", cluster_selection)
        cluster_set = set()
        if subreddit_selection is not None:
            subreddit_clusters = cluster_df[
                cluster_df["subreddit"].isin(subreddit_selection)
            ][CLUSTER_ASSIGNMENT_DISPLAY_NAME].unique()
            cluster_set.update(subreddit_clusters)
            logger.info("Clusters used by subreddits: %s", cluster_set)
        if cluster_selection is not None:
            cluster_set.update(cluster_selection)

        # Set unselected clusters to 'other'
        if len(cluster_set) > 0:
            logger.info("Highlighted clusters will be: %s", cluster_set)
            cluster_df[CLUSTER_ASSIGNMENT_DISPLAY_NAME].cat.set_categories(
                list(cluster_set) + [UNSELECTED_CLUSTER_KEY], rename=True, inplace=True
            )
            cluster_df[CLUSTER_ASSIGNMENT_DISPLAY_NAME].fillna(
                UNSELECTED_CLUSTER_KEY, inplace=True
            )

    figpx = px.scatter(
        cluster_df,
        x="tsne_1",
        y="tsne_2",
        text="subreddit",
        color=CLUSTER_ASSIGNMENT_DISPLAY_NAME,
        hover_data=["subreddit", model_name],
    )

    # Hide text annotation initially
    for d in figpx.data:
        d.mode = "markers"

    if is_only_highlight_selection:
        logger.info("Updating color for unselected clusters")
        for d in figpx.data:
            if d.name == UNSELECTED_CLUSTER_KEY:
                d.marker.color = UNSELECTED_COLOR

    layout = go.Layout(
        updatemenus=[
            dict(
                type="buttons",
                # xanchor="left",
                # yanchor="bottom",
                # y=1.3,
                buttons=[
                    dict(
                        method="restyle",
                        label="Toggle subreddit labels",
                        visible=True,
                        args=[{"mode": "markers"}],
                        args2=[{"mode": "markers+text"}],
                    )
                ],
            )
        ],
        showlegend=True,
        legend_title_text=CLUSTER_ASSIGNMENT_DISPLAY_NAME,
        title=f"t-SNE Projection of Community2Vec Subreddit Clusterings<br>with {model_name}",
    )

    fig = go.Figure(data=figpx.data, layout=layout)
    logger.info("Figure successfully generated")
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
    logger.info("Generating display table")
    logger.info("Selected subreddits: %s", selected_subreddits)
    logger.info("Show cluster neighbors option: %s", is_show_cluster_neighbors)
    logger.info("Selected clusters: %s", selected_subreddits)
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

    logger.info(
        "Selecting these clusters for display in table: %s", all_selected_clusters
    )
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
        # TODO Plotly handles logging strangely, so use logger.info or workaround to not silence logging,
        # see https://community.plotly.com/t/logging-debug-messages-suppressed-in-callbacks/17854
        app.run_server(debug=args.debug)
    except Exception as e:
        logger.error(e)

