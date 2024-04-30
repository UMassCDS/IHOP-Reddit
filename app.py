"""Visualize subreddit clusters using a dash app.
Run locally for development and debugging using `python app.py` and visit http://127.0.0.1:8050

Can also be served with gunicorn: `gunicorn --bind 0.0.0.0:8050 app:server`

The app can be configured to accept different model paths by changing config.json file structured as:
{
    "logger": {<log config>},
    "model_paths": {
        "Identifier that will appear on the UI": "<path to output of ihop.community2vec.py model training>",
        }
}

# TODO vector models should be configurable (one model for each available time range? )
# TODO list of subreddits can be chosen from a collection with descriptions
"""
import json
import logging
import pathlib

import dash
import dash_bootstrap_components as dbc
import dash_daq
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

import ihop.visualizations as iv
import ihop.utils
import ihop.community2vec as ic2v
import ihop.clustering

logger = logging.getLogger(__name__)

CONFIG = "config.json"
spark_conf, logging_conf, conf = ihop.utils.parse_config_file(CONFIG)
print("Configuration:", CONFIG)
ihop.utils.configure_logging(logging_conf)
logger.info("Logging configured")

# Config stores a dictionary mapping month -> path to best c2v model
MODEL_DIRS = conf["model_paths"]
# Assume the tsne visualization for a month is stored in the model directory and named tsne.csv per the DVC tsne_visualization stage
TSNE_CSV_NAME = "tsne.csv"

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server
# APP DISPLAY CONSTANTS
STARTING_NUM_CLUSTERS = 250
STARTING_RANDOM_SEED = 100

CLUSTER_ASSIGNMENT_DISPLAY_NAME = "Cluster Assignment"
UNSELECTED_CLUSTER_KEY = "other"
UNSELECTED_COLOR = "#D3D3D3"

MODEL_DESCRIPTION_MD = """[Community2Vec models](https://aclanthology.org/W17-2904/) position subreddit communities in multidimensional space such that subreddits with similar user bases are close together. They can be trained on Reddit comments over any time period and tuned to perform well on a set of pre-defined analogy tasks , like matching up sports teams with the cities they play in, `r/Nationals - r/washingtondc + r/toronto = r/Torontobluejays`.

You can use clusterings of the learned community vectors to create groupings of subreddits based on overlapping users. This strategy can be used to understand social dimensions in Reddit, such as political polarization, as shown by Waller and Anderson in [Quantifying social organization and political polarization in online platforms](https://www.nature.com/articles/s41586-021-04167-x).
"""

MONTH_SELECTION_MD = """Here we present models trained on the comments from a single month. Each month of data undergoes the same preprocessing where we select the top 10,000 most popular subreddits by number of comments and remove deleted users and comments. We also remove the top 5% of most freqently commenting users in each month, which is heuristic for removing bots. In experiments, 5% was seen to improve performance on the subreddit analogy task compared to removing 10%, 2% or none of the most frequently commenting users.

Select which month's data and tuned community2vec model to see its metrics and parameters. The model you select for each month's data achieved the highest accuracy on a [predetermined analogy set](https://github.com/UMassCDS/IHOP/tree/main/ihop/resources/analogies) across many experiments.
"""

KMEANS_METRICS_MD = """The quality of topics produced by clustering the community embeddings can be subjective. However, a human should be able to intuit each cluster's theme based on the dominant topic, such as 'politics', 'music', or 'sports'. Changing the number of clusters should correspond to a change in granularity of the labels, e.g. 'sports' vs 'basketball'.

Generating gold-standard labels for clusters isn't feasible, so instead we can rely on metrics that measure the overlap and dispersion of clusters:

- **Silhouette Coefficient**: Ranges between -1 (clustering is totally incorrect) and 1 (clusters are dense and well separated), scores around 0 indicate overlapping clusters.

- **Calinski-Harabasz Index**: Higher for models with clusters that are dense and well separated.

- **Davies-Bouldin Index**: Measures separation/similarity between clusters by taking ratio of within-cluster distances to between-cluster distances. The minimum is zero, lower scores indicate better separation

These scores can help you compare different groupings of the data using the same number of clusters. The scores for your current model are:
"""

TSNE_DESCRIPTION_MD = """T-distributed Stochastic Neighbor Embedding (t-SNE) is a way of visualizing high-dimensional data in 2D or 3D, so we can actually make sense of it. However, like map projections of the globe into 2D, this can cause distortions, as described in [How to Use t-SNE Effectively](https://distill.pub/2016/misread-tsne/). This visualization should be treated as a guide to help explore subreddit clusters interatively, not a definitive representation of how similar subreddits are to each other.

Each point is a subreddit and the color marks the cluster it gets assigned to. """

MONTH_SELECTION_SECTION = dash.html.Div(
    children=[
        dash.html.H2("Select the time period"),
        dash.dcc.Markdown(MONTH_SELECTION_MD),
        dash.dcc.Dropdown(
            list(MODEL_DIRS.keys()), id="month-dropdown", value="December 2022"
        ),
        dash.html.Br(),
        dash.html.H2("Model Details"),
        dash.dcc.Loading(
            dash.html.Div(
                children=[
                    dash.html.Div(id="analogy-results-section"),
                    dash.html.Div(id="community2vec-params-section"),
                ]
            ),
        ),
        dash.html.Br(),
    ]
)

DOWNLOAD_CLUSTER_CSV = dash.html.Div(
    [
        dbc.Button("Download Clusters CSV", id="cluster_csv_button", n_clicks=0), 
        dash.dcc.Download(id="download_cluster_csv"),
    ]
)


# First section of page, define KMeans paramters, train model button and metrics values and explanation
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
            dbc.Button("Train clustering model", id="clustering_button",),
        ]
    ),
    dash.html.Br(),
    dash.html.Div(
        children=[
            dash.html.H2("K-means clustering metrics"),
            dash.dcc.Markdown(KMEANS_METRICS_MD),
            dash.dcc.Loading(
                id="loading-metrics",
                type="default",
                children=[
                    dash.html.Article(id="cluster-metrics"), 
                    DOWNLOAD_CLUSTER_CSV,
                    ],
            ),
            dash.html.Br(),
        ]
    ),
]

# Subreddit dropdown menu
SUBREDDIT_DROPDOWN = [
    dash.html.Label("Select subreddits"),
    dash.dcc.Dropdown(
        multi=True,
        id="subreddit-dropdown",
    ),
]

# Cluster number dropdown menu
CLUSTER_DROPDOWN = [
    dash.html.Label("Select clusters"),
    dash.dcc.Dropdown(multi=True, id="cluster-dropdown"),
]

# Final section on the page, allows user to filter subreddits they're interested in
# by subreddit or cluster number, then display the clusters in the table.
# Also includes buttons for highlighting selected clusters in graph viz
SUBREDDIT_FILTERING_SECTION = dash.html.Div(
    children=[
        dash.html.H2("Filter by Subreddits and Clusters"),
        dbc.Row(
            children=[
                dbc.Col(
                    children=SUBREDDIT_DROPDOWN,
                    width=8,
                ),
                dbc.Col(
                    dash.html.Div(
                        # Clicking this button will grey out all but the clusters from the drop down in the graph visualization.
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
                        # This switch toggles including all the subreddits in the cluster
                        # in the table results, even if the user hasn't included them in the subreddit filter.
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

# Entire page body, all sections are contained within
BODY = dash.html.Div(
    children=[
        dash.html.H1("Community2Vec Subreddit Clusters"),
        INTRODUCTION_SECTION,
        dash.html.Br(),
        dbc.Accordion(
            start_collapsed=False,
            always_open=True,
            active_item=["item-0", "item-1", "item-2"],
            children=[
                dbc.AccordionItem(
                    MONTH_SELECTION_SECTION, title="Load Community2Vec Model"
                ),
                dbc.AccordionItem(
                    KMEANS_PARAM_SECTION,
                    title="K-means Cluster Parameters",
                ),
                dbc.AccordionItem(
                    MODEL_PLOT_SECTION, title="t-SNE Visualization of Clusters"
                ),
            ],
        ),
        dash.html.Br(),
        SUBREDDIT_FILTERING_SECTION,
        dash.html.Br(),
        # Stores the dataframe with cluster assignments and the name of the cluster model (for exporting labels)
        dash.dcc.Store(id="cluster-assignment"),
        # Stores the list of subbreddits available in the c2v model, for user to select in drop down
        dash.dcc.Store(id="subreddits"),
        # Store tsne coordinates for the loaded c2v model, so you only
        # have to compute them once
        dash.dcc.Store(id="tsne-df"),
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


def get_model_param_details(metrics_dict):
    """Returns Dash component display about the parameters used for training the model.

    :param metrics_dict: _description_
    :type metrics_dict: _type_
    :return: _description_
    :rtype: _type_
    """
    tmp_dict = metrics_dict.copy()
    skip_keys = [
        ic2v.MODEL_ID_KEY,
        ic2v.ANALOGY_ACC_KEY,
        ic2v.DETAILED_ANALOGY_KEY,
        ic2v.CONTEXTS_PATH_KEY,
    ]
    num_users = tmp_dict.pop(ic2v.NUM_USERS_KEY)
    max_comments = tmp_dict.pop(ic2v.MAX_COMMENTS_KEY)
    remaining_params = list()
    for k, v in tmp_dict.items():
        if k not in skip_keys:
            remaining_params.append(f"* {k}: {v}")
    params_list = "\n".join(remaining_params)
    return [
        dash.html.H3("Community2Vec Model Parameters"),
        dash.dcc.Markdown(
            f"""A total of {num_users:,} different users were used to train this model, with a maximum of {max_comments} comments from a single user.
             The community2vec model parameters used are as follows, please refer to the [Gensim Word2Vec documentation](https://radimrehurek.com/gensim/models/word2vec.html) for more detailed descriptions:\n{params_list}
             """
        ),
    ]


def get_model_accuracy_display(month_str, metrics_dict):
    """Returns the Dash component display section for model accuracy

    :param month_str: str, identifier for the model
    :param analogy_accuracy: float
    :param detailed_analogy_str: str, comma separated string returned by Gensim's analogy solver tool
    """
    detailed_analogy_str = metrics_dict[ic2v.DETAILED_ANALOGY_KEY]
    analogy_accuracy = metrics_dict[ic2v.ANALOGY_ACC_KEY]
    detailed_acc_items = [
        acc_item.split(":") for acc_item in detailed_analogy_str.split(",")
    ]
    # The last acc in the list is always the total accuracy
    total_acc = detailed_acc_items.pop()
    markdown_acc_list = "\n".join(
        [f"* {item[0]}: {item[1]}" for item in detailed_acc_items]
    )
    return [
        dash.html.H3("Subreddit Analogy Performance"),
        dash.dcc.Markdown(
            f"""This {month_str} model achieved an accuracy of {analogy_accuracy*100:.2f}% on the subreddit analogy task or {total_acc[1]} analogies solved correctly, broken down as:\n{markdown_acc_list}"""
        ),
    ]


@app.callback(
    dash.Output("tsne-df", "data"),
    dash.Output("subreddit-dropdown", "options"),
    dash.Output("analogy-results-section", "children"),
    dash.Output("community2vec-params-section", "children"),
    dash.Input("month-dropdown", "value"),
)
def load_vector_model(selected_month):
    """
    :param n_clicks: _description_
    :param selected_month: _description_
    """
    logger.info("Selected month: %s", selected_month)
    current_model_path = pathlib.Path(MODEL_DIRS[selected_month])
    c2v_model = ic2v.GensimCommunity2Vec.load(current_model_path)
    logger.info("Community2Vec model loaded from %s", current_model_path)

    sorted_subreddits = sorted(c2v_model.get_index_to_key())

    logger.info("Starting to get tsne values for %s", current_model_path)
    tsne_df = iv.load_tsne_dataframe(current_model_path / TSNE_CSV_NAME)
    tsne_json = iv.jsonify_stored_df(tsne_df)
    logger.info("Tsne coordinates stored for %s", current_model_path)

    logger.info("Loading metrics for %s", current_model_path)
    with (current_model_path / "metrics.json").open() as metrics_file:
        metrics_dict = json.load(metrics_file)
    model_params = get_model_param_details(metrics_dict)
    accuracy_results = get_model_accuracy_display(selected_month, metrics_dict)

    return tsne_json, sorted_subreddits, accuracy_results, model_params


@app.callback(
    dash.Output("cluster-assignment", "data"),
    dash.Output("model-name", "children"),
    dash.Output("cluster-metrics", "children"),
    dash.Input("clustering_button", "n_clicks"),
    dash.State("n-clusters", "value"),
    dash.State("random-seed", "value"),
    dash.Input("month-dropdown", "value"),
    dash.Input("tsne-df", "data"),
    running=[(dash.Output("clustering_button", "disabled"), True, False)]
)
def train_clusters(n_clicks, n_clusters, random_seed, c2v_identifier, tsne_json_data):
    """Trains kmeans cluster with given number of clusters and random seed.

    :param n_clicks: int, button click indicator which triggers training the model (value unused)
    :param n_clusters: int, number of clusters to create
    :param random_seed: int, random seed for reproducibility
    :param c2v_identifier: str, name of community2vec model currently loaded, usually named with a time frame
    :param tsne_json_data: tsne coordinates serialized as json

    :return: Return cluster assignments with a model name as a json {'name': 'model name', 'clusters': json_serialized_pandas_dataframe}
    """
    tsne_df = iv.unjsonify_stored_df(tsne_json_data)

    c2v_model = ic2v.GensimCommunity2Vec.load(MODEL_DIRS[c2v_identifier])
    model_name = f"{c2v_identifier} Kmeans Cluster Assignment {n_clusters} clusters and random state {random_seed}"

    # TODO: eventually we may want to support different types of models. The ClusteringModelFactory should allow that fairly easily
    cluster_model = ihop.clustering.ClusteringModelFactory.init_clustering_model(
        ihop.clustering.ClusteringModelFactory.KMEANS,
        c2v_model.get_normed_vectors(),
        c2v_model.get_index_as_dict(),
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
    dash.State("month-dropdown", "value"),
)
def get_cluster_visualization(
    cluster_json,
    subreddit_selection,
    cluster_selection,
    is_only_highlight_selection,
    community2vec_identifier,
):
    """Build the plotly visualization for a model

    :param cluster_json: The json bundled cluster-assignment data
    :param subreddit_selection: list of user selected subreddit values
    :param cluster_dropdown: list of user selected cluster values
    :param is_only_highlight_selection: int, number of times the user has clicked on the hightlight subreddit button, used as a boolean value (0 or > 0)
    :param community2vec_identifier: str, name of community2vec model currently loaded, usually named with a time frame
    :param train_cluster_clicks: int, number of clicks on train cluster model, triggers plot creation
    """
    # The user hasn't create cluster labelings yet, the graph will be empty
    if cluster_json is None:
        return {}

    # The model_name column of the dataframe always contains the cluster ID
    model_name = cluster_json["name"]
    logger.info("Updating graph visualization, model: %s", model_name)
    # The model name column is intentionally not categorical, so we can grey out unselected values in the plot
    cluster_df = iv.unjsonify_stored_df(cluster_json["clusters"], [model_name])
    cluster_df[CLUSTER_ASSIGNMENT_DISPLAY_NAME] = cluster_df[model_name]
    cluster_set = set()

    # Collect up all selected cluster assignments for highlighting
    if is_only_highlight_selection is not None:
        logger.info("Highlight selected clusters was clicked")
        logger.info("Subreddit list given: %s", subreddit_selection)
        logger.info("Cluster list given: %s", cluster_selection)

        # Clusters included because a subreddit is selected
        if subreddit_selection is not None:
            selected_subreddits_df = cluster_df[
                cluster_df["subreddit"].isin(subreddit_selection)
            ]
            subreddit_clusters = selected_subreddits_df[model_name].unique()
            cluster_set.update(subreddit_clusters)
            logger.info("Clusters used by subreddits: %s", cluster_set)

        # Cluster id is explicitly selected
        if cluster_selection is not None:
            cluster_set.update(cluster_selection)

    # Display name is a cluster id or 'other', it's just for the scatter plot display
    # Set unselected clusters to 'other'
    logger.info("Highlighted clusters will be: %s", cluster_set)

    cluster_df = iv.assign_other_category_column(
        cluster_df,
        model_name,
        CLUSTER_ASSIGNMENT_DISPLAY_NAME,
        cluster_set,
        UNSELECTED_CLUSTER_KEY,
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

    # Grey out unselected clusters
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
        title=f"t-SNE Projection of {community2vec_identifier} Community2Vec Subreddit Clusterings<br>with {model_name}",
    )

    fig = go.Figure(data=figpx.data, layout=layout)
    logger.info("Figure successfully generated")
    return fig


@app.callback(
    dash.Output("cluster-dropdown", "options"), dash.Input("n-clusters", "value")
)
def set_cluster_dropdown(n_clusters):
    """Give the cluster selection options the cluster index numbers to choose from"""
    return [i for i in range(int(n_clusters))]


@app.callback(
    dash.Output("subreddit-cluster-table", "children"),
    dash.Input("cluster-assignment", "data"),
    dash.Input("subreddit-dropdown", "value"),
    dash.Input("cluster-dropdown", "value"),
    dash.Input("show-in-cluster-neighbors", "on"),
)
def get_display_table(
    cluster_json,
    selected_subreddits,
    selected_clusters,
    is_show_cluster_neighbors,
):
    """Builds the DataTable showing all selected subreddits and clusters

    :param cluster_json: The json bundled cluster-assignment data
    :param subreddit_selection: list of user selected subreddit values
    :param selected_clsuters: list of user selected subreddit values
    :param is_show_cluster_neighbors:
    :parma train_cluster_clicks: int, whether the user has clicked the cluster train button. If they haven't, the table will be empty
    """
    if cluster_json is None:
        return dash.dash_table.DataTable(
            [],
            [
                {"name": "subreddit", "id": "subreddit"},
                {"name": "cluster assignment", "id": "cluster assignment"},
            ],
        )

    logger.info("Generating display table")
    logger.info("Selected subreddits: %s", selected_subreddits)
    logger.info("Show cluster neighbors option: %s", is_show_cluster_neighbors)
    logger.info("Selected clusters: %s", selected_subreddits)
    model_name = cluster_json["name"]
    cluster_df = iv.unjsonify_stored_df(cluster_json["clusters"], [model_name])

    if selected_subreddits is None:
        selected_subreddits = []

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

@app.callback(
    dash.Output("download_cluster_csv", "data"),
    dash.Input("cluster_csv_button", "n_clicks"),
    dash.Input("cluster-assignment", "data"),
    prevent_initial_call=True
)
def download_cluster_csv(n_clicks, cluster_json):
    trigger = dash.ctx.triggered_id
    logger.info("Cluster download triggered by '%s'", trigger)
    if trigger == "cluster_csv_button":
        logger.info("Cluster download button clicked times: %s", n_clicks)
        model_name = cluster_json["name"]
        cluster_df = iv.unjsonify_stored_df(cluster_json["clusters"], [model_name])
        cluster_df[CLUSTER_ASSIGNMENT_DISPLAY_NAME] = cluster_df[model_name]
        csv_name = f"{model_name}.csv"
        logger.info("Downloading clustering data to %s", csv_name)
        return dash.dcc.send_data_frame(cluster_df.to_csv, csv_name, index=False)
    else:
        raise dash.exceptions.PreventUpdate

if __name__ == "__main__":
    print("Starting IHOP subreddit visualization application")
    logger.info("Starting app")
    try:
        # TODO Plotly handles logging strangely, so use logger.info or workaround to not silence logging,
        # see https://community.plotly.com/t/logging-debug-messages-suppressed-in-callbacks/17854
        app.run_server()
    except Exception as e:
        logger.error(e)
