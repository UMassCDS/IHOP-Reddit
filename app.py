"""Visualize subreddit clusters
Run using `python app.py` and visit http://127.0.0.1:8050
"""
import argparse
import logging

from dash import Dash, dcc, html
import plotly.express as px
import pandas as pd

import ihop.utils

logger = logging.getLogger(__name__)

app = Dash(__name__)

app.layout = html.Div(children=[html.H1(children="Hello Dash")])


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

