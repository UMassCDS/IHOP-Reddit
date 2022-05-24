"""This modules stores pre-defined groupings of subreddits that are useful for searching and visualizations.
Members of collections go in text files in this module, one collection member per line in the file.
"""
from collections import namedtuple
import importlib.resources

Collection = namedtuple("Collection", "name file description")

# Define name of grouping: file name, description
COLLECTIONS_LIST = [
    Collection(
        "Denigrating toward immigrants",
        "denigrating_language_toward_immigrants.txt",
        "Subreddits that frequently have denigrating language about immigrants",
    )
]


SUBREDDIT_GROUPINGS = {c.name: c for c in COLLECTIONS_LIST}


def get_collection_members(collection_name):
    """Returns the members in the collection

    :param collection_name: str, name of collection to retrieve
    """
    results = []
    with importlib.resources.open_text(
        "ihop.resources.collections", SUBREDDIT_GROUPINGS[collection_name].file
    ) as f:
        results = [l.strip() for l in f.readlines()]
    return results
