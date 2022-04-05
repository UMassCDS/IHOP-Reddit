"""Module to define handling of analogies for community2vec
"""
import csv
import importlib.resources


def generate_analogies(seed_terms):
    """Generates 4-tuples of analogies that can be fed to Gensim KeyedVector
    for solving.
    Input `[('a','b'),('c','d')]` returns `[('a','b','c','d')]` for
    'a is to b as c is to d`

    :param seed_terms: list of 2-tuples, each tuple is a symmetric side of an analogy
    """
    results = list()
    for i in range(len(seed_terms)):
        a, b = seed_terms[i]
        for j in range(i + 1, len(seed_terms)):
            c, d = seed_terms[j]
            results.append((a, b, c, d))
    return results


def get_analogies(csv_path_list=None):
    """Returns analogies from CSV or the default subreddit algebra analogies
     as list of 4-tuples for benchmarking community to vec.
     CSV files should be formated as follows generate sum(range(n)) analogies
     like "a is to b as c is to d" when there are n rows:
     a,b
     c,d

    :param csv_path_list: list of str/Path, optional paths to headerless csv files
    """
    analogies = list()
    if csv_path_list is None:
        analogy_contents = [
            x
            for x in importlib.resources.contents("ihop.resources.analogies")
            if x.endswith(".csv")
        ]
        for analogy_file in analogy_contents:
            analogy_reader = csv.reader(
                importlib.resources.open_text("ihop.resources.analogies", analogy_file)
            )
            current_file_analogies = generate_analogies([row for row in analogy_reader])
            analogies.extend(current_file_analogies)
    else:
        for analogy_file in csv_path_list:
            with open(analogy_file) as analogies_f:
                analogy_reader = csv.reader(analogies_f)
                current_file_analogies = generate_analogies(
                    [row for row in analogy_reader]
                )
                analogies.extend(current_file_analogies)

    return analogies
