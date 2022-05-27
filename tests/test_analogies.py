import os

import pytest

import ihop.resources.analogies as ihopanalogies


@pytest.fixture
def analogies_csv(fixture_dir):
    return [os.path.join(fixture_dir, "test_analogies.csv")]


def test_default_analogies():
    analogies = ihopanalogies.get_analogies()
    assert len(analogies) == 113842
    assert ("boston", "redsox", "toronto", "Torontobluejays") in analogies
    assert ("philadelphia", "sixers", "tulsa", "Thunder") in analogies
    assert ("Buffalo", "buffalobills", "sanfrancisco", "49ers") in analogies
    assert ("montreal", "Habs", "phoenix", "Coyotes") in analogies
    assert ("Drexel", "philadelphia", "umass", "amherst") in analogies


def test_generate_analogies(analogies_csv):
    analogies = ihopanalogies.get_analogies(analogies_csv)
    assert analogies == [
        ("a", "b", "c", "d"),
        ("a", "b", "e", "f"),
        ("c", "d", "e", "f"),
    ]
