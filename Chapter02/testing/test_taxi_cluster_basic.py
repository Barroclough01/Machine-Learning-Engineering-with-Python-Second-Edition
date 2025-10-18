import pytest
from model import cluster_and_label
from helper import get_taxi_data

@pytest.mark.skip(reason="From edition 1, does not work due to not uploading taxi data in repo")
def test_cluster_and_label():
    """
    Tests the cluster_and_label function.

    This test should pass if the function returns a dictionary.

    The test is skipped because the taxi data is not uploaded to the repository.
    """
    df = get_taxi_data()
    results = cluster_and_label(df)
    assert isinstance(results, dict)
