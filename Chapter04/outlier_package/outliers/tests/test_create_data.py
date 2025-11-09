import numpy
import pytest
import outliers.utils.data


@pytest.fixture()
def dummy_data():
    """
    Returns a numpy array of a dataset containing outliers.
    The dataset is generated using the create_data function from outliers.utils.data.
    """
    data = outliers.utils.data.create_data()
    return data

def test_data_is_numpy(dummy_data):
    """
    Test that the dummy data is a numpy array.

    Parameters
    ----------
    dummy_data : numpy.ndarray
        A numpy array of a dataset containing outliers.

    Returns
    -------
    None
    """

    assert isinstance(dummy_data, numpy.ndarray)

def test_data_is_large(dummy_data):
    """
    Test that the dummy data is a large dataset.

    Parameters
    ----------
    dummy_data : numpy.ndarray
        A numpy array of a dataset containing outliers.

    Returns
    -------
    None
    """
    assert len(dummy_data)>100
