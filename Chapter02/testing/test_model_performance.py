import pytest
import numpy as np
from typing import Union
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from huggingface_hub import hf_hub_download
from sklearn.metrics import classification_report

import joblib

@pytest.fixture
def test_dataset() -> Union[np.array, np.array]:
    """
    Returns a tuple containing the test dataset and the corresponding labels.
    The dataset is the wine dataset, with the label being True for class 2 and False otherwise.
    The dataset is split into a training and test set using `train_test_split` with a random state of 42.
    """
    # Load the dataset
    X, y = load_wine(return_X_y=True)
    # create an array of True for 2 and False otherwise
    y = y == 2
    # Train and test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    return X_test, y_test

@pytest.fixture
def model() -> sklearn.ensemble._forest.RandomForestClassifier:
    """
    Returns a trained RandomForestClassifier model downloaded from the Hugging Face Hub.
    The model was trained on the wine dataset and is used for testing the performance of the model.
    """
    REPO_ID = "electricweegie/mlewp-sklearn-wine"
    FILENAME = "rfc.joblib"
    model = joblib.load(hf_hub_download(REPO_ID, FILENAME))
    return model


def test_model_inference_types(model, test_dataset):
    """
    Tests that the model's predict method returns a numpy array and that the test dataset is composed of numpy arrays.
    """
    
    assert isinstance(model.predict(test_dataset[0]), np.ndarray)
    assert isinstance(test_dataset[0], np.ndarray)
    assert isinstance(test_dataset[1], np.ndarray)

def test_model_performance(model, test_dataset):
    """
    Tests the performance of the model on the test dataset.
    The performance is measured using the F1-score and precision metrics.
    The model is expected to achieve an F1-score greater than 0.95 and a precision greater than 0.9 for class False, and an F1-score greater than 0.8 and a precision greater than 0.8 for class True.
    """
    metrics = classification_report(y_true=test_dataset[1], y_pred=model.predict(test_dataset[0]), output_dict=True)
    assert metrics['False']['f1-score'] > 0.95
    assert metrics['False']['precision'] > 0.9
    assert metrics['True']['f1-score'] > 0.8
    assert metrics['True']['precision'] > 0.8