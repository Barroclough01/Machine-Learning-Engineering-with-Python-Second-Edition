# BASED ON EXAMPLE FROM MLFLOW DOCS
# https://github.com/mlflow/mlflow/blob/master/examples/prophet/train.py
import pandas as pd
from prophet import Prophet

from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics

import mlflow
import mlflow.pyfunc
from typing import Any
from mlflow.models.signature import infer_signature

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


class FbProphetWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        """
        Initialize a FbProphetWrapper instance.

        Parameters
        ----------
        model : Prophet
            The Prophet model to be wrapped.

        Returns
        -------
        None
        """
        self.model = model
        super().__init__()

    def load_context(self, context):
        """
        Load the model context from the MLflow server.

        Parameters
        ----------
        context : mlflow.pyfunc.ModelContext
            The model context to be loaded.

        Returns
        -------
        None

        Notes
        -----
        This function is called by the MLflow server to load the model context
        from the server. The context is loaded into the model object, and
        can be accessed by the model object's methods.
        """
        # context is provided by MLflow at load time. If you need to
        # restore external artifacts, do it here. We keep this simple
        # because the Prophet model object is already serialized by MLflow
        # when logging the pyfunc wrapper.
        return None

    def predict(self, context: Any, model_input: pd.DataFrame) -> pd.DataFrame:
        """Make predictions using the wrapped Prophet model.

        Parameters
        ----------
        context : mlflow.pyfunc.ModelContext
            MLflow model context (provided at prediction time).
        model_input : pd.DataFrame
            Input dataframe. Two supported modes:
              - If it contains a column named "periods", the first value
                is taken as an integer number of periods to forecast and
                a future dataframe is created via Prophet's
                `make_future_dataframe`.
              - If it contains a column named "ds", it is treated as a
                dataframe of datestamps to run predict() on directly.

        Returns
        -------
        pd.DataFrame
            The DataFrame returned by Prophet's `predict` method.
        """
        # If user passed periods (e.g., during local scoring tests), create future df
        try:
            if isinstance(model_input, dict):
                # backward compatible: accept dict with 'periods'
                periods = int(model_input.get("periods", [0])[0])
                future = self.model.make_future_dataframe(periods=periods)
                return self.model.predict(future)

            if "periods" in model_input.columns:
                periods = (
                    int(model_input.loc[0, "periods"]) if not model_input.empty else 0
                )
                future = self.model.make_future_dataframe(periods=periods)
                return self.model.predict(future)

            # If input contains datestamps (ds), predict for those rows
            if "ds" in model_input.columns:
                # Ensure ds is datetime
                mi = model_input.copy()
                mi["ds"] = pd.to_datetime(mi["ds"], errors="coerce")
                mi = mi.dropna(subset=["ds"])  # drop invalid dates
                return self.model.predict(mi)

            # Fallback: try to coerce the whole input to numeric periods
            # and create a small future frame
            # (this is defensive; prefer explicit 'periods' or 'ds')
            periods = 0
            if not model_input.empty:
                # try to use the first column as integer periods
                try:
                    periods = int(
                        pd.to_numeric(model_input.iloc[0, 0], errors="coerce")
                    )
                except Exception:
                    periods = 0
            future = self.model.make_future_dataframe(periods=periods)
            return self.model.predict(future)
        except Exception as e:
            # Raise the exception so MLflow can capture the failure; include
            # a helpful message for debugging.
            raise RuntimeError(f"Error in FbProphetWrapper.predict: {e}") from e


seasonality = {"yearly": True, "weekly": True, "daily": True}


def train_predict(df_all_data, df_all_train_index, seasonality_params=seasonality):
    """
    Train a Prophet model on the given data and log the model and its metrics to MLflow.

    Parameters
    ----------
    df_all_data : pandas.DataFrame
        The dataframe containing all the data to be split into train and test sets.
    df_all_train_index : int
        The index to split the dataframe into train and test sets.
    seasonality_params : dict, optional
        A dictionary containing the yearly, weekly, and daily seasonality parameters for the Prophet model.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]
        A tuple containing the predicted values, the train dataframe, and the test dataframe.

    Notes
    -----
    This function will log the model and its metrics to MLflow. The model will be logged with the name "model" and the metrics will be logged with the name "rmse".
    """
    # grab split data
    df_train = df_all_data.copy().iloc[0:df_all_train_index]
    df_test = df_all_data.copy().iloc[df_all_train_index:]

    # Useful for multiple runs (only doing one run in this sample notebook)
    with mlflow.start_run():
        # create Prophet model
        model = Prophet(
            yearly_seasonality=seasonality_params["yearly"],
            weekly_seasonality=seasonality_params["weekly"],
            daily_seasonality=seasonality_params["daily"],
        )
        # train and predict
        model.fit(df_train)

        # Evaluate Metrics
        df_cv = cross_validation(
            model, initial="540 days", period="180 days", horizon="90 days"
        )
        df_p = performance_metrics(df_cv)

        # Print out metrics
        print("  CV: \n%s" % df_cv.head())
        print("  Perf: \n%s" % df_p.head())

        # Log parameter, metrics, and model to MLflow
        mlflow.log_metric("rmse", df_p.loc[0, "rmse"])

        # Try to infer a model signature and attach an input example so MLflow
        # does not emit warnings about missing signatures and examples.
        try:
            if "ds" in df_test.columns:
                example_input = df_test[["ds"]].head(10)
            else:
                example_input = pd.DataFrame({"periods": [10]})

            try:
                example_output = model.predict(example_input)
            except Exception:
                example_output = model.predict(df_test.head(10))

            signature = infer_signature(example_input, example_output)
            mlflow.pyfunc.log_model(
                "model",
                python_model=FbProphetWrapper(model),
                signature=signature,
                input_example=example_input,
            )
        except Exception:
            # If signature inference fails, log the model without signature to avoid aborting training
            mlflow.pyfunc.log_model("model", python_model=FbProphetWrapper(model))

        print(
            "Logged model with URI: runs:/{run_id}/model".format(
                run_id=mlflow.active_run().info.run_id
            )
        )

    predicted = model.predict(df_test)
    return predicted, df_train, df_test


if __name__ == "__main__":
    # Read in Data
    df = pd.read_csv("../../Chapter01/forecasting/rossman_store_data/train.csv")
    df.rename(columns={"Date": "ds", "Sales": "y"}, inplace=True)
    # Filter out store and item 1
    df_store1 = df[(df['Store'] == 1)].reset_index(drop=True)

    train_index = int(0.8 * df_store1.shape[0])
    predicted, df_train, df_test = train_predict(
        df_all_data=df_store1, df_all_train_index=train_index, seasonality_params=seasonality
    )
