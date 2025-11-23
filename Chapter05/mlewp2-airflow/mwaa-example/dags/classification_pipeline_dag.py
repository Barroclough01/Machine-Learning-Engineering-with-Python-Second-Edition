import datetime
from datetime import timedelta
import pendulum
from airflow import DAG
from airflow.operators.python import PythonOperator

import logging

logging.basicConfig(level=logging.INFO)

default_args = {
    "owner": "Andrew McMahon",
    "depends_on_past": False,
    # Use pendulum instead of deprecated days_ago
    "start_date": pendulum.now().subtract(days=2),
    "email": ["example@example.com"],
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


def get_data():
    pass


def train_model():
    pass


def persist_model():
    pass


with DAG(
    dag_id="classification_pipeline",
    default_args=default_args,
    schedule="@daily",
    catchup=False,
) as dag:
    logging.info("DAG started ...")

    get_data_task = PythonOperator(
        task_id="get_data",
        python_callable=get_data,
    )

    train_model_task = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
    )

    persist_model_task = PythonOperator(
        task_id="persist_model",
        python_callable=persist_model,
    )

    get_data_task >> train_model_task >> persist_model_task


# #instantiate DAG
# dag = DAG(
#     'classification_pipeline',
#     default_args=default_args,
#     description="Basic pipeline for classifying the Wine Dataset",
#     schedule_interval=timedelta(days=1),
# )


# get_data = BashOperator(
#     task_id='get_data',
#     bash_command='python3 /usr/local/airflow/scripts/get_data.py',
#     dag=dag,
# )

# train_model= BashOperator(
#     task_id='train_model',
#     depends_on_past=False,
#     bash_command='python3 /usr/local/airflow/scripts/train_model.py',
#     retries=3,
#     dag=dag,
# )

# # Persist to MLFlow
# persist_model = BashOperator(
#     task_id='persist_model',
#     depends_on_past=False,
#     bash_command='python3 /usr/local/airflow/scripts/persist_model.py',
#     retries=3,
#     dag=dag,
# )

# get_data >> train_model >> persist_model
