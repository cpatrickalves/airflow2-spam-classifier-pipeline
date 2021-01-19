from datetime import timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
from airflow.decorators import dag, task
import logging as logger
from spam import NaiveBayesSolver
import requests, zipfile
import os, sys, psutil


# These args will get passed on to each operator
default_args = {
    'owner': 'Patrick Alves',
    'depends_on_past': False,
    'email': ['cpatrickalves@gmail.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    #'retries': 1,
    #'retry_delay': timedelta(minutes=5),
    'sla': timedelta(minutes=15)
}

def checkIfProcessRunning(processName):
    '''
    Check if there is any running process that contains the given name processName.
    '''
    #Iterate over the all the running process
    for proc in psutil.process_iter():
        try:
            # Check if process name contains the given name string.
            if processName.lower() in proc.name().lower():
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return False;


@dag(default_args=default_args, schedule_interval=None, start_date=days_ago(2))
def antispam_ml_pipeline():
    """
    ### A Machine Learning pipeline to train an antispam model
    """

    @task()
    def download_training_set(file_url, ts_name):
        """
        Download the training set as a ZIP file from some URL
        """

        tmpfile = "downloads/tmp.zip"

        # Create directories
        for d in ["data", "downloads"]:
            if not os.path.exists(d):
                os.makedirs(d)

        logger.info("Downloading training data ...")
        r = requests.get(file_url)
        if r.status_code == 200:
            with open(tmpfile, "wb") as f:
                f.write(r.content)
            logger.info(f"File saved at {tmpfile}")
        else:
            logger.error(f"Error download file from {file_url}")
            raise ValueError(f"Error download file from {file_url}")

        try:
            logger.info(f"Unziping training data from {tmpfile}...")
            with zipfile.ZipFile(tmpfile, 'r') as zip_ref:
                zip_ref.extractall("data")
            dataset_path = os.path.join("data", ts_name)
            return dataset_path

        except Exception as e:
            logger.exception(e)
            raise ValueError(f"Error while unzip file {tmpfile}")

    @task()
    def clean_data(dataset_path):
        """
        Some cleaning
        """

        logger.info("Cleaning training data ...")
        return dataset_path

    @task()
    def train_model(dataset_path: str , model_name: str = "testmodel"):
        """
        Train model
        """
        nb = NaiveBayesSolver.NaiveBayesSolver()

        model_file = f"models/{model_name}"

        nb.train(os.path.join(dataset_path, "train"), model_file)
        nb.predict(os.path.join(dataset_path, "test"), model_file)
        return model_name

    # Deploy the demo app (Streamlit app)
    # The above Operator works, but it will lock the shell while the
    # Streamlit app is running, so the task will neve end.
    #deploy = BashOperator(
    #    task_id='deploy_app',
    #    bash_command="streamlit run /airflow/dags/spam/demo_app.py --server.port 80 &",
    #    sla= timedelta(seconds=5)
    #)

    @task()
    def deploy_app(model_path:str):
        """
        Deploy a demo of the spam classifier as a Streamlit app
        """
        logger.info("Deploying model in SpamTest app ...")
        if checkIfProcessRunning('streamlit'):
            logger.info("Streamlit app already running")
        else:
            logger.info("Starting streamlit app ...")
             # Run this from python to avoid locking the shell
            os.system("streamlit run /airflow/dags/spam/demo_app.py --server.port 80 &")

    # TS URL
    file_url = "https://github.com/cpatrickalves/airflow2-spam-classifier-pipeline/files/5814551/TS-2021003-124601.zip"
    ts_name = "TS-2021003-124601"  # TS-YYYYMMDD-HHMMSS

    # Run the pipeline
    dataset_path = download_training_set(file_url, ts_name)
    clean_dataset = clean_data(dataset_path)
    model_path = train_model(clean_dataset)
    deploy_app(model_path)

model_training_dag = antispam_ml_pipeline()