import logging
import click 
import os
import mlflow

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

_steps = [
    "download_data",
    "fine_tuning_model",
    "register_model"
]

@click.command()
@click.option("--steps",default="all",type=str)

def run_pipeline(steps):
    os.environ["MLFLOW_TRACKING_URI"] = "http://127.0.0.1:81"
    os.environ["AWS_ACCESS_KEY_ID"] = "minio"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://127.0.0.1:9000"
    
    EXPERIMENT_NAME = "pipeline"
    mlflow.set_experiment(EXPERIMENT_NAME)
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    logger.info(f"pipeline experiment_id is {experiment.experiment_id}")
    
    active_steps = steps.split(",") if steps != "all" else _steps
    logger.info("pipeline active steps to execute in this run: %s", active_steps)
    
    with mlflow.start_run(run_name="pipeline",nested=True,experiment_id=experiment.experiment_id) as active_run:
        if "download_data" in active_steps:
            download_run = mlflow.run(".",entry_point="download_data",
                                      parameters={},run_name="download",
                                      experiment_id=experiment.experiment_id)
            download_run = mlflow.tracking.MlflowClient().get_run(download_run.run_id)
            filepath_uri = download_run.data.params["local_folder"]
            logger.info('downloaded data is located locally in folder: %s', filepath_uri)
            logger.info(download_run)

    logger.info('finished mlflow pipeline run with a run_id = %s', active_run.info.run_id)


if __name__ == "__main__":
    run_pipeline()