import logging
import click 
import os
import mlflow

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

_steps = [
    "download_data",
    "fine_tuning_model",
    "register_model",
    "inference_model"
]

@click.command()
@click.option("--steps",default="all",type=str)
@click.option("--tracking_uri",default="http://127.0.0.1:81",type=str)
@click.option("--s3_uri",default="http://127.0.0.1:9000",type=str)

def run_pipeline(steps,tracking_uri,s3_uri):
    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    os.environ["AWS_ACCESS_KEY_ID"] = "minio"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = s3_uri
    
    EXPERIMENT_NAME = "Inference"
    mlflow.set_experiment(EXPERIMENT_NAME)
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    logger.info(f"pipeline experiment_id is {experiment.experiment_id}")
    
    active_steps = steps.split(",") if steps != "all" else _steps
    logger.info("pipeline active steps to execute in this run: %s", active_steps)
    
    #! this line of code is added to avoid runtime error RESOURCE_DOES_NOT_EXIST when mlflow call start_run()
    #! When we use the mlflow cli run its create run_id but this run_id is not exits in server
    #! to avoid this issue manually delete this run_id
    #? refer ---  https://github.com/mlflow/mlflow/issues/4830
    os.environ.pop("MLFLOW_RUN_ID",None)
    
    with mlflow.start_run(run_name="pipeline",nested=True,experiment_id=experiment.experiment_id) as active_run:
        if "download_data" in active_steps:
            download_run = mlflow.run(".",entry_point="download_data",
                                      parameters={},run_name="download",
                                      experiment_id=experiment.experiment_id)
            download_run = mlflow.tracking.MlflowClient().get_run(download_run.run_id)
            filepath_uri = download_run.data.params["local_folder"]
            logger.info('downloaded data is located locally in folder: %s', filepath_uri)
            logger.info(download_run)
            
        if "finetune_model" in active_steps:
            finetune_run = mlflow.run(".",entry_point="finetune_model",
                                      parameters={"data_path":filepath_uri},run_name="finetune",
                                      experiment_id=experiment.experiment_id)
            finetune_run_id = finetune_run.run_id
            finetune_run = mlflow.tracking.MlflowClient().get_run(finetune_run_id)
            logger.info(finetune_run)
            
        if "register_model" in active_steps:
            if finetune_run_id is not None and finetune_run_id != 'None':
                register_model_run = mlflow.run(".", "register_model", parameters={"mlflow_run_id": finetune_run_id},
                                                run_name="register_model",experiment_id=experiment.experiment_id,)
                register_model_run = mlflow.tracking.MlflowClient().get_run(register_model_run.run_id)
                logger.info(register_model_run)
            else:
                logger.info("no model to register since no trained model run id.")
                
        if "inference_model" in active_steps:
            if finetune_run_id is not None and finetune_run_id != 'None':
                inference_model_run = mlflow.run(".", "inference_model", parameters={"finetune_model_uri": finetune_run_id},
                                                run_name="inference_model",experiment_id=experiment.experiment_id,)
                inference_model_run = mlflow.tracking.MlflowClient().get_run(inference_model_run.run_id)
                logger.info(inference_model_run)
            else:
                logger.info("no finetune model to create inference model")
    logger.info('finished mlflow pipeline run with a run_id = %s', active_run.info.run_id)


if __name__ == "__main__":
    run_pipeline()