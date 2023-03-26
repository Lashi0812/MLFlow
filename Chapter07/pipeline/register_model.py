import click
import mlflow
import logging

logging.basicConfig(level=logging.INFO,format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

@click.command(help="This program to register the model")
@click.option("--mlflow_run_id",default=None)
@click.option("--registered_model_name",default="finetune_model")

def task(mlflow_run_id,registered_model_name):
    if mlflow_run_id is None or mlflow_run_id == "None":
        logger.info("No model to register.")
        return
    
    with mlflow.start_run(run_name="register_model") as tracker:
        logged_model = f"runs:/{mlflow_run_id}/model"
        logger.info(f"logged model uri is : {logged_model}")
        mlflow.register_model(model_uri=logged_model,name=registered_model_name)
        mlflow.set_tag("pipeline_step",__file__)
        
    logger.info("finished registering model to %s", registered_model_name)
    
if __name__ == "__main__":
    task()