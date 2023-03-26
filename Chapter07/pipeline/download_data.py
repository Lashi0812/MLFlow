# %%
import click
import logging
import mlflow
from flash.core.data.utils import download_data

logging.basicConfig(level=logging.INFO,format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

# %%
@click.command(help="This program will download the data which is used for the sentiment classifcation")
@click.option("--download_url",default="https://pl-flash-data.s3.amazonaws.com/imdb.zip",
              help="remote url for where the data is located")
@click.option("--local_folder",default="../data/",help="Local store")



def task(download_url,local_folder):
    with mlflow.start_run(run_name="download") as ml_run:
        logger.info(f"Downloading data from {download_url}")
        download_data(url=download_url,path=local_folder)
        mlflow.log_param("download_url",download_url)
        mlflow.log_param("local_folder",local_folder)
        mlflow.log_param("download run id",ml_run.info.run_id)
        mlflow.set_tag("pipeline_step",__file__)
        mlflow.log_artifacts(local_dir=local_folder,artifact_path="data")
    
    logger.info(f"finished downloading data to {local_folder}")
    
if __name__ == "__main__":
    task()
        