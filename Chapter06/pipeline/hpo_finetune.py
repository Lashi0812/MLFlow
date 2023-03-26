
# %%
import os
import logging
from pathlib import Path


import mlflow
import torchmetrics

import flash
from flash.text import TextClassificationData,TextClassifier

from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.integration.mlflow import mlflow_mixin
# %%

logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s:%(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
logger = logging.getLogger()

@mlflow_mixin
def trainable_model(config,data_dir,num_epochs=3,num_gpus=0):
    data_module = TextClassificationData.from_csv(
        input_field="review",
        target_fields="sentiment",
        train_file=f"{data_dir}/imdb/train.csv",
        val_file=f"{data_dir}/imdb/valid.csv",
        test_file=f"{data_dir}/imdb/test.csv",
        batch_size=config["batch_size"]
    )
    model = TextClassifier(num_classes=data_module.num_classes,
                           backbone=config["foundational_model"],
                           learning_rate=config["lr"],
                           optimizer=config["optimizer_type"],
                           labels=data_module.labels,
                           metrics=torchmetrics.F1Score(num_classes=data_module.num_classes))
    
    mlflow.pytorch.autolog()
    metrics = {"loss":"val_cross_entropy","f1":"val_f1score"}
    
    trainer = flash.Trainer(max_epochs=num_epochs,
                            callbacks=[TuneReportCallback(metrics=metrics,on="validation_end")],
                            gpus=num_gpus)
    trainer.finetune(model=model,
                     datamodule=data_module,
                     strategy=config["finetune_strategy"])
    
    mlflow.log_param("batch_size",config["batch_size"])
    mlflow.set_tag("pipeline_step",__file__)
    

def run_hpo_model(num_samples=10,num_epochs=3,gpus_per_trial=0,
                  tracking_uri=None,experiment_name="ray_tune_hpo"):
    data_dir = str(Path.cwd().parents[0]/"data")
    
    # step 1 : set MLflow config
    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://127.0.0.1:9000"
    os.environ["AWS_ACCESS_KEY_ID "] = "minio"
    os.environ["AWS_SECRET_ACCESS_KEY "] = "minio123"
    
    mlflow.set_tracking_uri = tracking_uri
    mlflow.set_experiment(experiment_name=experiment_name)
    
    
    # set 2 Create the config file for search_space and mlflow_config
    config = {
        "batch_size": tune.choice([32,64,128]),
        "lr":tune.loguniform(lower=1e-4,upper=1e-1),
        "foundational_model":"prajjwal1/bert-tiny",        
        "optimizer_type":"Adam",
        "finetune_strategy":"freeze",
        "mlflow":{
            "tracking_uri":mlflow.get_tracking_uri(),
            "experiment_name":experiment_name
        }
    }
    
    # define the trainable object
    trainable = tune.with_parameters(trainable=trainable_model,
                                     data_dir = data_dir,
                                     num_epochs=num_epochs,
                                     num_gpus=gpus_per_trial)
    
    # call the tune.run
    analysis = tune.run(
        run_or_experiment=trainable,
        resources_per_trial={"cpu":1,"gpu":gpus_per_trial},
        metric="f1",
        mode="max",
        config=config,
        num_samples=num_samples,
        name="hpo_tuning_model",
    )
    
    logging.info(f"Best hyperparameters found were : {analysis.best_config}")
    

def task():
    run_hpo_model(num_samples=2,num_epochs=1,gpus_per_trial=0,
                  tracking_uri="http://127.0.0.1:81",
                  experiment_name="hpo_tuning_model")
    
    
if __name__ == "__main__":
    task()