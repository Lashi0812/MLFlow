import click
import logging
import mlflow

import torch
import flash
from flash.text import TextClassificationData,TextClassifier

logging.basicConfig(level=logging.INFO,format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

@click.command(help="This program fine tune the bert model for sentiment classification")
@click.option("--foundation_model",default="prajjwal1/bert-tiny",
              help="This is back bone for the model")
@click.option("--finetune_strategy",default="freeze")
@click.option("--data_path",default="../data")

def task(foundation_model,finetune_strategy,data_path):
    data_module = TextClassificationData.from_csv(
        input_field="review",
        target_fields="sentiment",
        train_file =f"{data_path}/imdb/train.csv",
        val_file=f"{data_path}/imdb/valid.csv",
        test_file=f"{data_path}/imdb/test.csv",
        batch_size = 64
    )
    model = TextClassifier(num_classes=data_module.num_classes,
                           labels=data_module.labels,
                           backbone=foundation_model)
    
    trainer = flash.Trainer(max_epochs=1,gpus=torch.cuda.device_count())
    
    mlflow.pytorch.autolog()
    with mlflow.start_run(run_name="finetune") as tracker:
        trainer.finetune(model,data_module,strategy=finetune_strategy)
        
        mlflow.log_params(model.hparams)
        
        run_id = tracker.info.run_id
        logger.info(f"run_id : {run_id} ,lifecycle_stage :{mlflow.get_run(run_id).info.lifecycle_stage}")
        
        mlflow.log_param("finetune run id",run_id)
        mlflow.set_tag("pipeline_step",__file__)
        
if __name__ == "__main__":
    task()
        