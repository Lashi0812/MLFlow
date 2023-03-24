# %% [markdown]
"""
# Notebook for fine-tuning the pretrained model
"""
# %% [markdown]
"""
## Importing the libraries
"""

# %%
import os
import flash
import torch 
import mlflow

from flash.text import TextClassificationData,TextClassifier
import torchmetrics



# %%
os.environ["MLFLOW_TRACKING_URI"] = "http://127.0.0.1:81"
os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://127.0.0.1:9000"

# %%
data_module = TextClassificationData.from_csv(
    input_field="review",
    target_fields="sentiment",
    train_file="../../data/imdb/train.csv",
    test_file ="../../data/imdb/test.csv",
    val_file="../../data/imdb/valid.csv",
    batch_size=64   
)


# %%
classifier_model = TextClassifier(num_classes=data_module.num_classes,
                                  labels=data_module.labels,
                                  backbone="prajjwal1/bert-tiny",
                                  metrics=torchmetrics.F1Score(num_classes=data_module.num_classes))
trainer = flash.Trainer(max_epochs=3,
                        gpus=torch.cuda.device_count())

# %%
EXPERIMENT_NAME = "Code_Version"
if not mlflow.get_experiment_by_name(EXPERIMENT_NAME):
    mlflow.create_experiment(name=EXPERIMENT_NAME)

mlflow.set_experiment(EXPERIMENT_NAME)
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
print(f"Experiment Id : {experiment.experiment_id}")


# %%
mlflow.pytorch.autolog()
with mlflow.start_run(experiment_id=experiment.experiment_id,run_name="interactive_mode") as model_tracking_run:
    trainer.finetune(model=classifier_model,datamodule=data_module)
    mlflow.log_params(classifier_model.hparams)
    
# %%
run_id = model_tracking_run.info.run_id
print(f"run_id : {run_id} and lifecycle_stage {mlflow.get_run(run_id).info.lifecycle_stage} ")
# %%
logged_model_uri = f"runs:/{run_id}/model"
mlflow.register_model(logged_model_uri,"finetune_model")
# %%
