#%%
import os
import json
from functools import partial
from pathlib import Path
from cachetools import LRUCache
import tempfile
import shutil

import pandas as pd
import click
import mlflow
import torch

import flash
from flash.text import TextClassificationData

# %%
@click.command()
@click.option("--tracking_uri",type=str,nargs=1)
@click.option("--s3_uri",type=str,nargs=1)
def task(tracking_uri,s3_uri):
    os.environ["AWS_ACCESS_KEY_ID"] = "minio"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = s3_uri
    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    print(tracking_uri)

    # %%
    EXPERIMENT_NAME = "Inference"
    mlflow.set_experiment(EXPERIMENT_NAME)
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    print(experiment.experiment_id)
    # %%
    class InferencePipeline(mlflow.pyfunc.PythonModel):
        def __init__(self,finetune_model_uri,inference_model_uri) -> None:
            self.cache = LRUCache(maxsize=100)
            self.finetune_model_uri = finetune_model_uri
            self.inference_model_uri = inference_model_uri
        
        def load_context(self, context):
            self.finetune_classifier = mlflow.pytorch.load_model(self.finetune_model_uri)
            self.trainer = partial(flash.Trainer(gpus=torch.cuda.device_count()).predict,model=self.finetune_classifier,
                                        output="labels")
            
        def preprocess_step_cache(self,row):
            if row[0] in self.cache:
                print(f"found cache result")
                return self.cache[row[0]]
            
        def sentiment_classifier(self,row):
            
            # preprocess: check cache
            caches_result = self.preprocess_step_cache(row)
            if caches_result is not None:
                return caches_result
            
            # model inference
            data_module = TextClassificationData.from_lists(
                predict_data=[row[0]],
                batch_size=10
            )
            pred_label = self.trainer(datamodule=data_module)
            
            # postprocessing: add additional information
            response = json.dumps({
                "response":{
                    "predict_label":pred_label
                },
                "model_metadata":{
                    "finetune_uri":self.finetune_model_uri,
                    "inference_model_uri":self.inference_model_uri
                }
            })
            
            # cache the results
            self.cache[row[0]] =  response
            
            return response
        def predict(self, context, model_input):
            results = model_input.apply(self.sentiment_classifier,axis=1,
                                        result_type="broadcast")   
            return results    
                

    # %%
    # input and output format
    input = json.dumps([{"name":"review","type":"string"}])
    output = json.dumps([{"name":"sentiment","type":"string"}])

    signature = mlflow.pyfunc.ModelSignature.from_dict({"inputs":input,
                                                        "outputs":output})


    # %%
    CONDA_ENV = str(Path(__file__).parents[1]/"conda.yaml")
    print(CONDA_ENV)
    # %%
    MODEL_ARTIFACT_PATH  = "inference_pipeline_model"
    with mlflow.start_run(run_name="wrap model") as model_run:
        finetune_model_uri = "runs:/d41b177213884a8dad30cc84f9562dd2/model"
        inference_model_uri = f"runs:/{model_run.info.run_id}/{MODEL_ARTIFACT_PATH}"
        mlflow.pyfunc.log_model(artifact_path=MODEL_ARTIFACT_PATH,
                                conda_env=CONDA_ENV,
                                python_model=InferencePipeline(finetune_model_uri,inference_model_uri),
                                signature=signature)
        
    # %%
    input = {"review":["what a disappointing movie","Great movie", "Great movie"]}
    input_df = pd.DataFrame(input)
    input_df

    with  mlflow.start_run(run_name="wrap_inference") as infer_run:
        loaded_model = mlflow.pyfunc.load_model(inference_model_uri)
        input_df["response"] = loaded_model.predict(input_df)
        tempdir = tempfile.mkdtemp()
        try:
            response_path = os.path.join(tempdir,"response.csv")
            input_df.to_csv(response_path)
            mlflow.log_artifact(response_path)
        finally:
            shutil.rmtree(tempdir)
        
        
    # %%
    print(input_df)
    # %%


if __name__ == "__main__":
    task()