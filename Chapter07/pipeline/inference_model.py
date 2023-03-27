import os
import json
import tempfile
import shutil
import logging
from functools import partial
from pathlib import Path
from cachetools import LRUCache


import pandas as pd
import click
import mlflow
import torch

import flash
from flash.text import TextClassificationData


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

CONDA_ENV = str(Path(__file__).parents[1]/"conda.yaml")
EXPERIMENT_NAME = "Inference"
MODEL_ARTIFACT_PATH  = "inference_pipeline_model"


mlflow.set_experiment(EXPERIMENT_NAME)
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
logger.info(experiment.experiment_id)

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
            


# input and output format
input = json.dumps([{"name":"review","type":"string"}])
output = json.dumps([{"name":"sentiment","type":"string"}])

signature = mlflow.pyfunc.ModelSignature.from_dict({"inputs":input,
                                                    "outputs":output})

@click.command(help="This program creates a multi-step inference pipeline model.")
@click.option("--finetune_model_uri",type=str)
def task(finetune_model_uri):    
    with mlflow.start_run(run_name="wrap model") as model_run:
        finetune_model_uri = f"runs:/{finetune_model_uri}/model"
        inference_model_uri = f"runs:/{model_run.info.run_id}/{MODEL_ARTIFACT_PATH}"
        mlflow.pyfunc.log_model(artifact_path=MODEL_ARTIFACT_PATH,
                                conda_env=CONDA_ENV,
                                python_model=InferencePipeline(finetune_model_uri,inference_model_uri),
                                signature=signature,
                                registered_model_name=MODEL_ARTIFACT_PATH)
        logger.info(f"finetune model uri {finetune_model_uri}")
        logger.info(f"inference model uri {inference_model_uri}")
        mlflow.log_param("finetune model uri",finetune_model_uri)
        mlflow.log_param("inference model uri",inference_model_uri)
        mlflow.set_tag("pipeline_step",__file__)
    
    logger.info("finished logging inference pipeline model")    

# input = {"review":["what a disappointing movie","Great movie", "Great movie"]}
# input_df = pd.DataFrame(input)
# input_df

# with  mlflow.start_run(run_name="wrap_inference") as infer_run:
#     loaded_model = mlflow.pyfunc.load_model(inference_model_uri)
#     input_df["response"] = loaded_model.predict(input_df)
#     tempdir = tempfile.mkdtemp()
#     try:
#         response_path = os.path.join(tempdir,"response.csv")
#         input_df.to_csv(response_path)
#         mlflow.log_artifact(response_path)
#     finally:
#         shutil.rmtree(tempdir)
    


if __name__ == "__main__":
    task()