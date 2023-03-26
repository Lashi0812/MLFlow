
# %%
import json
import os
from pathlib import Path
from functools import partial

import pandas as pd
import mlflow

import flash
from flash.text import TextClassificationData


# %%
os.environ["MLFLOW_TRACKING_URI"] = "http://127.0.0.1:81"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://127.0.0.1:9000"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"
os.environ["AWS_ACCESS_KEY_ID"] = "minio"

# %%
EXPERIMENT_NAME = "Inference"
mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.set_tracking_uri("http://127.0.0.1:81")
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
print(f"Experiment_ID : {experiment.experiment_id}")

# %%
class InferenceModel(mlflow.pyfunc.PythonModel):
    def __init__(self,finetune_model_uri) -> None:
        self.finetune_model_uri = finetune_model_uri
        
    def load_context(self, context):
        self.finetune_text_classifier = mlflow.pytorch.load_model(model_uri=self.finetune_model_uri)
        self.trainer = partial(flash.Trainer().predict,model=self.finetune_text_classifier,output="labels")
        
        
    def predict(self, context, model_input):
        data = TextClassificationData.from_data_frame(
            input_field="reviews",
            predict_data_frame=model_input,
            batch_size=2
        )
        results = self.trainer(datamodule=data)
        return results
    

# %%
input = json.dumps([{"name":"reviews","type":"string"}])
output = json.dumps([{"name":"sentiment","type":"string"}])

signature = mlflow.pyfunc.ModelSignature.from_dict({"inputs":input,"outputs":output})

# %%
CONDA_ENV = str(Path(__file__).parents[1]/"conda.yaml")
print(CONDA_ENV)
# %%

MODEL_ARTIFACT_PATH = "inference_model"
with mlflow.start_run(run_name="wrapped_inference_model") as active_run:
    finetune_uri = "runs:/ce8c1b97d4c4464496288a8f947aedbd/model"
    inference_uri = f"runs:/{active_run.info.run_id}/{MODEL_ARTIFACT_PATH}"
    mlflow.pyfunc.log_model(artifact_path=MODEL_ARTIFACT_PATH,
                            conda_env=CONDA_ENV,
                            python_model=InferenceModel(finetune_uri),
                            signature=signature)
    

# %%
input = {"reviews":["Superb a movie","Waste of time"]}
input_df = pd.DataFrame(input)
input_df
# %%
with mlflow.start_run(run_name="warp_inference_pipeline") as prediction_run:
    loaded_model = mlflow.pyfunc.load_model(inference_uri)
    results = loaded_model.predict(input_df)
    input_df["prediction"] = results[0]
    input_df.to_csv("prediction.csv")
    mlflow.log_artifact("prediction.csv")
    
    
# %%
print(results)


# %%
