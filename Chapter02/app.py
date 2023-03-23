import torch
import mlflow
import flash
from flash.core.data.utils import download_data
from flash.text import TextClassificationData,TextClassifier


if __name__ == "__main__": 

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("mlflow1")

    experiment = mlflow.get_experiment_by_name("mlflow1")

    # download_data("https://pl-flash-data.s3.amazonaws.com/imdb.zip","../data/")

    data_module = TextClassificationData.from_csv(
        input_field="review",
        target_fields="sentiment",
        train_file="./data/imdb/train.csv",
        val_file="./data/imdb/valid.csv",
        test_file="./data/imdb/test.csv",
        batch_size=64,
    )

    model = TextClassifier(backbone="prajjwal1/bert-tiny",
                        num_classes=data_module.num_classes)

    trainer = flash.Trainer(max_epochs=1,gpus=torch.cuda.device_count())

    mlflow.pytorch.autolog()

    with mlflow.start_run(experiment_id=experiment.experiment_id,
                        run_name="colab"):
        trainer.finetune(model,datamodule=data_module,
                        strategy="freeze")
        # trainer.test()