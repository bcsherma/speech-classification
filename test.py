import argparse
from tensorflow.keras import callbacks

import wandb
from wandb.keras import WandbCallback
from tensorflow.keras.models import load_model

import utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="test-data:latest")
    return parser.parse_args()


if __name__ == "__main__":
    config = parse_args()
    run = wandb.init(config=config, job_type="testing")
    dataset = run.use_artifact(config.dataset)
    dataset.download(root="data/")
    labels, fnames, audio, spectrograms = utils.load_data("data/test.npz")
    labels = utils.one_hot_labels(labels)
    model = run.use_artifact(config.model)
    run.log(model.metadata)
    model_path = model.download(root="models/")
    model_name = config.model.split(":")[0]
    model = load_model(f"models/{model_name}.keras")
    model.summary()
    model.evaluate(spectrograms, labels, callbacks=[WandbCallback()])
    preds = model.predict(spectrograms)
    run.log({"Confusion_Matrix": utils.make_confusion_matrix(preds, labels)})
    run.log(
        {
            "Misclassifications": utils.misclassification_table(
                preds, spectrograms, labels, audio
            )
        }
    )
    run.finish()
