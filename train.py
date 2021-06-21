import argparse
import uuid

import utils
import numpy as np

from tensorflow.keras import Sequential, layers, losses, optimizers
from wandb.keras import WandbCallback

import wandb

N_CLASSES = 10
SAMPLE_RATE = 16000
LABELS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
SPECTROGRAM_SHAPE = (99, 161, 1)


def build_model(**kwargs):
    model = Sequential(
        [
            layers.Input(shape=SPECTROGRAM_SHAPE),
            layers.BatchNormalization(),
            layers.Conv2D(kwargs["filters"], 2, activation="relu"),
            layers.Conv2D(kwargs["filters"], 2, activation="relu"),
            layers.MaxPool2D(pool_size=(2, 2)),
            layers.Dropout(rate=kwargs["dropout"]),
            layers.Conv2D(kwargs["filters"] * 2, 3, activation="relu"),
            layers.Conv2D(kwargs["filters"] * 2, 3, activation="relu"),
            layers.MaxPool2D(pool_size=(2, 2)),
            layers.Dropout(rate=kwargs["dropout"]),
            layers.Conv2D(kwargs["filters"] * 4, 3, activation="relu"),
            layers.MaxPool2D(pool_size=(2, 2)),
            layers.Dropout(rate=kwargs["dropout"]),
            layers.Flatten(),
            layers.Dense(kwargs["dense_units"], activation="relu"),
            layers.BatchNormalization(),
            layers.Dense(kwargs["dense_units"], activation="relu"),
            layers.BatchNormalization(),
            layers.Dense(10, activation="softmax"),
        ]
    )
    model.summary()
    if kwargs["optimizer"] == "rmsprop":
        opt = optimizers.RMSprop(learning_rate=kwargs["learning_rate"])
    elif kwargs["optimizer"] == "adam":
        opt = optimizers.Adam(learning_rate=kwargs["learning_rate"])
    model.compile(loss=losses.binary_crossentropy, metrics="accuracy", optimizer=opt)
    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="training-data:latest")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--optimizer", type=str, default="rmsprop")
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--filters", type=int, default=8)
    parser.add_argument("--dense_units", type=int, default=128)
    return parser.parse_args()


if __name__ == "__main__":
    config = parse_args()
    run = wandb.init(config=config)
    artifact = run.use_artifact(config.dataset)
    artifact.download(root="data/")
    labels, fnames, audio, spectrograms = utils.load_data("data/train.npz")
    labels = utils.one_hot_labels(labels)
    model = build_model(**wandb.config)
    model.fit(
        spectrograms,
        labels,
        batch_size=32,
        epochs=run.config["epochs"],
        validation_split=0.25,
        callbacks=[WandbCallback()],
    )
    preds = model.predict(spectrograms)
    run.log({"Confusion_Matrix": utils.make_confusion_matrix(preds, labels)})
    run.log(
        {
            "Misclassifications": utils.misclassification_table(
                preds, spectrograms, labels, audio
            )
        }
    )
    random_name = uuid.uuid4().hex[:8]
    fname = f"models/{random_name}.keras"
    model.save(fname)
    model_artifact = wandb.Artifact(
        name="convnet", type="model", metadata=dict(run.config)
    )
    model_artifact.add_file(fname)
    run.log_artifact(model_artifact)
    run.finish()
