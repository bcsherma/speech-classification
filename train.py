import argparse

import utils

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
    model.compile(
        loss=losses.categorical_crossentropy, metrics="accuracy", optimizer=opt
    )
    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--filters", type=int, default=8)
    parser.add_argument("--dense_units", type=int, default=128)
    parser.add_argument("--model_name", type=str, default="")
    parser.add_argument("--save_plots", action="store_true")
    parser.add_argument("--test", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    config = parse_args()
    config_dict = {
        "dataset": config.dataset,
        "dense_units": config.dense_units,
        "epochs": config.epochs,
        "dropout": config.dropout,
        "optimizer": config.optimizer,
        "filters": config.filters,
        "learning_rate": config.learning_rate,
    }
    run = wandb.init(
        name=config.name, config=config_dict, job_type="training", save_code=True
    )
    artifact = run.use_artifact(config.dataset)
    artifact.download(root="data/")
    labels, fnames, audio, spectrograms = utils.load_data("data/train.npz")
    labels = utils.one_hot_labels(labels)
    vlabels, vfnames, vaudio, vspectrograms = utils.load_data("data/validation.npz")
    vlabels = utils.one_hot_labels(vlabels)
    model = build_model(**wandb.config)
    model.fit(
        spectrograms,
        labels,
        batch_size=32,
        epochs=run.config["epochs"],
        validation_data=(vspectrograms, vlabels),
        callbacks=[WandbCallback()],
    )
    if config.save_plots:
        preds = model.predict(vspectrograms)
        run.log({"Confusion_Matrix": utils.make_confusion_matrix(preds, vlabels)})
        run.log(
            {
                "Misclassifications": utils.misclassification_table(
                    preds, vspectrograms, vlabels, vaudio
                )
            }
        )
    if config.model_name:
        fname = f"models/{config.model_name}.keras"
        model.save(fname)
        model_artifact = wandb.Artifact(
            name=config.model_name, type="model", metadata=dict(run.config)
        )
        model_artifact.add_file(fname)
        run.log_artifact(model_artifact)

    if config.test:
        labels, fnames, audio, spectrograms = utils.load_data("data/test.npz")
        labels = utils.one_hot_labels(labels)
        loss, acc = model.evaluate(spectrograms, labels)
        run.log({"test_loss": loss, "test_accuracy": acc})

    run.finish()
