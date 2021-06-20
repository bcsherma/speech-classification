import argparse
import random

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
from tensorflow.keras import Sequential, layers, losses, optimizers
from wandb.keras import WandbCallback

import wandb

N_CLASSES = 10
SAMPLE_RATE = 16000
SPECTROGRAM_SHAPE = (99, 161, 1)


def load_data():
    labels_and_fnames = np.load("labels_and_fnames.npy")
    audio = np.load("audio.npy")
    spectrograms = np.load("spectrograms.npy")
    labels, fnames = np.split(labels_and_fnames, 2, axis=1)
    return labels, fnames, audio, spectrograms


def build_model(**kwargs):
    model = Sequential(
        [
            layers.Input(shape=SPECTROGRAM_SHAPE),
            layers.Conv2D(8, 2, activation="relu"),
            layers.Conv2D(8, 2, activation="relu"),
            layers.MaxPool2D(pool_size=(2, 2)),
            layers.Dropout(rate=kwargs["dropout"]),
            layers.Conv2D(16, 3, activation="relu"),
            layers.Conv2D(16, 3, activation="relu"),
            layers.MaxPool2D(pool_size=(2, 2)),
            layers.Dropout(rate=kwargs["dropout"]),
            layers.Conv2D(32, 3, activation="relu"),
            layers.MaxPool2D(pool_size=(2, 2)),
            layers.Dropout(rate=kwargs["dropout"]),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dense(128, activation="relu"),
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


def make_confusion_matrix(predictions, true_values, labels):
    matrix = confusion_matrix(true_values, predictions)
    plt.figure(figsize=(12, 12))
    plt.matshow(matrix, cmap=plt.cm.Blues)
    plt.colorbar()
    plt.ylabel("True value")
    plt.xlabel("Predictions")
    plt.xticks(ticks=range(len(labels)), labels=labels)
    plt.yticks(ticks=range(len(labels)), labels=labels)
    return plt.gcf()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--optimizer", type=str, default="rmsprop")
    parser.add_argument("--learning_rate", type=float, default=0.001)
    return parser.parse_args()


if __name__ == "__main__":
    run = wandb.init(config=parse_args())
    labels, fnames, audio, spectrograms = load_data()
    enc = OneHotEncoder()
    enc.fit(labels)
    labels = enc.transform(labels).toarray()
    (
        train_labels,
        test_labels,
        train_fnames,
        test_fnames,
        train_audio,
        test_audio,
        train_specs,
        test_specs,
    ) = train_test_split(labels, fnames, audio, spectrograms, train_size=0.8)
    model = build_model(**wandb.config)
    model.fit(
        train_specs,
        train_labels,
        validation_data=(test_specs, test_labels),
        batch_size=32,
        epochs=run.config["epochs"],
        callbacks=[WandbCallback()],
    )

    y_pred = np.argmax(model.predict(test_specs), axis=1)
    y_true = np.argmax(test_labels, axis=1)
    run.log(
        {
            "confusion_matrix": make_confusion_matrix(
                y_pred, y_true, list(enc.categories_[0])
            )
        }
    )

    misclassified_idx = list(np.where(y_pred != y_true)[0])
    columns = ["label", "audio", "spectrogram", "filename", "prediction"]
    table_data = []
    # Select ten misclassified examples at random
    for idx in random.sample(misclassified_idx, 10):
        table_data.append(
            [
                enc.categories_[0][y_true[idx]],
                wandb.Audio(test_audio[idx], sample_rate=SAMPLE_RATE),
                wandb.Image(test_specs[idx]),
                test_fnames[idx][0],
                enc.categories_[0][y_pred[idx]],
            ]
        )
    table = wandb.Table(data=table_data, columns=columns)
    run.log({"misclassifications": table})
