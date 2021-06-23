from data import SAMPLE_RATE
import random

import wandb
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical


LABELS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
LB = LabelEncoder()
LB.fit(LABELS)
SAMPLE_RATE = 16000


def load_data(fname):
    """Loads saved numpy dataset."""
    npz = np.load(fname)
    return npz["labels"], npz["files"], npz["audio"], npz["specs"]


def save_data(data, fname):
    """Saves data in uncompressed numpy format."""
    labels = np.array([d[0] for d in data])
    files = np.array([d[1] for d in data])
    audio = np.array([d[2] for d in data])
    specs = np.array([d[3] for d in data])
    np.savez(fname, labels=labels, files=files, audio=audio, specs=specs)


def one_hot_labels(y):
    """Transform an array of labels into a one-hot matrix."""
    return to_categorical(LB.transform(y))


def make_confusion_matrix(preds, labels):
    preds = np.argmax(preds, axis=1)
    y_true = np.argmax(labels, axis=1)
    return wandb.plot.confusion_matrix(
        y_true=y_true, preds=preds, class_names=LB.classes_
    )


def misclassification_table(preds, spectrograms, labels, audio):
    pred_labels = np.argmax(preds, axis=1)
    y_true = np.argmax(labels, axis=1)
    misclassified_idx = list(np.where(pred_labels != y_true)[0])
    columns = ["label", "prediction", "audio", "spectrogram"]
    table_data = []
    # Select ten misclassified examples at random
    # TODO: Show softmax output for each misclassified example in table
    for idx in random.sample(misclassified_idx, 32):
        table_data.append(
            [
                LB.classes_[y_true[idx]],
                LB.classes_[pred_labels[idx]],
                wandb.Audio(audio[idx], sample_rate=SAMPLE_RATE),
                wandb.Image(spectrograms[idx]),
            ]
        )
    return wandb.Table(data=table_data, columns=columns)
