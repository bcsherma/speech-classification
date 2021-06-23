import os
import random
import re
from glob import glob
import argparse

import numpy as np
from scipy import signal
from scipy.io import wavfile
from sklearn.model_selection import train_test_split

import utils
import wandb

SAMPLE_RATE = 16000
LABELS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
EXAMPLES_PER_LABEL = 350
DATA_PATH = os.path.join("./audio")
DATA_DESC = """
Download the dataset from kaggle.com/c/tensorflow-speech-recognition-challenge.
Extract train/audio from train.7z and place audio/ in the same dir as this code.
""".strip()


def transform_wav(filename):
    """Transforms wav file into an np array."""
    sample_rate, samples = wavfile.read(filename)
    assert sample_rate == SAMPLE_RATE
    if len(samples) < SAMPLE_RATE:
        # Make sure each audio sample is exactly 1 second long
        samples = np.pad(samples, (SAMPLE_RATE - len(samples), 0))
    return samples


def log_spectrogram(audio, window_size=20, step_size=10, eps=1e-10):
    """Converts np array of audio samples into spectrogram."""
    # Directly lifted from kaggle.com/davids1992/speech-representation-and-data-exploration
    nperseg = int(round(window_size * SAMPLE_RATE / 1e3))
    noverlap = int(round(step_size * SAMPLE_RATE / 1e3))
    _, _, spec = signal.spectrogram(
        audio,
        fs=SAMPLE_RATE,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=False,
    )
    return np.log(spec.T.astype(np.float32) + eps)


def organize_files_by_label(dir):
    """
    Reads audio samples labeled by subdir and returns a dict mapping labels
    to filenames.
    """
    files = glob(os.path.join(dir, r"*/*" + ".wav"))
    regex = re.compile(r".+/(\w+)/(\w+)\.wav$")
    label_to_file = dict()
    for path in files:
        match = regex.match(path)
        if match:
            label, filename = match.groups()
            if label not in LABELS:
                continue
            if label not in label_to_file:
                label_to_file[label] = [filename]
            else:
                label_to_file[label].append(filename)
    return {
        l: random.sample(examples, EXAMPLES_PER_LABEL)
        for l, examples in label_to_file.items()
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = {
        "data_set_name": args.name,
        "sample_rate": SAMPLE_RATE,
        "examples_per_class": EXAMPLES_PER_LABEL,
    }
    run = wandb.init(
        name=args.name,
        job_type="dataset-preparation",
        config=config,
        save_code=True,
    )
    original_data = wandb.Artifact(
        "source-data", type="raw-data", description=DATA_DESC
    )
    original_data.add_reference(
        "https://www.kaggle.com/c/tensorflow-speech-recognition-challenge"
    )
    run.use_artifact(original_data)

    # Construct dataset from wav files
    label_to_files = organize_files_by_label(DATA_PATH)
    data = []
    for l in label_to_files:
        for f in label_to_files[l]:
            audio = transform_wav(os.path.join(DATA_PATH, l, f + ".wav"))
            spectrogram = log_spectrogram(audio)
            spectrogram = spectrogram.reshape(tuple(list(spectrogram.shape) + [1]))
            data.append([l, f, audio, spectrogram])

    # Re-order the dataset
    random.shuffle(data)

    # Split dataset into train, validation, test (60/20/20)
    train, test = train_test_split(data, train_size=0.8)
    train, validation = train_test_split(train, train_size=0.75)

    # Save datasets to disk
    utils.save_data(train, "data/train.npz")
    utils.save_data(validation, "data/validation.npz")
    utils.save_data(test, "data/test.npz")

    # Log class distributions per dataset to wandb
    for data, title in [(train, "train"), (validation, "validation"), (test, "test")]:
        labels = [d[0] for d in data]
        table = [[l, labels.count(l)] for l in utils.LABELS]
        table = wandb.Table(data=table, columns=["label", "count"])
        run.log(
            {
                f"{title}-class-distribution": wandb.plot.bar(
                    table, "label", "count", title=f"{title} class distribution"
                )
            }
        )

        # Save dataset version as wandb artifacts
        artifact = wandb.Artifact(
            args.name,
            type="dataset",
            description=f"train/val/test data stored as compressed npz",
        )
        artifact.add_file("data/train.npz")
        artifact.add_file("data/validation.npz")
        artifact.add_file("data/test.npz")
        run.log_artifact(artifact)
