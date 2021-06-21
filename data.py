import os
import random
import re
from glob import glob

import numpy as np
from scipy import signal
from scipy.io import wavfile
from sklearn.model_selection import train_test_split

import wandb

SAMPLE_RATE = 16000
LABELS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
EXAMPLES_PER_LABEL = 1000
DATA_PATH = os.path.join("./audio")


def transform_wav(filename):
    sample_rate, samples = wavfile.read(filename)
    assert sample_rate == SAMPLE_RATE
    if len(samples) < SAMPLE_RATE:
        # Make sure each audio sample is exactly 1 second long
        samples = np.pad(samples, (SAMPLE_RATE - len(samples), 0))
    return samples


def log_spectrogram(audio, window_size=20, step_size=10, eps=1e-10):
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


if __name__ == "__main__":
    config = {
        "sample_rate": SAMPLE_RATE,
        "examples_per_class": EXAMPLES_PER_LABEL,
    }
    run = wandb.init(job_type="dataset-preparation", config=config, save_code=True)
    desc = """
    Download the dataset from https://www.kaggle.com/c/tensorflow-speech-recognition-challenge
    extract train/audio and place audio in the working directory that this code
    is running within. This just saves the location of the original data.
    """.strip()
    original_data = wandb.Artifact("source-data", type="raw-data", description=desc)
    original_data.add_reference(
        "https://www.kaggle.com/c/tensorflow-speech-recognition-challenge"
    )
    run.use_artifact(original_data)

    label_to_files = organize_files_by_label(DATA_PATH)
    labels = []
    filenames = []
    audio = []
    for l in label_to_files:
        for f in label_to_files[l]:
            labels.append(l)
            filenames.append(f)
            audio.append(transform_wav(os.path.join(DATA_PATH, l, f + ".wav")))
    labels = np.array(labels)
    filenames = np.array(filenames)
    audio = np.array(audio)
    spectrograms = np.stack([log_spectrogram(clip) for clip in audio])
    spectrograms = spectrograms.reshape(tuple(list(spectrograms.shape) + [1]))

    (
        train_labels,
        test_labels,
        train_fnames,
        test_fnames,
        train_audio,
        test_audio,
        train_specs,
        test_specs,
    ) = train_test_split(
        labels,
        filenames,
        audio,
        spectrograms,
        train_size=0.8,
    )

    np.savez(
        "data/train.npz",
        specs=train_specs,
        labels=train_labels,
        audio=train_audio,
        files=train_fnames,
    )
    np.savez(
        "data/test.npz",
        specs=test_specs,
        labels=test_labels,
        audio=test_audio,
        files=test_fnames,
    )

    train_art = wandb.Artifact(
        "training-data",
        type="dataset",
        description="Training data stored as compressed npz.",
    )
    train_art.add_file("data/train.npz")
    run.log_artifact(train_art)

    test_art = wandb.Artifact(
        "test-data",
        type="dataset",
        description="Test data stored as compressed npz.",
    )
    test_art.add_file("data/train.npz")
    run.log_artifact(test_art)
