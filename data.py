import os
import re
from glob import glob
import random

import numpy as np
from scipy.io import wavfile
from scipy import signal

SAMPLE_RATE = 16000
LABELS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
EXAMPLES_PER_LABEL = 1000
DATA_PATH = os.path.join("./train/audio")


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
    label_to_files = organize_files_by_label(DATA_PATH)

    labels_filenames = []
    audio = []

    for l in label_to_files:
        for f in label_to_files[l]:
            labels_filenames.append([l, f])
            audio.append(transform_wav(os.path.join(DATA_PATH, l, f + ".wav")))

    labels_filenames = np.array(labels_filenames)
    audio = np.array(audio)

    spectrograms = np.stack([log_spectrogram(clip) for clip in audio])
    spectrograms = spectrograms.reshape(tuple(list(spectrograms.shape) + [1]))

    np.save("labels_and_fnames.npy", labels_filenames)
    np.save("audio.npy", audio)
    np.save("spectrograms.npy", spectrograms)
