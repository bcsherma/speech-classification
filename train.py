import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential, layers, losses

N_CLASSES = 10
SAMPLE_RATE = 16000
SPECTROGRAM_SHAPE = (99, 161, 1)


def load_data():
    labels_and_fnames = np.load("labels_and_fnames.npy")
    audio = np.load("audio.npy")
    spectrograms = np.load("spectrograms.npy")
    labels, fnames = np.split(labels_and_fnames, 2, axis=1)
    return labels, fnames, audio, spectrograms


def build_model():
    model = Sequential(
        [
            layers.Input(shape=SPECTROGRAM_SHAPE),
            layers.BatchNormalization(),
            layers.Conv2D(8, 2, activation="relu"),
            layers.Conv2D(8, 2, activation="relu"),
            layers.MaxPool2D(pool_size=(2, 2)),
            layers.Dropout(rate=0.2),
            layers.Conv2D(16, 3, activation="relu"),
            layers.Conv2D(16, 3, activation="relu"),
            layers.MaxPool2D(pool_size=(2, 2)),
            layers.Dropout(rate=0.2),
            layers.Conv2D(32, 3, activation="relu"),
            layers.MaxPool2D(pool_size=(2, 2)),
            layers.Dropout(rate=0.2),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.BatchNormalization(),
            layers.Dense(128, activation="relu"),
            layers.BatchNormalization(),
            layers.Dense(10, activation="softmax"),
        ]
    )
    model.summary()
    model.compile(loss=losses.binary_crossentropy, metrics="accuracy")
    return model


if __name__ == "__main__":
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
    model = build_model()
    model.fit(
        train_specs,
        train_labels,
        validation_data=(test_specs, test_labels),
        batch_size=32,
        epochs=10,
    )
