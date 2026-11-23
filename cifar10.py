"""Small-scale, reproducible CIFAR-10 classifier experiments."""
from __future__ import annotations

import argparse
import random

import numpy as np


def sample_subset(features, labels, count: int, seed: int = 42):
    """Deterministically sample at most count items from matching arrays."""
    if count < 1:
        raise ValueError("count must be positive")
    if len(features) != len(labels):
        raise ValueError("features and labels must have equal lengths")
    generator = np.random.default_rng(seed)
    indices = generator.choice(len(features), size=min(count, len(features)), replace=False)
    return features[indices], np.asarray(labels).reshape(-1)[indices]


def run_experiment(model_name: str, train_size: int, test_size: int,
                   epochs: int, seed: int) -> dict[str, float]:
    import tensorflow as tf
    from sklearn.decomposition import PCA
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.neural_network import BernoulliRBM
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train, y_train = sample_subset(x_train, y_train, train_size, seed)
    x_test, y_test = sample_subset(x_test, y_test, test_size, seed + 1)
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    train_flat = x_train.reshape(len(x_train), -1)
    test_flat = x_test.reshape(len(x_test), -1)
    results: dict[str, float] = {}
    chosen = ["cnn", "svm", "forest", "rbm"] if model_name == "all" else [model_name]

    for name in chosen:
        if name == "cnn":
            network = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(32, 32, 3)),
                tf.keras.layers.Conv2D(32, 3, activation="relu"),
                tf.keras.layers.MaxPooling2D(2),
                tf.keras.layers.Conv2D(64, 3, activation="relu"),
                tf.keras.layers.MaxPooling2D(2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(10, activation="softmax"),
            ])
            network.compile(
                optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
            )
            network.fit(
                x_train, y_train, epochs=epochs, batch_size=64,
                validation_split=0.1, verbose=2
            )
            predictions = np.argmax(network.predict(x_test, verbose=0), axis=1)
        elif name == "svm":
            classifier = make_pipeline(
                StandardScaler(), PCA(n_components=min(100, len(train_flat) - 1)),
                SVC(kernel="rbf")
            )
            classifier.fit(train_flat, y_train)
            predictions = classifier.predict(test_flat)
        elif name == "forest":
            classifier = RandomForestClassifier(
                n_estimators=100, n_jobs=-1, random_state=seed
            )
            classifier.fit(train_flat, y_train)
            predictions = classifier.predict(test_flat)
        elif name == "rbm":
            # One RBM + logistic regression. This is NOT a deep belief network.
            classifier = make_pipeline(
                BernoulliRBM(n_components=64, n_iter=3, random_state=seed, verbose=False),
                LogisticRegression(max_iter=500)
            )
            classifier.fit(train_flat, y_train)
            predictions = classifier.predict(test_flat)
        else:
            raise ValueError(f"Unknown model: {name}")
        accuracy = float(accuracy_score(y_test, predictions))
        results[name] = accuracy
        print(f"{name:<8} test accuracy: {accuracy:.4f}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["cnn", "svm", "forest", "rbm", "all"], default="cnn")
    parser.add_argument("--train-size", type=int, default=4000)
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.train_size, args.test_size, args.epochs) < 1:
        parser.error("train-size, test-size and epochs must be positive")
    run_experiment(args.model, args.train_size, args.test_size, args.epochs, args.seed)


if __name__ == "__main__":
    main()
