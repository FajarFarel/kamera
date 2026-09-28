import argparse
import csv
import os
import joblib

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_AUGMENTED_FILE = os.path.join("dataset", "gestures_augmented.csv")
DEFAULT_ORIGINAL_FILE = os.path.join("dataset", "gestures.csv")
MODEL_DIR = "models"
MODEL_FILE = os.path.join(MODEL_DIR, "gesture_model.joblib")


def get_default_dataset_file():
    if os.path.exists(DEFAULT_AUGMENTED_FILE):
        return DEFAULT_AUGMENTED_FILE
    return DEFAULT_ORIGINAL_FILE


def load_dataset(dataset_file=None):
    if not dataset_file:
        dataset_file = get_default_dataset_file()

    if not os.path.exists(dataset_file):
        raise FileNotFoundError(
            f"Dataset tidak ditemukan: {dataset_file}\n"
            "Jalankan collect_dataset.py atau augment_dataset.py terlebih dahulu."
        )

    X = []
    y = []

    with open(dataset_file, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        feature_names = [f"f{i}" for i in range(42)]

        for row in reader:
            try:
                features = [float(row[name]) for name in feature_names]
                label = row["label"].strip()

                if label:
                    X.append(features)
                    y.append(label)
            except (ValueError, KeyError):
                continue

    if len(X) < 20:
        raise RuntimeError(
            f"Dataset terlalu sedikit ({len(X)} sample). "
            "Kumpulkan lebih banyak data."
        )

    return X, y, dataset_file


def main():
    parser = argparse.ArgumentParser(description="Train Gesture Model")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path file dataset CSV (default: gestures_augmented.csv jika ada, atau gestures.csv)",
    )
    args = parser.parse_args()

    X, y, dataset_used = load_dataset(args.dataset)

    # stratify menjaga proporsi setiap gesture di train/test.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # ExtraTreesClassifier memberikan batas pemisah sangat presisi & confidence tinggi.
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                ExtraTreesClassifier(
                    n_estimators=150,
                    random_state=42,
                    n_jobs=1,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    train_prediction = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, train_prediction)

    prediction = model.predict(X_test)
    accuracy = accuracy_score(y_test, prediction)

    print("\n=== TRAINING RESULT ===")
    print(f"Dataset used  : {dataset_used}")
    print(f"Total samples : {len(X)}")
    print(f"Train samples : {len(X_train)}")
    print(f"Test samples  : {len(X_test)}")
    print(f"Training accuracy: {train_accuracy * 100:.2f}%")
    print(f"Testing accuracy: {accuracy * 100:.2f}%")
    print("\nClassification report:")
    print(classification_report(y_test, prediction, zero_division=0))

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_FILE)

    print(f"\nModel saved: {MODEL_FILE}")


if __name__ == "__main__":
    main()
