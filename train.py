import csv
import os
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATASET_FILE = os.path.join("dataset", "gestures.csv")
MODEL_DIR = "models"
MODEL_FILE = os.path.join(MODEL_DIR, "gesture_model.joblib")


def load_dataset():
    if not os.path.exists(DATASET_FILE):
        raise FileNotFoundError(
            f"Dataset tidak ditemukan: {DATASET_FILE}\n"
            "Jalankan collect_dataset.py terlebih dahulu."
        )

    X = []
    y = []

    with open(DATASET_FILE, "r", newline="", encoding="utf-8") as f:
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

    return X, y


def main():
    X, y = load_dataset()

    # stratify menjaga proporsi setiap gesture di train/test.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Random Forest cukup ringan untuk 42 feature landmark.
    # StandardScaler tidak wajib untuk RF, tetapi pipeline ini memudahkan
    # jika classifier diganti di kemudian hari.
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=42,
                    n_jobs=-1,
                    class_weight="balanced",
                    min_samples_leaf=2,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)
    accuracy = accuracy_score(y_test, prediction)

    print("\n=== TRAINING RESULT ===")
    print(f"Total samples : {len(X)}")
    print(f"Train samples : {len(X_train)}")
    print(f"Test samples  : {len(X_test)}")
    print(f"Accuracy      : {accuracy * 100:.2f}%")
    print("\nClassification report:")
    print(classification_report(y_test, prediction, zero_division=0))

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_FILE)

    print(f"\nModel saved: {MODEL_FILE}")


if __name__ == "__main__":
    main()
