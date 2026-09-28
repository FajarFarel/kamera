import csv
import math
import os
import random
from collections import Counter

INPUT_DATASET = os.path.join("dataset", "gestures.csv")
OUTPUT_DATASET = os.path.join("dataset", "gestures_augmented.csv")
TARGET_PER_POSE = 300


def augment_features(features):
    """
    Melakukan transformasi geometri sederhana dan noise pada 42 landmark (x,y).
    """
    # 1. Rotasi acak antara -15 sampai +15 derajat
    angle = random.uniform(-math.radians(15), math.radians(15))
    cos_a, sin_a = math.cos(angle), math.sin(angle)

    # 2. Skala acak (0.95 - 1.05)
    scale = random.uniform(0.95, 1.05)

    augmented = []
    for i in range(0, 42, 2):
        x, y = features[i], features[i + 1]

        # Rotasi
        xr = x * cos_a - y * sin_a
        yr = x * sin_a + y * cos_a

        # Skala
        xr *= scale
        yr *= scale

        # Noise getaran halus
        xr += random.gauss(0, 0.015)
        yr += random.gauss(0, 0.015)

        augmented.extend([xr, yr])

    return augmented


def mirror_features(features):
    """
    Membalik koordinat X (mirroring kiri/kanan).
    """
    mirrored = []
    for i in range(0, 42, 2):
        mirrored.extend([-features[i], features[i + 1]])
    return mirrored


def mixup(f1, f2, alpha=0.5):
    """
    Interpolasi linier antara dua sampel pose yang sama.
    """
    return [alpha * a + (1 - alpha) * b for a, b in zip(f1, f2)]


def generate_augmented_dataset(
    input_path=INPUT_DATASET,
    output_path=OUTPUT_DATASET,
    target_count=TARGET_PER_POSE,
    seed=42,
):
    random.seed(seed)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File dataset tidak ditemukan: {input_path}")

    data_by_label = {}
    feature_names = [f"f{i}" for i in range(42)]

    with open(input_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                label = row["label"].strip()
                features = [float(row[f"f{i}"]) for i in range(42)]
                if label:
                    data_by_label.setdefault(label, []).append(features)
            except (ValueError, KeyError):
                continue

    print("=== PROSES AUGMENTASI DATASET ===")
    print("Sampel asli per pose:")
    for label, samples in data_by_label.items():
        print(f"  - {label:<8}: {len(samples)} data")

    augmented_rows = []

    for label, samples in data_by_label.items():
        original_samples = [list(s) for s in samples]

        # Buat pool sampel (sampel asli + sampel mirror)
        mirrored_samples = [mirror_features(s) for s in original_samples]
        pool = original_samples + mirrored_samples

        current_list = list(original_samples)

        while len(current_list) < target_count:
            # Campurkan mixup dan augmentasi transformasi
            if random.random() < 0.4 and len(pool) >= 2:
                s1, s2 = random.sample(pool, 2)
                alpha = random.uniform(0.3, 0.7)
                new_sample = mixup(s1, s2, alpha)
            else:
                base_sample = random.choice(pool)
                new_sample = augment_features(base_sample)

            current_list.append(new_sample)

        # Ambil tepat target_count sampel
        for feat in current_list[:target_count]:
            augmented_rows.append((feat, label))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(feature_names + ["label"])
        for feat, label in augmented_rows:
            writer.writerow(feat + [label])

    counts = Counter([r[1] for r in augmented_rows])

    print(f"\nDataset baru berhasil dibuat: {output_path}")
    print(f"Total baris data : {len(augmented_rows)}")
    print("Distribusi data per pose:")
    for label, count in counts.items():
        print(f"  - {label:<8}: {count} data")

    return output_path


if __name__ == "__main__":
    generate_augmented_dataset()
