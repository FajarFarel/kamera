import csv
import os
import time
import cv2

from hand_detector import HandDetector


DATASET_FILE = os.path.join("dataset", "gestures.csv")

# Ubah / tambah gesture sesuai kebutuhan project.
GESTURES = {
    ord("1"): "PALM",
    ord("2"): "LOVE",
    ord("3"): "PEACE",
    ord("4"): "FIST",
    ord("5"): "THUMB",
    ord("6"): "INDEX",
    ord("7"): "OK",
    ord("0"): "NONE",
}


def normalize_landmarks(pts):
    """
    Mengubah 21 landmark (x,y) menjadi 42 feature yang relatif terhadap wrist.
    Hasil tidak bergantung pada posisi/ukuran tangan di frame.
    """
    if len(pts) != 21:
        return None

    wrist_x, wrist_y = pts[0][1], pts[0][2]

    coords = []
    for _, x, y in pts:
        coords.extend([x - wrist_x, y - wrist_y])

    # Gunakan jarak wrist -> middle MCP sebagai skala.
    # Hindari ketergantungan terhadap ukuran tangan di kamera.
    ref_x, ref_y = pts[9][1] - wrist_x, pts[9][2] - wrist_y
    scale = (ref_x * ref_x + ref_y * ref_y) ** 0.5

    if scale < 1e-6:
        return None

    return [v / scale for v in coords]


def mirror_features(features):
    """
    Augmentasi kiri/kanan sederhana.
    Karena koordinat sudah wrist-relative, mirror cukup membalik X.
    """
    mirrored = []
    for i in range(0, len(features), 2):
        mirrored.extend([-features[i], features[i + 1]])
    return mirrored


def ensure_dataset():
    os.makedirs(os.path.dirname(DATASET_FILE), exist_ok=True)

    if not os.path.exists(DATASET_FILE):
        with open(DATASET_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [f"f{i}" for i in range(42)] + ["label"]
            )


def append_sample(features, label):
    with open(DATASET_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(features + [label])


def main():
    ensure_dataset()

    detector = HandDetector(
        max_hands=1,
        detect_conf=0.7,
        track_conf=0.7,
    )

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Kamera tidak bisa dibuka.")

    selected_label = None
    sample_count = 0
    last_save = 0.0

    print("\n=== GESTURE DATASET COLLECTOR ===")
    print("1 PALM | 2 LOVE | 3 PEACE | 4 FIST")
    print("5 THUMB | 6 INDEX | 7 OK | 0 NONE")
    print("SPACE = simpan sample")
    print("M = simpan sample + mirror augmentation")
    print("ESC = keluar\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        frame = cv2.flip(frame, 1)

        # detector.detect juga menggambar landmark.
        frame, hands = detector.detect(frame)

        if selected_label:
            status = f"LABEL: {selected_label}"
        else:
            status = "Pilih label: 1-7 / 0"

        cv2.putText(
            frame,
            status,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            f"SAMPLES THIS RUN: {sample_count}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            "SPACE=save  M=save+mirror  ESC=quit",
            (20, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        cv2.imshow("Gesture Dataset Collector", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break

        if key in GESTURES:
            selected_label = GESTURES[key]
            print(f"Selected: {selected_label}")
            continue

        if key in (ord(" "), ord("m"), ord("M")):
            if not selected_label:
                print("Pilih label dulu.")
                continue

            if not hands:
                print("Tidak ada tangan terdeteksi.")
                continue

            # Hindari menyimpan berkali-kali karena key repeat.
            now = time.time()
            if now - last_save < 0.15:
                continue

            features = normalize_landmarks(hands[0])

            if features is None:
                print("Landmark invalid.")
                continue

            append_sample(features, selected_label)
            sample_count += 1
            last_save = now

            if key in (ord("m"), ord("M")):
                append_sample(mirror_features(features), selected_label)
                sample_count += 1
                print(f"Saved: {selected_label} + mirror")

            else:
                print(f"Saved: {selected_label}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
