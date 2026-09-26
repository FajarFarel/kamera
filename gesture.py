import math
import os

import joblib


class GestureClassifier:
    """
    ML gesture classifier.

    Input:
        21 hand landmarks dalam format project sekarang:
        [(id, x, y), ...]

    Model:
        models/gesture_model.joblib

    Jika model belum ada, classifier otomatis fallback ke rule-based
    classifier lama supaya project tetap bisa jalan.
    """

    MODEL_FILE = os.path.join("models", "gesture_model.joblib")

    def __init__(self, model_path=None, confidence_threshold=0.50):
        self.model_path = model_path or self.MODEL_FILE
        self.confidence_threshold = confidence_threshold
        self.model = None

        self.load_model()

    def load_model(self):
        if not os.path.exists(self.model_path):
            print(
                f"[GestureClassifier] ML model belum ditemukan: "
                f"{self.model_path}"
            )
            print("[GestureClassifier] Menggunakan rule-based fallback.")
            return

        try:
            self.model = joblib.load(self.model_path)
            print(f"[GestureClassifier] ML model loaded: {self.model_path}")
        except Exception as e:
            print(f"[GestureClassifier] Gagal load model: {e}")
            self.model = None

    @staticmethod
    def normalize_landmarks(pts):
        """
        Harus sama dengan preprocessing pada collect_dataset.py.
        """
        if len(pts) != 21:
            return None

        wrist_x, wrist_y = pts[0][1], pts[0][2]

        features = []

        for _, x, y in pts:
            features.extend(
                [
                    x - wrist_x,
                    y - wrist_y,
                ]
            )

        ref_x = pts[9][1] - wrist_x
        ref_y = pts[9][2] - wrist_y

        scale = math.sqrt(
            ref_x * ref_x +
            ref_y * ref_y
        )

        if scale < 1e-6:
            return None

        return [value / scale for value in features]

    def predict(self, pts):
        """
        Return:
            (gesture, confidence)

        Contoh:
            ("PALM", 0.97)
        """
        if self.model is None:
            return self._rule_based(pts), 1.0

        features = self.normalize_landmarks(pts)

        if features is None:
            return "NONE", 0.0

        try:
            prediction = self.model.predict([features])[0]

            confidence = 1.0

            if hasattr(self.model, "predict_proba"):
                probabilities = self.model.predict_proba([features])[0]
                confidence = float(max(probabilities))

            if confidence < self.confidence_threshold:
                return "NONE", confidence

            return str(prediction), confidence

        except Exception as e:
            print(f"[GestureClassifier] Prediction error: {e}")
            return "NONE", 0.0

    def classify(self, finger_state, pts):
        """
        Compatibility dengan API lama.
        Sekarang classify() menggunakan ML jika model tersedia.
        """
        gesture, _ = self.predict(pts)
        return gesture

    def finger_heart(self, pts):
        if len(pts) != 21:
            return False

        thumb = pts[4]
        index = pts[8]

        wrist = pts[0]
        middle = pts[9]

        hand_size = math.dist(
            (wrist[1], wrist[2]),
            (middle[1], middle[2])
        )

        distance = math.dist(
            (thumb[1], thumb[2]),
            (index[1], index[2])
        )

        if hand_size == 0:
            return False

        return distance < hand_size * 0.45

    def _rule_based(self, pts):
        """
        Fallback classifier dari versi project lama.
        """
        state = self._finger_state(pts)

        if state is None:
            return "NONE"

        thumb, index, middle, ring, pinky = state

        # 👌
        if self._is_ok(pts):
            return "OK"

        # 👍
        if thumb and not index and not middle and not ring and not pinky:
            return "THUMB"

        # ☝
        if not thumb and index and not middle and not ring and not pinky:
            return "INDEX"

        # ✌
        if not thumb and index and middle and not ring and not pinky:
            return "PEACE"

        # ✋
        if thumb and index and middle and ring and pinky:
            return "PALM"

        # ✊
        if not thumb and not index and not middle and not ring and not pinky:
            return "FIST"

        # ❤️
        if self.finger_heart(pts):
            return "LOVE"

        return "NONE"

    @staticmethod
    def _finger_state(pts):
        if len(pts) != 21:
            return None

        p = [(pt[1], pt[2]) for pt in pts]

        index = math.dist(p[0], p[8]) > math.dist(p[0], p[6])
        middle = math.dist(p[0], p[12]) > math.dist(p[0], p[10])
        ring = math.dist(p[0], p[16]) > math.dist(p[0], p[14])
        pinky = math.dist(p[0], p[20]) > math.dist(p[0], p[18])
        thumb = math.dist(p[17], p[4]) > math.dist(p[17], p[3])

        return [thumb, index, middle, ring, pinky]

    def _is_ok(self, pts):
        if len(pts) != 21:
            return False

        p = [(pt[1], pt[2]) for pt in pts]

        d = math.dist(p[4], p[8])
        ref = math.dist(p[0], p[9])

        state = self._finger_state(pts)
        if state is None:
            return False

        middle, ring, pinky = state[2], state[3], state[4]

        return (
            d < ref * 0.3
            and middle
            and ring
            and pinky
        )

    # API compatibility dengan hand_detector.py lama.
    def is_thumb_up(self, pts):
        return self.predict(pts)[0] == "THUMB"

    def is_palm(self, pts):
        return self.predict(pts)[0] == "PALM"

    def index_up(self, pts):
        return self.predict(pts)[0] == "INDEX"

    def is_love(self, pts):
        return self.predict(pts)[0] == "LOVE"

    def is_fist(self, pts):
        return self.predict(pts)[0] == "FIST"

    def is_ok(self, pts):
        return self.predict(pts)[0] == "OK"
