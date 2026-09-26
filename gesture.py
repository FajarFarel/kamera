import math


class GestureClassifier:

    def __init__(self):
        pass

    # ======================================
    # Finger Heart
    # ======================================

    def finger_heart(self, pts):

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

    # ======================================
    # Main Classifier
    # ======================================

    def classify(self, finger_state, pts):

        thumb, index, middle, ring, pinky = finger_state

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