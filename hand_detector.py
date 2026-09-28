import cv2
import mediapipe as mp
import math


class HandDetector:

    def __init__(
        self,
        max_hands=2,
        detect_conf=0.7,
        track_conf=0.7
    ):

        self.mp_hands = mp.solutions.hands

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detect_conf,
            min_tracking_confidence=track_conf
        )

        self.drawer = mp.solutions.drawing_utils

        self.connections = self.mp_hands.HAND_CONNECTIONS

    # ============================================
    # DETECT HAND
    # ============================================

    def detect(self, frame, draw=True):

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        result = self.hands.process(rgb)

        hands = []

        if result.multi_hand_landmarks:

            h, w, _ = frame.shape

            for hand in result.multi_hand_landmarks:

                pts = []

                for i, lm in enumerate(hand.landmark):

                    x = int(lm.x * w)
                    y = int(lm.y * h)

                    pts.append(
                        (i, x, y)
                    )

                hands.append(pts)

                if draw:
                    self.draw_hand(frame, pts)

        return frame, hands

    # ============================================
    # DRAW HAND (WARNA SENDIRI)
    # ============================================

    def draw_hand(
        self,
        frame,
        pts
    ):

        finger_color = [
            (0,0,255),      # Thumb
            (255,0,0),      # Index
            (0,255,0),      # Middle
            (0,255,255),    # Ring
            (255,0,255)     # Pinky
        ]

        fingers = [

            [0,1,2,3,4],

            [0,5,6,7,8],

            [0,9,10,11,12],

            [0,13,14,15,16],

            [0,17,18,19,20]

        ]

        for color,finger in zip(
            finger_color,
            fingers
        ):

            for i in range(len(finger)-1):

                p1 = pts[finger[i]]

                p2 = pts[finger[i+1]]

                cv2.line(

                    frame,

                    (p1[1],p1[2]),

                    (p2[1],p2[2]),

                    color,

                    3

                )

        for p in pts:

            cv2.circle(

                frame,

                (p[1],p[2]),

                6,

                (255,255,255),

                -1

            )

    # ============================================
    # FINGER STATE
    # ============================================

    def finger_state(
        self,
        pts
    ):

        if len(pts)!=21:
            return None

        # Ambil koordinat (x, y) saja
        p = [(pt[1], pt[2]) for pt in pts]

        # Logic: Jarak titik ujung (tip) ke pergelangan (wrist)
        # harus lebih jauh dibanding jarak sendi tengah (PIP) ke pergelangan
        # Wrist = landmark 0

        index = math.dist(p[0], p[8]) > math.dist(p[0], p[6])
        middle = math.dist(p[0], p[12]) > math.dist(p[0], p[10])
        ring = math.dist(p[0], p[16]) > math.dist(p[0], p[14])
        pinky = math.dist(p[0], p[20]) > math.dist(p[0], p[18])

        # Untuk Jempol: Jarak ujung jempol (4) ke pangkal kelingking (17)
        # vs jarak sendi jempol (3) ke pangkal kelingking (17)
        thumb = math.dist(p[17], p[4]) > math.dist(p[17], p[3])

        return [
            thumb,
            index,
            middle,
            ring,
            pinky
        ]

    # ============================================
    # LOVE
    # ============================================

    def finger_heart(
        self,
        pts
    ):

        thumb = pts[4]

        index = pts[8]

        wrist = pts[0]

        middle = pts[9]

        size = math.dist(

            (wrist[1],wrist[2]),

            (middle[1],middle[2])

        )

        d = math.dist(

            (thumb[1],thumb[2]),

            (index[1],index[2])

        )

        return d < size*0.45

    # ============================================
    # GESTURE CHECKERS
    # ============================================

    def is_thumb_up(self, pts):
        return self.get_gesture(pts) == "THUMB"

    def is_palm(self, pts):
        return self.get_gesture(pts) == "PALM"

    def index_up(self, pts):
        return self.get_gesture(pts) == "INDEX"

    def is_love(self, pts):
        return self.get_gesture(pts) == "LOVE"

    def is_fist(self, pts):
        return self.get_gesture(pts) == "FIST"

    def is_ok(self, pts):
        # Akurasi tinggi: Cek jarak jempol (4) dan telunjuk (8)
        # Serta pastikan jari lain (12, 16, 20) tegak
        p = [(pt[1], pt[2]) for pt in pts]

        # Jarak ujung jempol ke ujung telunjuk
        d = math.dist(p[4], p[8])

        # Ukuran tangan sebagai referensi (wrist ke middle base)
        ref = math.dist(p[0], p[9])

        # Jari lain harus tegak
        state = self.finger_state(pts)
        middle, ring, pinky = state[2], state[3], state[4]

        return (d < ref * 0.3) and middle and ring and pinky

    # ============================================
    # GESTURE CLASSIFIER
    # ============================================

    def get_gesture(
        self,
        pts
    ):

        state = self.finger_state(
            pts
        )

        if state is None:
            return "NONE"

        thumb,index,middle,ring,pinky = state

        # 👌 (Priority Check)
        if self.is_ok(pts):
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

        # 👌
        if self.is_ok(pts):
            return "OK"

        return "NONE"
