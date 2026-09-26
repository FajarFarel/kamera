import cv2
import os


class StickerOverlay:

    def __init__(self):

        self.stickers = {}

        self.load_assets()

    # =============================
    # LOAD PNG
    # =============================

    def load_assets(self):

        folder = "assets"

        files = {

            "THUMB": "thumb.png",
            "PALM": "palm.png",
            "FIST": "fist.png",
            "PEACE": "peace.png",
            "LOVE": "love.png",
            "INDEX": "index.png",
            "OK": "ok.png"

        }

        for name, file in files.items():

            path = os.path.join(folder, file)

            if os.path.exists(path):

                img = cv2.imread(
                    path,
                    cv2.IMREAD_UNCHANGED
                )

                self.stickers[name] = img

    # =============================
    # DRAW STICKER
    # =============================

    def draw(

        self,

        frame,

        gesture,

        x,

        y,

        size=120

    ):

        if gesture not in self.stickers:
            return

        sticker = self.stickers[gesture]

        if sticker is None:
            return

        sticker = cv2.resize(

            sticker,

            (size, size)

        )

        h, w = sticker.shape[:2]

        x = int(x - w // 2)

        y = int(y - h - 20)

        if x < 0:
            x = 0

        if y < 0:
            y = 0

        if x + w > frame.shape[1]:
            return

        if y + h > frame.shape[0]:
            return

        # Cek apakah gambar punya channel alpha (transparansi)
        if sticker.shape[2] == 4:
            alpha = sticker[:, :, 3] / 255.0
            for c in range(3):
                frame[y:y+h, x:x+w, c] = (
                    alpha * sticker[:, :, c]
                    +
                    (1 - alpha)
                    * frame[y:y+h, x:x+w, c]
                )
        else:
            # Kalau nggak ada alpha, langsung tempel aja RGB-nya
            frame[y:y+h, x:x+w] = sticker[:, :, :3]

    # =============================
    # AVAILABLE?
    # =============================

    def has(self, gesture):

        return gesture in self.stickers