# camera.py
import sys
import cv2
import os
import time
from PyQt5.QtWidgets import QApplication,QMainWindow,QWidget,QVBoxLayout,QLabel,QSizePolicy
from PyQt5.QtGui import QImage,QPixmap
from PyQt5.QtCore import Qt,QTimer
from pygrabber.dshow_graph import FilterGraph
from gesture import GestureClassifier
from hand_detector import HandDetector
from utils import WindowHelper

class CameraTab(QWidget):
    def __init__(self):
        super().__init__()
        self.detector=HandDetector()
        self.classifier=GestureClassifier()
        self.camera=None

        # Capture settings
        self.last_capture_time = 0
        self.capture_delay = 2 # detik cooldown setelah capture
        self.capture_folder = "captures"
        if not os.path.exists(self.capture_folder):
            os.makedirs(self.capture_folder)

        # Countdown settings
        self.countdown_start = 0
        self.countdown_duration = 3 # detik nahan gesture

        # Dragging settings (2-Stage)
        self.is_focused = False
        self.is_dragging = False
        self.drag_start_hand = None
        self.drag_start_win = None
        self.focus_timer = 0
        self.focus_lifetime = 0.8 # detik toleransi saat ganti gaya

        # External Window Target
        self.target_hwnd = None
        self.target_title = ""
        self.drag_win_initial_pos = None

        # Mouse settings
        self.last_click_time = 0
        self.click_delay = 0.5 # detik cooldown antar klik
        self.mouse_pos_smooth = None
        self.mouse_is_dragging = False
        self.mouse_start_hand = None
        self.mouse_start_system = None

        self.label=QLabel("Camera Preview")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding)
        lay=QVBoxLayout(self)
        lay.setContentsMargins(0,0,0,0)
        lay.addWidget(self.label)
        self.camera_index=0
        self.load_cameras()
        self.timer=QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.camera=cv2.VideoCapture(self.camera_index)
        self.timer.start(30)

    def load_cameras(self):
        try:
            devs=FilterGraph().get_input_devices()
            for i,d in enumerate(devs):
                print(i,d)
        except Exception:
            pass
 
    def update_frame(self):
        if self.camera is None:return
        ok,frame=self.camera.read()
        if not ok:return
        frame=cv2.flip(frame,1)
        h, w, _ = frame.shape

        frame,hands=self.detector.detect(frame, draw=False)

        peace_detected = False
        palm_pos = None
        fist_pos = None
        thumb_pos = None
        ok_detected = False

        for lm in hands:
            # Ambil koordinat pergelangan tangan untuk posisi teks
            x,y=lm[0][1],lm[0][2]

            gesture, _ = self.classifier.predict(lm)

            if gesture != "NONE":
                cv2.putText(
                    frame,
                    gesture,
                    (x, max(y - 20, 30)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )

                if gesture == "PEACE":
                    peace_detected = True

                if gesture == "PALM":
                    palm_pos = (x, y)

                if gesture == "FIST":
                    fist_pos = (x, y)

                if gesture == "THUMB" or gesture == "OK":
                    # Gunakan ujung jempol (4) untuk tracking mouse
                    thumb_pos = (lm[4][1], lm[4][2])

                if gesture == "OK":
                    ok_detected = True

        # Logic Mouse Control (Thumb to Move, OK to Click) - RELATIVE MOVEMENT
        current_time = time.time()

        if thumb_pos:
            from utils import MouseHelper

            # Skala layar untuk konversi gerakan hand ke monitor
            screen = QApplication.primaryScreen().size()
            scale_x = screen.width() / w
            scale_y = screen.height() / h

            if not self.mouse_is_dragging:
                # Mulai Grabbing Mouse
                self.mouse_is_dragging = True
                self.mouse_start_hand = thumb_pos
                self.mouse_start_system = MouseHelper.get_position()
            else:
                # Hitung selisih gerakan tangan
                dx = int((thumb_pos[0] - self.mouse_start_hand[0]) * scale_x)
                dy = int((thumb_pos[1] - self.mouse_start_hand[1]) * scale_y)

                # Tentukan target posisi mouse (posisi sistem awal + delta)
                tx = self.mouse_start_system[0] + dx
                ty = self.mouse_start_system[1] + dy

                # Gunakan smoothing agar pointer tidak bergetar
                if self.mouse_pos_smooth is None:
                    self.mouse_pos_smooth = (tx, ty)
                else:
                    alpha = 0.4 # Smoothing factor
                    self.mouse_pos_smooth = (
                        int(self.mouse_pos_smooth[0] + (tx - self.mouse_pos_smooth[0]) * alpha),
                        int(self.mouse_pos_smooth[1] + (ty - self.mouse_pos_smooth[1]) * alpha)
                    )

                MouseHelper.move(self.mouse_pos_smooth[0], self.mouse_pos_smooth[1])

            cv2.putText(frame, "MOUSE CONTROL (RELATIVE)", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

            # Klik saat deteksi OK
            if ok_detected and (current_time - self.last_click_time > self.click_delay):
                MouseHelper.click()
                self.last_click_time = current_time
                print("Mouse Clicked!")
        else:
            # Lepas gesture = mouse berhenti di tempat
            self.mouse_is_dragging = False
            self.mouse_pos_smooth = None

        # Feedback visual klik
        if current_time - self.last_click_time < 0.3:
            cv2.putText(frame, "CLICK!", (w // 2 - 50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

        # Logic Window Dragging (2-Stage: Palm to Focus, Fist to Drag)
        current_time = time.time()

        # Gunakan toleransi waktu (focus_timer) agar tidak langsung reset saat ganti gaya
        is_still_focused = (current_time - self.focus_timer < self.focus_lifetime)

        # PRIORITAS 1: Jika sedang DRAGGING (Mengepal) - Gerakkan Window Target
        if self.is_dragging and fist_pos and self.target_hwnd:
            # Hitung skala antara layar dan resolusi kamera (1:1 tracking)
            screen = QApplication.primaryScreen().size()
            scale_x = screen.width() / w
            scale_y = screen.height() / h

            dx = int((fist_pos[0] - self.drag_start_hand[0]) * scale_x)
            dy = int((fist_pos[1] - self.drag_start_hand[1]) * scale_y)

            # Pindahkan Jendela Target (External)
            new_x = self.drag_win_initial_pos[0] + dx
            new_y = self.drag_win_initial_pos[1] + dy
            WindowHelper.move_window(self.target_hwnd, new_x, new_y)

            # Update timer agar lock tidak lepas
            self.focus_timer = current_time
            cv2.putText(frame, f"MOVING: {self.target_title[:20]}...", (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # PRIORITAS 2: Jika belum dragging tapi mengepal setelah fokus (GRAB Target)
        elif fist_pos and self.is_focused and is_still_focused and self.target_hwnd:
            self.is_dragging = True
            self.drag_start_hand = fist_pos
            # Simpan posisi awal jendela target
            tx, ty, tw, th = WindowHelper.get_rect(self.target_hwnd)
            self.drag_win_initial_pos = (tx, ty)
            self.focus_timer = current_time
            cv2.putText(frame, "GRABBED!", (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        # PRIORITAS 3: Mencari fokus awal (Telapak tangan terbuka) - Cari Window Teratas
        elif palm_pos:
            # Cari jendela teratas di desktop (exclude kamera sendiri)
            if not self.is_focused:
                current_win_id = int(self.window().winId())
                hwnd, title = WindowHelper.get_top_external_window(current_win_id)
                if hwnd:
                    self.target_hwnd = hwnd
                    self.target_title = title

            self.is_focused = True
            self.is_dragging = False
            self.focus_timer = current_time

            display_title = self.target_title if self.target_title else "No Target Found"
            cv2.putText(frame, f"TARGET: {display_title[:20]}", (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # RESET: Jika tangan hilang atau toleransi waktu habis
        elif current_time - self.focus_timer > self.focus_lifetime:
            self.is_focused = False
            self.is_dragging = False
            self.target_hwnd = None
            self.target_title = ""

        # Logic Countdown & Capture
        current_time = time.time()

        # Cek apakah baru aja capture (biar nggak langsung mulai countdown lagi)
        in_cooldown = (current_time - self.last_capture_time < self.capture_delay)

        if peace_detected and not in_cooldown:
            if self.countdown_start == 0:
                self.countdown_start = current_time

            elapsed = current_time - self.countdown_start
            remaining = int(self.countdown_duration - elapsed) + 1

            if elapsed >= self.countdown_duration:
                # Waktunya JEP RET!
                filename = f"IMG_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                filepath = os.path.join(self.capture_folder, filename)
                cv2.imwrite(filepath, frame)
                print(f"Captured: {filepath}")

                self.last_capture_time = current_time
                self.countdown_start = 0 # Reset countdown
            else:
                # Gambar Angka Countdown di Tengah Layar
                text = str(remaining)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 5
                thickness = 10
                text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                text_x = (w - text_size[0]) // 2
                text_y = (h + text_size[1]) // 2

                # Kasih shadow biar kebaca
                cv2.putText(frame, text, (text_x+5, text_y+5), font, font_scale, (0, 0, 0), thickness)
                cv2.putText(frame, text, (text_x, text_y), font, font_scale, (0, 255, 255), thickness)
        else:
            # Kalau gesture dilepas atau lagi cooldown, reset timer
            self.countdown_start = 0

        # Feedback visual "CAPTURED!" (setelah jepret)
        if current_time - self.last_capture_time < 0.8:
            cv2.putText(frame, "CAPTURED!", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        h,w,c=rgb.shape
        img=QImage(rgb.data,w,h,c*w,QImage.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(img).scaled(self.label.size(),Qt.KeepAspectRatio,Qt.FastTransformation))

class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Camera Gesture")
        self.setCentralWidget(CameraTab())

if __name__=="__main__":
    app=QApplication(sys.argv)
    w=MainApp();w.resize(900,650);w.show()
    sys.exit(app.exec_())
