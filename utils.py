import ctypes
import ctypes.wintypes

user32 = ctypes.windll.user32

GW_HWNDNEXT = 2

class WindowHelper:
    @staticmethod
    def get_window_title(hwnd):
        length = user32.GetWindowTextLengthW(hwnd)
        buff = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buff, length + 1)
        return buff.value

    @staticmethod
    def is_visible(hwnd):
        return user32.IsWindowVisible(hwnd)

    @staticmethod
    def get_rect(hwnd):
        rect = ctypes.wintypes.RECT()
        user32.GetWindowRect(hwnd, ctypes.byref(rect))
        return rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top

    @staticmethod
    def move_window(hwnd, x, y):
        # SWP_NOSIZE = 0x0001 | SWP_NOZORDER = 0x0004 | SWP_NOACTIVATE = 0x0010
        user32.SetWindowPos(hwnd, 0, int(x), int(y), 0, 0, 0x0001 | 0x0004 | 0x0010)

    @staticmethod
    def get_top_external_window(exclude_hwnd):
        # Mulai dari jendela paling atas di sistem
        hwnd = user32.GetTopWindow(None)
        while hwnd:
            if hwnd != exclude_hwnd and user32.IsWindowVisible(hwnd):
                title = WindowHelper.get_window_title(hwnd)
                # Filter jendela sistem yang tidak perlu digerakkan
                if title and title not in ["Program Manager", "Start", "Taskbar", "Settings"]:
                    # Pastikan jendela punya ukuran (bukan window tersembunyi/service)
                    left, top, w, h = WindowHelper.get_rect(hwnd)
                    if w > 100 and h > 100:
                        return hwnd, title
            hwnd = user32.GetWindow(hwnd, GW_HWNDNEXT)
        return None, ""

class MouseHelper:
    @staticmethod
    def get_position():
        point = ctypes.wintypes.POINT()
        user32.GetCursorPos(ctypes.byref(point))
        return point.x, point.y

    @staticmethod
    def move(x, y):
        user32.SetCursorPos(int(x), int(y))

    @staticmethod
    def click():
        # MOUSEEVENTF_LEFTDOWN = 0x0002, MOUSEEVENTF_LEFTUP = 0x0004
        user32.mouse_event(0x0002, 0, 0, 0, 0)
        user32.mouse_event(0x0004, 0, 0, 0, 0)
