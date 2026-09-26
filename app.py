import sys

from PyQt5.QtWidgets import QApplication

from camera import MainApp


def main():
    app = QApplication(sys.argv)

    window = MainApp()
    window.resize(900, 650)
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()