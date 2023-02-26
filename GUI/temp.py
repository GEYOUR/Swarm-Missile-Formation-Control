import sys
import uuid

from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton


class MainWindow(QMainWindow):
    singleton: 'MainWindow' = None

    def __init__(self):
        super().__init__()
        btn = QPushButton(f'RESTART\n{uuid.uuid4()}')
        btn.clicked.connect(MainWindow.restart)
        self.setCentralWidget(btn)
        self.show()

    @staticmethod
    def restart():
        MainWindow.singleton = MainWindow()


def main():
    app = QApplication([])
    MainWindow.restart()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
