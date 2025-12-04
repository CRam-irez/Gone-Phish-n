import sys
import time
from PyQt6 import QtWidgets, uic
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QShortcut, QKeySequence

from src.model_url import URLPhishingDetector
from src.model_email import MultinomialNB


# Load models
url_model = URLPhishingDetector.load("phishing_model.pkl")
email_model = MultinomialNB.load("email_spam_model.pkl")


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("ui/ui_mainwindow.ui", self)

        # Button connections
        self.pushButton_check_url.clicked.connect(self.check_url)
        self.pushButton_check_email.clicked.connect(self.check_email)

        # Enter key for URL box
        self.lineEdit_url.returnPressed.connect(self.pushButton_check_url.click)

        # Ctrl+Enter/Shift+Enter for Email box
        QShortcut(QKeySequence("Ctrl+Return"), self.lineEdit_email).activated.connect(self.pushButton_check_email.click)
        QShortcut(QKeySequence("Shift+Return"), self.lineEdit_email).activated.connect(self.pushButton_check_email.click)

        # Styling/startup
        self.progressBar.setVisible(False)
        self.lineEdit_email.setPlaceholderText("Paste full email body here...")

        self.show()

    def check_url(self):
        text = self.lineEdit_url.text().strip()
        if not text:
            self.show_result("Please enter a URL!", "#ff6b6b")
            return
        if not text.startswith(("http://", "https://")):
            text = "https://" + text
        self.analyze(text, url_model, "URL")

    def check_email(self):
        text = self.lineEdit_email.toPlainText().strip()
        if not text:
            self.show_result("Please enter email text!", "#ff6b6b")
            return
        self.analyze(text, email_model, "Email")

    def analyze(self, text, model, type_):
        self.textEdit.setPlainText(f"Analyzing {type_}...\n\n{text}")
        self.progressBar.setVisible(True)
        self.progressBar.setValue(30)
        QtWidgets.QApplication.processEvents()
        time.sleep(0.3)
        self.progressBar.setValue(70)
        QtWidgets.QApplication.processEvents()

        prob = model.predict_proba(text) * 100
        self.progressBar.setValue(100)

        if prob > 70:
            verdict, color = "MALICIOUS DETECTED!", "#ff4444"
        elif prob > 40:
            verdict, color = "SUSPICIOUS", "#ffa500"
        else:
            verdict, color = "SAFE", "#44ff44"

        result = f"{verdict}\n\nRisk Level: {prob:.1f}%\n\nChecked:\n{text}"
        self.show_result(result, color)
        self.progressBar.setVisible(False)

    def show_result(self, text, color):
        self.textEdit.setPlainText(text)
        self.textEdit.setStyleSheet(f"""
            QTextEdit {{
                background-color: #0d1117;
                color: {color};
                font-size: 18px;
                font-weight: bold;
                border: 3px solid {color};
                border-radius: 15px;
                padding: 20px;
            }}
        """)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec())