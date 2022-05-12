from ctypes import alignment
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5 import QtGui
import sys
import datetime

class Stream(QObject):
    '''For outputting terminal in GUI'''
    newText = pyqtSignal(str)

    def write(self, text):
        self.newText.emit(str(text))

class App(QDialog):
    def __init__(self):
        super().__init__()
        self.title = 'mmWave FSL'
        self.left = 50
        self.top = 50
        self.width = 1024
        self.height = 768
        self.initUI()

        sys.stdout = Stream(newText=self.onUpdateText)

    # ----- FUNCTIONS FOR TERMINAL ON GUI -----

    def onUpdateText(self, text):
        cursor = self.process.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(text)
        self.process.setTextCursor(cursor)
        self.process.ensureCursorVisible()

    def __del__(self):
        sys.stdout = sys.__stdout__

    def closeEvent(self, event):
        """Shuts down application on close."""
        # Return standard output to defaults.
        sys.stdout = sys.__stdout__
        super().closeEvent(event)

    # ------ END OF FUNCTIONS FOR TERMINAL ON GUI -----

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        # self.center()

        # Layout:
        # 3 sections: Terminal, recording buttons, plot
        horizontalLayout = QHBoxLayout()

        # QVBoxLayout for Terminal and recording buttons
        leftLayout = QVBoxLayout()
        
        # Terminal
        self.terminalGroupBox = QGroupBox('Terminal')
        self.createTerminalGroupBox()

        # Record
        self.recordGroupBox = QGroupBox('Buttons')
        self.createRecordGroupBox()

        leftLayout.addWidget(self.terminalGroupBox)
        leftLayout.addWidget(self.recordGroupBox)

        # QVBoxLayout for plot
        rightLayout = QVBoxLayout()
        self.plotGroupBox = QGroupBox('Plot')


        rightLayout.addWidget(self.plotGroupBox)



        # Putting it together
        horizontalLayout.addLayout(leftLayout)
        horizontalLayout.addLayout(rightLayout)
        self.setLayout(horizontalLayout)

        self.show()

    def createTerminalGroupBox(self):
        terminalVbox = QVBoxLayout()

        self.process = QTextEdit()
        self.process.moveCursor(QtGui.QTextCursor.Start)
        self.process.ensureCursorVisible()
        # self.process.setLineWrapColumnOrWidth(500)
        # self.process.setLineWrapMode(QTextEdit.FixedPixelWidth)

        terminalVbox.addWidget(self.process)
        self.terminalGroupBox.setLayout(terminalVbox)

    def createRecordGroupBox(self):
        recordHbox = QHBoxLayout()

        self.rec_button = QPushButton('Rec/Stop')
        self.rec_button.clicked.connect(self.clicked)
        
        self.filename_label = QLabel(self, alignment=Qt.AlignLeft)
        self.filename_label.setText('Filename will be shown here upon recording')
        
        recordHbox.addWidget(self.rec_button)
        recordHbox.addWidget(QLabel(text='Filename: ', alignment=Qt.AlignRight))
        recordHbox.addWidget(self.filename_label)
        self.recordGroupBox.setLayout(recordHbox)


    def clicked(self):
        print('Filename updated')
        self.filename_label.setText("raw_" + str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")) +'.pkl')

    def center(self):
        """centers the window on the screen"""
        screen = QDesktopWidget().screenGeometry()
        size = self.geometry()
        self.move((screen.width() - size.width()) // 2,
                  (screen.height() - size.height()) // 2)

    

# if __name__ == '__main__':
#     app = QApplication(sys.argv)
#     ex = App()

#     # behaviour to trigger on exit
#     sys.exit(app.exec_())