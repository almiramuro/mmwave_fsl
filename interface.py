from asyncio.format_helpers import _format_args_and_kwargs
from ctypes import alignment
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5 import QtGui
import sys
import datetime
import threading
import serial.tools.list_ports

import IWR1443_reader

class Stream(QObject):
    '''For outputting terminal in GUI'''
    newText = pyqtSignal(str)

    def write(self, text):
        self.newText.emit(str(text))

class App(QDialog):
    def __init__(self):
        super().__init__()

        self.sensorIsRunning = False

        self.title = 'mmWave FSL'
        self.left = 50
        self.top = 50
        self.width = 1024
        self.height = 768
        self.initUI()

        sys.stdout = Stream(newText=self.onUpdateText)

    # ----- THREADING -----

    def thread(self):
        # self.configFileName = '1443config.cfg' 
        # self.CLIportPath = '/dev/tty.usbmodemR10310411'
        # self.DATAportPath = '/dev/tty.usbmodemR10310414'

        self.reader = IWR1443_reader.IWR1443_Reader(self.configFileName, self.CLIportPath, self.DATAportPath)

        self.t = threading.Thread(target=self.loop)
        self.t.start()

    def loop(self):
        self.reader.loop()

    # ----- END OF THREADING -----

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

        # Run closing functions here
        try:
            self.reader.stop()
            self.t.join()
        except AttributeError:
            pass

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

        #Configuration
        self.settingsGroupBox = QGroupBox('Configuration')
        self.createSettingsGroupBox()
        
        
        # Terminal
        self.terminalGroupBox = QGroupBox('Terminal')
        self.createTerminalGroupBox()

        # Record
        self.recordGroupBox = QGroupBox('Logging')
        self.createRecordGroupBox()

        leftLayout.addWidget(self.settingsGroupBox)
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

    # ----- CONFIGURATION PART -----

    def createSettingsGroupBox(self):
        settingsVBox = QVBoxLayout()

        # Ports Row
        serialPortsHBox = QHBoxLayout()
        self.CLIportComboBox = QComboBox()
        self.DATAportComboBox = QComboBox()
        
        # Get available ports
        ports = [port for port, _, _ in serial.tools.list_ports.comports()]
        self.CLIportComboBox.addItems(ports)
        self.CLIportComboBox.currentIndexChanged.connect(self.selectCLIPort)

        self.DATAportComboBox.addItems(ports)
        self.DATAportComboBox.currentIndexChanged.connect(self.selectDATAPort)

        serialPortsHBox.addWidget(QLabel(text='CLI Port: ', alignment=Qt.AlignCenter))
        serialPortsHBox.addWidget(self.CLIportComboBox)
        serialPortsHBox.addWidget(QLabel(text='DATA Port: ', alignment=Qt.AlignCenter))
        serialPortsHBox.addWidget(self.DATAportComboBox)

        # Config File Row
        configFileHBox = QHBoxLayout()
        self.configFileButton = QPushButton(text="Select Config File")
        self.configFileButton.clicked.connect(self.getConfigFile)
        self.configFileLabel = QLabel(text='Config Filename', alignment=Qt.AlignCenter)

        configFileHBox.addWidget(self.configFileButton)
        configFileHBox.addWidget(self.configFileLabel)

        # Start and Stop Sensor Row
        startStopButtonsHBox = QHBoxLayout()
        self.initializeButton = QPushButton(text="Initialize Sensor")
        self.initializeButton.clicked.connect(self.initializeSensor)
        self.startStopButton = QPushButton(text='Stop Sensor')
        self.startStopButton.setEnabled(False)
        self.startStopButton.clicked.connect(self.startStopSensor)

        startStopButtonsHBox.addWidget(self.initializeButton)
        startStopButtonsHBox.addWidget(self.startStopButton)
        
        settingsVBox.addLayout(serialPortsHBox)
        settingsVBox.addLayout(configFileHBox)
        settingsVBox.addLayout(startStopButtonsHBox)
        self.settingsGroupBox.setLayout(settingsVBox)
        
    def selectCLIPort(self):
        self.CLIportPath = self.CLIportComboBox.currentText()
        print('{} is selected as the CLI Port'.format(self.CLIportPath))

    def selectDATAPort(self):
        self.DATAportPath = self.DATAportComboBox.currentText()
        print('{} is selected as the DATA Port'.format(self.DATAportPath))

    def getConfigFile(self):
        self.configFileName, _ = QFileDialog.getOpenFileName(self)
        self.configFileLabel.setText(self.configFileName)
        print("'{}' selected as config file".format(self.configFileName))
        self.initializeButton.setEnabled(True)

    def initializeSensor(self):
        self.thread()
        self.sensorIsRunning = True
        self.initializeButton.setEnabled(False)
        self.startStopButton.setEnabled(True)
        self.rec_button.setEnabled(False)

    def startStopSensor(self):
        if self.sensorIsRunning:
            self.reader.CLIport.write(('sensorStop\n').encode())
            print('sensorStop')
            self.startStopButton.setText('Start Sensor')
            self.rec_button.setEnabled(True)

        else:
            self.reader.CLIport.write(('sensorStart\n').encode())
            print('sensorStart')
            self.startStopButton.setText('Stop Sensor')
            self.rec_button.setEnabled(False)

        self.sensorIsRunning = not self.sensorIsRunning

    # ----- END OF CONFIGURATION PART -----


    def createTerminalGroupBox(self):
        terminalVbox = QVBoxLayout()

        self.process = QTextEdit(self)
        self.process.moveCursor(QtGui.QTextCursor.Start)
        self.process.ensureCursorVisible()
        # self.process.setLineWrapColumnOrWidth(500)
        # self.process.setLineWrapMode(QTextEdit.FixedPixelWidth)

        terminalVbox.addWidget(self.process)
        self.terminalGroupBox.setLayout(terminalVbox)

    def createRecordGroupBox(self):
        recordHbox = QHBoxLayout()

        self.rec_button = QPushButton('Record Data')
        self.rec_button.setCheckable(True)
        self.rec_button.clicked[bool].connect(self.recordData)
        
        # self.filename_label = QLabel(self, text='Filename will be shown here upon recording', alignment=Qt.AlignLeft)
        self.filenameTextBox = QLineEdit()
        
        recordHbox.addWidget(self.rec_button)
        recordHbox.addWidget(QLabel(text='Filename: ', alignment=Qt.AlignRight))
        recordHbox.addWidget(self.filen)
        self.recordGroupBox.setLayout(recordHbox)


    def recordData(self, pressed):
        if pressed:
            print('Data is being recorded')
        else:
            print('Data is not being recorded')
        self.filename_label.setText("raw_" + str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")) +'.pkl')

    def center(self):
        """centers the window on the screen"""
        screen = QDesktopWidget().screenGeometry()
        size = self.geometry()
        self.move((screen.width() - size.width()) // 2,
                  (screen.height() - size.height()) // 2)


    

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()

    # behaviour to trigger on exit
    sys.exit(app.exec_())