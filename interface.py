import os.path
import sys
import datetime
import threading
import platform
import serial.tools.list_ports
import time
import pickle
import numpy as np

import torch.nn as nn
import torch
from dl_model.model import wordNet
from pre_processing.main_preprocess import cluster, normalize, decay, createMultiview, saveFig

from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5 import QtGui

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import IWR1443_reader

# change these values depending on your machine for convenience
DEFAULT_CONFIG_FILE = '1443config.cfg'
# DEFAULT_CONFIG_FILE = '1443_3d.cfg'

# Windows
# DEFAULT_CLI_PORT = 'COM4'
# DEFAULT_DATA_PORT = 'COM5'

# Mac
DEFAULT_CLI_PORT = '/dev/cu.usbmodemR10310411'
DEFAULT_DATA_PORT = '/dev/cu.usbmodemR10310414'

# Define labels
CLASSES = open('dl_model/glosses','r',encoding='utf-8-sig').readlines()
CLASSES = [ gloss.strip() for gloss in CLASSES ]

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

        #Set default settings
        portsSet, configFileSet = False, False

        # Set serial ports by default
        if platform.system() == 'Windows':
            CLIportPath = DEFAULT_CLI_PORT
            DATAportPath = DEFAULT_DATA_PORT
        elif platform.system() == 'Darwin':
            CLIportPath = '/dev/cu.usbmodemR10310411'
            DATAportPath = '/dev/cu.usbmodemR10310414'

        if CLIportPath in self.ports:
            CLI_index = self.ports.index(CLIportPath)
            DATA_index = self.ports.index(DATAportPath)

            self.CLIportComboBox.setCurrentIndex(CLI_index)
            self.DATAportComboBox.setCurrentIndex(DATA_index)

            portsSet = True

        # Set config file by default
        if os.path.isfile('./' + DEFAULT_CONFIG_FILE):
            self.configFileName = DEFAULT_CONFIG_FILE
            self.configFileLabel.setText(self.configFileName)
            print("[BOARD CONFIG] '{}' selected as config file".format(self.configFileName))

            configFileSet = True

        # Enable init buttons if ports and config file are set
        if portsSet and configFileSet:
            self.initializeButton.setEnabled(True)


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

        # QVBoxLayouts for right side layouts
        rightLayout = QVBoxLayout()
        plotLayout = QVBoxLayout()
        modelLayout = QVBoxLayout()
        
        # QTabWidget for spltting data collection and model 
        tabWidget = QTabWidget()
        plotWidget = QWidget()
        modelWidget = QWidget()
        tabWidget.addTab(plotWidget, 'Plot')
        tabWidget.addTab(modelWidget, 'Model')

        # Plot
        self.plotGroupBox = QGroupBox('Plot')
        self.createPlotGroupBox()

        # Plot settings
        self.plotSettingsGroupBox = QGroupBox('Plot Settings')
        self.createPlotSettingsGroupBox()

        # Model settings
        self.modelSettingsGroupBox = QGroupBox('Load Model')
        self.createModelSettings()

        # Putting right layout together
        plotLayout.addWidget(self.plotGroupBox)
        plotLayout.addWidget(self.plotSettingsGroupBox)
        plotWidget.setLayout(plotLayout)

        modelLayout.addWidget(self.modelSettingsGroupBox)
        modelWidget.setLayout(modelLayout)

        rightLayout.addWidget(tabWidget)

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
        self.ports = [port for port, _, _ in serial.tools.list_ports.comports()]

        self.CLIportComboBox.addItems(self.ports)
        self.CLIportComboBox.currentIndexChanged.connect(self.selectCLIPort)

        self.DATAportComboBox.addItems(self.ports)
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
        self.initializeButton.setEnabled(False)

        startStopButtonsHBox.addWidget(self.initializeButton)
        
        settingsVBox.addLayout(serialPortsHBox)
        settingsVBox.addLayout(configFileHBox)
        settingsVBox.addLayout(startStopButtonsHBox)
        self.settingsGroupBox.setLayout(settingsVBox)
        
    def selectCLIPort(self):
        self.CLIportPath = self.CLIportComboBox.currentText()
        print('[BOARD CONFIG] {} is selected as the CLI Port'.format(self.CLIportPath))

    def selectDATAPort(self):
        self.DATAportPath = self.DATAportComboBox.currentText()
        print('[BOARD CONFIG] {} is selected as the DATA Port'.format(self.DATAportPath))

    def getConfigFile(self):
        self.configFileName, _ = QFileDialog.getOpenFileName(self, 'Open Config File', './' + DEFAULT_CONFIG_FILE, 'Config Files (*.cfg)')
        if self.configFileName == '':
            if os.path.isfile('./' + DEFAULT_CONFIG_FILE):
                self.configFileName = DEFAULT_CONFIG_FILE
                self.configFileLabel.setText(self.configFileName)
                print("[BOARD CONFIG] '{}' selected as config file by default".format(self.configFileName))
            else:
                self.configFileLabel.setText('No Config File Selected')
                print('[BOARD CONFIG] Please select a config file')
        else:
            self.configFileLabel.setText(self.configFileName)
            print("[BOARD CONFIG] '{}' selected as config file".format(self.configFileName))
            self.initializeButton.setEnabled(True)

    def initializeSensor(self):
        try:
            self.thread()
            self.sensorIsRunning = True
            self.initializeButton.setEnabled(False)
            self.startStopButton.setEnabled(True)
            self.rec_button.setEnabled(False)
            self.CLIportComboBox.setEnabled(False)
            self.DATAportComboBox.setEnabled(False)
            self.configFileButton.setEnabled(False)
        except:
            print("[BOARD CONFIG] Error initializing sensor")

    # ----- END OF CONFIGURATION PART -----

    # ----- TERMINAL PART -----

    def createTerminalGroupBox(self):
        terminalVbox = QVBoxLayout()

        self.process = QTextEdit(self)
        self.process.moveCursor(QtGui.QTextCursor.Start)
        self.process.ensureCursorVisible()
        # self.process.setLineWrapColumnOrWidth(500)
        # self.process.setLineWrapMode(QTextEdit.FixedPixelWidth)

        self.startStopButton = QPushButton(text='Stop Sensor')
        self.startStopButton.setEnabled(False)
        self.startStopButton.clicked.connect(self.startStopSensor)

        self.rightAfterRecord = False

        terminalVbox.addWidget(self.process)
        terminalVbox.addWidget(self.startStopButton)
        self.terminalGroupBox.setLayout(terminalVbox)

    def startStopSensor(self):
        if self.sensorIsRunning:
            # Stop sensor
            self.reader.CLIport.write(('sensorStop\n').encode())
            print('[BOARD] sensorStop')
            self.startStopButton.setText('Start Sensor')
            self.rec_button.setEnabled(True)

            # Log data
            if self.rec_button.isChecked():
                # Log file
                self.filename = self.filenameTextBox.text()
                self.reader.logFile(filename=self.filename)
                print("[DATA LOGGING] Data successfully saved to '{}.pkl'".format(self.filename))
                self.rec_button.click()
                
                # Plot logged file
                self.rightAfterRecord = True
                self.pklFileLabel.setText(self.filename+'.pkl')
                self.loadPklFile()
                self.rightAfterRecord = False

            if self.realtimeButton.isChecked():
                try:
                    self.infer(data=self.reader.frameData)
                except:
                    print("[MODEL] Failed to recognize data")

        else:
            # Start sensor
            
            # Reset data dictionary and timer
            self.reader.frameData = {}
            self.reader.start_time = time.time()

            self.reader.CLIport.write(('sensorStart\n').encode())
            print('[BOARD] sensorStart')
            self.startStopButton.setText('Stop Sensor')
            self.rec_button.setEnabled(False)

        self.sensorIsRunning = not self.sensorIsRunning

    # ----- END OF TERMINAL PART -----

    def createRecordGroupBox(self):
        recordHbox = QHBoxLayout()

        self.rec_button = QPushButton('Record Data')
        self.rec_button.setCheckable(True)
        self.rec_button.clicked[bool].connect(self.recordData)
        
        self.filenameTextBox = QLineEdit()
        
        recordHbox.addWidget(self.rec_button)
        recordHbox.addWidget(QLabel(text='Filename (.pkl): '))
        recordHbox.addWidget(self.filenameTextBox)
        self.recordGroupBox.setLayout(recordHbox)

    def recordData(self, pressed):
        if pressed:
            if self.filenameTextBox.text() == '':
                self.filenameTextBox.setText("raw_" + str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
            print("[DATA LOGGING] Data is being recorded with the filename '{}.pkl'".format(self.filenameTextBox.text()))
            self.filenameTextBox.setEnabled(False)
        else:
            self.filenameTextBox.setEnabled(True)
            self.filenameTextBox.clear()
            print('[DATA LOGGING] Data is not being recorded; Filename is cleared')

    # ----- PLOT PART -----

    def createPlotGroupBox(self):
        plotVBox = QVBoxLayout()

        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.plotButton = QPushButton('Plot')
        self.plotButton.clicked.connect(self.plotData)

        plotVBox.addWidget(self.toolbar)
        plotVBox.addWidget(self.canvas)
        plotVBox.addWidget(self.plotButton)

        self.plotGroupBox.setLayout(plotVBox)

    def plotData(self):
        # Disable buttons to prevent interrupting the plot
        self.plotButton.setEnabled(False)
        self.pklFileButton.setEnabled(False)

        # Plot data
        self.figure.clear()

        self.ax = self.figure.add_subplot(111, projection='3d')

        self.ax.set_xlim3d(-3, 3)
        self.ax.set_ylim3d(-3, 3)
        self.ax.set_zlim3d(-3, 3)

        self.ax.set_xlabel('x [m]')
        self.ax.set_ylabel('y [m]')
        self.ax.set_zlabel('z [m]')

        self.graph = self.ax.scatter([], [], [], s=5)
        
        self.ani = FuncAnimation(self.figure, self.updatePlot, self.timestamps, interval=33, blit=False, repeat=False)

        self.canvas.draw()

    def updatePlot(self, timestamp):
        xs = self.data[timestamp][:,0]
        ys = self.data[timestamp][:,1]
        zs = self.data[timestamp][:,2]
        
        self.graph._offsets3d = (xs, ys, zs)
        self.ax.set_title('Timestamp: {}'.format(timestamp))

        if timestamp == self.timestamps[len(self.timestamps)-1]:
            print("[PLOTTING] Animation done")
            self.plotButton.setEnabled(True)
            self.pklFileButton.setEnabled(True)

    # ----- END OF PLOT PART -----

    # ----- PLOT SETTINGS PART -----

    def createPlotSettingsGroupBox(self):
        plotSettingsHBox = QHBoxLayout()

        self.pklFileButton = QPushButton('Load .pkl File')
        self.pklFileButton.clicked.connect(self.loadPklFile)
        self.pklFileLabel = QLabel(text='pkl Filename', alignment=Qt.AlignCenter)

        plotSettingsHBox.addWidget(self.pklFileButton)
        plotSettingsHBox.addWidget(self.pklFileLabel)

        self.plotSettingsGroupBox.setLayout(plotSettingsHBox)

    def loadPklFile(self):
        if self.rightAfterRecord:
            self.pklFileName = self.pklFileLabel.text()
        else:
            self.pklFileName, _ = QFileDialog.getOpenFileName(self, 'Open PKL File', './', 'PKL Files (*.pkl)')
            if self.pklFileName == '':
                print('[PLOT SETTINGS] No pkl file selected')
                return
            self.pklFileLabel.setText(self.pklFileName)
        
        print("[PLOT SETTINGS] PKL file '{}' loaded".format(self.pklFileName))

        with open(self.pklFileName, 'rb') as handle:
            self.data = pickle.load(handle)
        
        self.timestamps = list(self.data.keys())

    # ----- END OF PLOT SETTINGS PART -----

    # ----- MODEL SETTINGS PART -----

    def createModelSettings(self):
        '''
        Widgets needed:
        - .pth file selection
        - cuda or cpu select
        - ? input file (can be set to mirror record filename)
        - display gloss
        '''
        modelSettingsVBox = QVBoxLayout()

        # Model file select row
        pthFileHBox = QHBoxLayout()
        self.pthFileButton = QPushButton(text="Select pre-trained model")
        self.pthFileButton.clicked.connect(self.getPthFile)
        self.pthFileLabel = QLabel(text='.pth Filename', alignment=Qt.AlignCenter)

        pthFileHBox.addWidget(self.pthFileButton)
        pthFileHBox.addWidget(self.pthFileLabel)

        # Load model button
        self.loadModelButton = QPushButton('Load Model')
        self.loadModelButton.clicked.connect(self.loadModel)

        #Toggle button for activating realtime
        self.realtimeButton = QPushButton('Real-time Infer')
        self.realtimeButton.setCheckable(True)
        self.realtimeButton.clicked[bool].connect(self.realtimeInfer)
        self.realtimeButton.setEnabled(False)

        #Pkl File select for non-realtime
        modelPklFileHBox = QHBoxLayout()
        self.modelPklFileButton = QPushButton(text="Select pkl file to infer")
        self.modelPklFileButton.clicked.connect(self.getModelPklFile)
        self.modelPklFileButton.setEnabled(False)
        self.modelPklFileLabel = QLabel(text='.pkl Filename', alignment=Qt.AlignCenter)

        modelPklFileHBox.addWidget(self.modelPklFileButton)
        modelPklFileHBox.addWidget(self.modelPklFileLabel)

        self.manualInferButton = QPushButton('Manual Infer')
        self.manualInferButton.clicked.connect(self.infer)
        self.manualInferButton.setEnabled(False)

        self.glossLabel = QLabel(text='Translation here', alignment=Qt.AlignCenter)
        self.glossLabel.setFont(QtGui.QFont('Helvetica', 32))

        self.timerLabel = QLabel(text='Latency', alignment=Qt.AlignCenter)
        
        modelSettingsVBox.addLayout(pthFileHBox)
        modelSettingsVBox.addWidget(self.loadModelButton)
        modelSettingsVBox.addWidget(self.realtimeButton)
        modelSettingsVBox.addLayout(modelPklFileHBox)
        modelSettingsVBox.addWidget(self.manualInferButton)
        modelSettingsVBox.addWidget(self.glossLabel)
        modelSettingsVBox.addWidget(self.timerLabel)
        self.modelSettingsGroupBox.setLayout(modelSettingsVBox)

    def getPthFile(self):
        self.pthFileName, _ = QFileDialog.getOpenFileName(self, 'Select model file', './dl_model/checkpoints/', 'Pth Files (*.pth)')
        if self.pthFileName == '':
            self.pthFileLabel.setText('No .pth File Selected')
            print('[MODEL CONFIG] Please select a model file')
        else:
            self.pthFileLabel.setText(self.pthFileName)
            print("[MODEL CONFIG] '{}' selected as .pth file".format(self.pthFileName))
            # initialize model buttons here

    def loadModel(self):
        try:
            torch.manual_seed(1)
            torch.cuda.manual_seed(1)
            np.random.seed(1)
            torch.backends.cudnn.deterministic = True

            self.net = wordNet(2048, class_size=len(CLASSES), num_layers=2, batch_size=5, dropout=0.65, use_cuda=False, frameCount=10, dataParallel=True)
            self.net.load_state_dict(torch.load(self.pthFileLabel.text(), map_location='cpu'), strict=False)
            self.m = nn.Softmax(dim=1)
            self.net.eval()

            print("[MODEL CONFIG] '{}' model loaded".format(self.pthFileLabel.text().split('/')[-1]))

            # Activate model buttons
            self.realtimeButton.setEnabled(True)
            self.modelPklFileButton.setEnabled(True)
            self.manualInferButton.setEnabled(True)
        except:
            print("[MODEL CONFIG] Failed to load selected model")

    def realtimeInfer(self, pressed):
        if pressed:
            self.modelPklFileButton.setEnabled(False)
            self.manualInferButton.setEnabled(False)
        else:
            self.modelPklFileButton.setEnabled(True)
            self.manualInferButton.setEnabled(True)

    def getModelPklFile(self):
        self.modelPklFileName, _ = QFileDialog.getOpenFileName(self, 'Open PKL File', './data/', 'PKL Files (*.pkl)')
        if self.modelPklFileName == '':
            print('[MODEL CONFIG] No pkl file selected')
            return
        self.modelPklFileLabel.setText(self.modelPklFileName)

    def infer(self, data=None):
        time_start = time.time()

        if data == None:
            print('infering', self.modelPklFileLabel.text())
            with open(self.modelPklFileLabel.text(), "rb") as pm_data:
                pm_contents = pickle.load(pm_data, encoding="bytes")

        else:
            print('infering from recording')
            pm_contents = data

        f = 10 #number of frames

        #Outlier Removal and Translation
        c = 0
        for key, pts in pm_contents.items():
            c += len(pts)
            pm_contents[key] = cluster(pts, e = 0.8, outlier=True)
            pm_contents[key] = normalize(pts)

        # Aggregate Frames
        aggframes = decay(pm_contents, c, f)


        # Cluster
        clustFrames = []                        # array to contain dictionaries
        for _, xyz in aggframes.items():        # iterated len(aggframes) times which is num of frames
            if(c < 10):
                clustFrames.append(dict({0:xyz}))
                continue
            clust = cluster(xyz, e = 0.5, min_samp = 5, outlier=True)       # dictionary with key color c,and item of np array size n x 3 (pts)
            if(len(clust) == 0): continue
            clustFrames.append(clust)

        while(len(clustFrames) < 10):
            clustFrames.append(dict({0:np.zeros((1,3))}))

        # Multiview
        data = {'xy': [], 'yz': [], 'xz': []} 
        for _3dframe in clustFrames:
            xyf,yzf,xzf = createMultiview(_3dframe)      # tuple containing 3 dictionaries xy, yz, xz 2dframes
            if(len(xyf.values()) == 0): continue
            data['xy'].append(np.concatenate(list(xyf.values())))
            data['yz'].append(np.concatenate(list(yzf.values())))
            data['xz'].append(np.concatenate(list(xzf.values())))

        outData = {'xy': [], 'yz': [], 'xz': []}
        for view in data.keys():
            for frame in data[view]:
                outData[view].append(saveFig(frame, axis=view, reSize=True, saveNumpy=True))
            
        data = [np.concatenate((outData['xy'], outData['yz'], outData['xz']), axis=0)]
        # print(data)
        preprocessed = torch.tensor(data, dtype=torch.float32)

        # model predict
        o = self.net(preprocessed, 10)
        prediction = torch.max(self.m(o), dim=1)[1].cpu().numpy().tolist()
        print(CLASSES[prediction[0]])
        self.glossLabel.setText(CLASSES[prediction[0]])

        time_end = time.time() - time_start
        self.timerLabel.setText(str(time_end))

    # ----- END OF MODEL SETTINGS PART -----

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