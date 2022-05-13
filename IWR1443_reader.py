import serial
import time
import datetime
import pickle
import numpy as np

# from interface import *
from PyQt5.QtCore import *

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#Constants

OBJ_STRUCT_SIZE_BYTES = 12
BYTE_VEC_ACC_MAX_SIZE = 2**15
MMWDEMO_UART_MSG_DETECTED_POINTS = 1
MMWDEMO_UART_MSG_RANGE_PROFILE = 2

class IWR1443_Reader:
    def __init__(self, configFilename, CLIportPath, DATAportPath):
        '''serialConfig'''
        self.configFilename = configFilename
        self.configParameters = {} # Initialize an empty dictionary to store the configuration parameters

        self.frameData = {} # Store data here

        # Initialize number of antennas
        self.numRxAnt = 4
        self.numTxAnt = 3

        # Initialize Constants
        self.maxBufferSize = 2**15
        self.magicWord = [2, 1, 4, 3, 6, 5, 8, 7]
        self.byteBuffer = np.zeros(2**15,dtype = 'uint8')
        self.byteBufferLength = 0
        
        self.CLIport = serial.Serial(CLIportPath, 115200)
        self.DATAport = serial.Serial(DATAportPath, 921600)

        # Read config file and send to board
        config = [line.rstrip('\r\n') for line in open(self.configFilename)]
        for i in config:
            self.CLIport.write((i+'\n').encode())
            print(i)
            time.sleep(0.01)

        self.parseConfigFile()

    def parseConfigFile(self):
        config = [line.rstrip('\r\n') for line in open(self.configFilename)]
        for i in config:
            splitWords = i.split(" ")

            if "profileCfg" in splitWords[0]:
                startFreq = int(float(splitWords[2]))
                idleTime = int(splitWords[3])
                rampEndTime = float(splitWords[5])
                freqSlopeConst = float(splitWords[8])
                numAdcSamples = int(splitWords[10])
                numAdcSamplesRoundTo2 = 1
                
                while numAdcSamples > numAdcSamplesRoundTo2:
                    numAdcSamplesRoundTo2 = numAdcSamplesRoundTo2 * 2
                    
                digOutSampleRate = int(splitWords[11])

            elif "frameCfg" in splitWords[0]:
                chirpStartIdx = int(splitWords[1])
                chirpEndIdx = int(splitWords[2])
                numLoops = int(splitWords[3])
                numFrames = int(splitWords[4])
                framePeriodicity = float(splitWords[5])
            
        numChirpsPerFrame = (chirpEndIdx - chirpStartIdx + 1) * numLoops
        self.configParameters["numDopplerBins"] = numChirpsPerFrame / self.numTxAnt
        self.configParameters["numRangeBins"] = numAdcSamplesRoundTo2
        self.configParameters["rangeResolutionMeters"] = (3e8 * digOutSampleRate * 1e3) / (2 * freqSlopeConst * 1e12 * numAdcSamples)
        self.configParameters["rangeIdxToMeters"] = (3e8 * digOutSampleRate * 1e3) / (2 * freqSlopeConst * 1e12 * self.configParameters["numRangeBins"])
        self.configParameters["dopplerResolutionMps"] = 3e8 / (2 * startFreq * 1e9 * (idleTime + rampEndTime) * 1e-6 * self.configParameters["numDopplerBins"] * self.numTxAnt)
        self.configParameters["maxRange"] = (300 * 0.9 * digOutSampleRate)/(2 * freqSlopeConst * 1e3)
        self.configParameters["maxVelocity"] = 3e8 / (4 * startFreq * 1e9 * (idleTime + rampEndTime) * 1e-6 * self.numTxAnt)

    def readData(self):
        #Initialize Variables
        self.magicOK = 0
        self.dataOK = 0
        self.frameNumber = 0
        self.detObj = {}
        self.tlv_type = 0

        readBuffer = self.DATAport.read(self.DATAport.in_waiting)
        self.byteVec = np.frombuffer(readBuffer, dtype='uint8')
        self.byteCount = len(self.byteVec)

        # Check that the buffer is not full, and then add the data to the buffer
        if (self.byteBufferLength + self.byteCount) < self.maxBufferSize:
            self.byteBuffer[self.byteBufferLength:self.byteBufferLength + self.byteCount] = self.byteVec[:self.byteCount]
            self.byteBufferLength = self.byteBufferLength + self.byteCount

        # Check that the buffer has some data
        if self.byteBufferLength > 16:
            
            # Check for all possible locations of the magic word
            possibleLocs = np.where(self.byteBuffer == self.magicWord[0])[0]

            # Confirm that is the beginning of the magic word and store the index in startIdx
            startIdx = []
            for loc in possibleLocs:
                check = self.byteBuffer[loc:loc+8]
                if np.all(check == self.magicWord):
                    startIdx.append(loc)
                
            # Check that startIdx is not empty
            if startIdx:
                
                # Remove the data before the first start index
                if startIdx[0] > 0 and startIdx[0] < self.byteBufferLength:
                    self.byteBuffer[:self.byteBufferLength-startIdx[0]] = self.byteBuffer[startIdx[0]:self.byteBufferLength]
                    self.byteBuffer[self.byteBufferLength-startIdx[0]:] = np.zeros(len(self.byteBuffer[self.byteBufferLength-startIdx[0]:]),dtype = 'uint8')
                    self.byteBufferLength = self.byteBufferLength - startIdx[0]
                    
                # Check that there have no errors with the byte buffer length
                if self.byteBufferLength < 0:
                    self.byteBufferLength = 0

                self.word = [1, 2**8, 2**16, 2**24] # word array to convert 4 bytes to a 32 bit number
                
                # Read the total packet length
                totalPacketLen = np.matmul(self.byteBuffer[12:12+4], self.word)
                
                # Check that all the packet has been read
                if (self.byteBufferLength >= totalPacketLen) and (self.byteBufferLength != 0):
                    self.magicOK = 1

        # If magicOK is equal to 1 then process the message
        if self.magicOK:
            
            # Initialize the pointer index
            idX = 0

            self.word = [1, 2**8, 2**16, 2**24] # word array to convert 4 bytes to a 32 bit number
            
            # Read the header
            magicNumber = self.byteBuffer[idX:idX+8]
            idX += 8
            version = format(np.matmul(self.byteBuffer[idX:idX+4], self.word), 'x')
            idX += 4
            totalPacketLen = np.matmul(self.byteBuffer[idX:idX+4], self.word)
            idX += 4
            platform = format(np.matmul(self.byteBuffer[idX:idX+4], self.word),'x')
            idX += 4
            frameNumber = np.matmul(self.byteBuffer[idX:idX+4], self.word)
            idX += 4
            timeCpuCycles = np.matmul(self.byteBuffer[idX:idX+4], self.word)
            idX += 4
            numDetectedObj = np.matmul(self.byteBuffer[idX:idX+4], self.word)
            idX += 4
            numTLVs = np.matmul(self.byteBuffer[idX:idX+4], self.word)
            idX += 4
            
            # Read the TLV messages
            for tlvIdx in range(numTLVs):

                # Check the header of the TLV message
                try:
                    tlv_type = np.matmul(self.byteBuffer[idX:idX+4], self.word)
                    idX += 4
                    tlv_length = np.matmul(self.byteBuffer[idX:idX+4], self.word)
                    idX += 4
                except:
                    pass
                
                # Read the data depending on the TLV message
                if tlv_type == MMWDEMO_UART_MSG_DETECTED_POINTS:
                                
                    # word array to convert 4 bytes to a 16 bit number
                    self.word = [1, 2**8]
                    tlv_numObj = np.matmul(self.byteBuffer[idX:idX+2], self.word)
                    idX += 2
                    tlv_xyzQFormat = 2**np.matmul(self.byteBuffer[idX:idX+2], self.word)
                    idX += 2
                    
                    # Initialize the arrays
                    rangeIdx = np.zeros(tlv_numObj,dtype = 'int16')
                    dopplerIdx = np.zeros(tlv_numObj,dtype = 'int16')
                    peakVal = np.zeros(tlv_numObj,dtype = 'int16')
                    x = np.zeros(tlv_numObj,dtype = 'int16')
                    y = np.zeros(tlv_numObj,dtype = 'int16')
                    z = np.zeros(tlv_numObj,dtype = 'int16')
                    
                    for objectNum in range(tlv_numObj):
                        
                        # Read the data for each object
                        rangeIdx[objectNum] =  np.matmul(self.byteBuffer[idX:idX+2], self.word)
                        idX += 2
                        dopplerIdx[objectNum] = np.matmul(self.byteBuffer[idX:idX+2], self.word)
                        idX += 2
                        peakVal[objectNum] = np.matmul(self.byteBuffer[idX:idX+2], self.word)
                        idX += 2
                        x[objectNum] = np.matmul(self.byteBuffer[idX:idX+2], self.word)
                        idX += 2
                        y[objectNum] = np.matmul(self.byteBuffer[idX:idX+2], self.word)
                        idX += 2
                        z[objectNum] = np.matmul(self.byteBuffer[idX:idX+2],self.word)
                        idX += 2
                        
                    # Make the necessary corrections and calculate the rest of the data
                    rangeVal = rangeIdx * self.configParameters["rangeIdxToMeters"]
                    dopplerIdx[dopplerIdx > (self.configParameters["numDopplerBins"]/2 - 1)] = dopplerIdx[dopplerIdx > (self.configParameters["numDopplerBins"]/2 - 1)] - 65535
                    dopplerVal = dopplerIdx * self.configParameters["dopplerResolutionMps"]
                    #x[x > 32767] = x[x > 32767] - 65536
                    #y[y > 32767] = y[y > 32767] - 65536
                    #z[z > 32767] = z[z > 32767] - 65536
                    x = x / tlv_xyzQFormat
                    y = y / tlv_xyzQFormat
                    z = z / tlv_xyzQFormat
                    
                    # Store the data in the detObj dictionary
                    self.detObj = {"numObj": tlv_numObj, "rangeIdx": rangeIdx, "range": rangeVal, "dopplerIdx": dopplerIdx, \
                            "doppler": dopplerVal, "peakVal": peakVal, "x": x, "y": y, "z": z, "ts": str(time.time()-self.start_time)}

                    # Append detObj to frameData
                    self.frameData[self.detObj["ts"]] = np.dstack([self.detObj["x"], self.detObj["y"], self.detObj["z"]])[0]
                    print(self.frameData[self.detObj["ts"]])
                    # print('hello')

                    self.dataOK = 1
            
            # Remove already processed data
            if idX > 0 and self.byteBufferLength > idX:
                shiftSize = totalPacketLen
                        
                self.byteBuffer[:self.byteBufferLength - shiftSize] = self.byteBuffer[shiftSize:self.byteBufferLength]
                self.byteBuffer[self.byteBufferLength - shiftSize:] = np.zeros(len(self.byteBuffer[self.byteBufferLength - shiftSize:]),dtype = 'uint8')
                self.byteBufferLength = self.byteBufferLength - shiftSize
                
                # Check that there are no errors with the buffer length
                if self.byteBufferLength < 0:
                    self.byteBufferLength = 0

    def logFile(self):
        filename = "raw_" + str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")) +'.pkl'
        with open(filename, 'wb') as handle:
            pickle.dump(self.frameData, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def loop(self):
        self.start_time = time.time()
        self.loop_running = True
        while self.loop_running:
            try:
                self.readData()
                # self.frameData[self.detObj["ts"]] = np.dstack([self.detObj["x"], self.detObj["y"], self.detObj["z"]])[0]
                # print(self.frameData[self.detObj["ts"]])
                time.sleep(1/30) #
            except KeyboardInterrupt:
                self.CLIport.write(('sensorStop\n').encode())
                self.CLIport.close()
                self.DATAport.close()
                break
    
    def stop(self):
        self.loop_running = False
        time.sleep(0.1)
        self.CLIport.write(('sensorStop\n').encode())
        self.CLIport.close()
        self.DATAport.close()

if __name__ == '__main__':
    configFileName = '1443config.cfg' 
    CLIportPath = '/dev/tty.usbmodemR10310411'
    DATAportPath = '/dev/tty.usbmodemR10310414'

    reader = IWR1443_Reader(configFileName, CLIportPath, DATAportPath)
    reader.loop() 

    reader.logFile()
