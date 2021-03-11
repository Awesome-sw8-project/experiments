import math
import time
import threading
import numpy as np
from pdr import filter
from pdr import util
from pdr import step

class Location:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def getX(self):
        return self.x

    def getY(self):
        return self.y

# Base class for PDR estimation with unspecified heading estimation.
class BasePDR:
    def __init__(self, startLocation):
        self.start = startLocation
        self.current = startLocation
        self.sampleTimestamps = []
        self.sampleCount = 0
        self.accData = []
        self.lastStepTimestamp = 0

    # Getter to initial position.
    def getStartLocation(self):
        return self.start

    # Getter to current position.
    def getCurrentLocation(self, timestamp, acceleratorData, magnometerData, gyroscopeData):
        self.__nextPosition(timestamp, acceleratorData, magnometerData, gyroscopeData)
        return self.current

    # Polymorphic heading estimation.
    def heading(self, acceleratorData, magnometerData):
        pass

    # Computes next position.
    def __nextPosition(self, timestamp, acceleratorData, magnometerData, gyroscopeData):
        self.sampleTimestamps.append(time.clock_gettime_ns(time.pthread_getcpuclockid(threading.get_ident())))
        self.sampleCount += 1
        acceleratorData.insert(0, timestamp)
        self.accData.append(acceleratorData)

        stepLength = self.__stepLength(timestamp)
        heading = self.heading(acceleratorData, magnometerData)
        x = self.current.getX() + stepLength * math.cos(heading)
        y = self.current.getY() + stepLength * math.sin(heading)
        self.current = Location(x, y)

    # Computes average sample rate in kHz.
    def __avgSampleRate(self):
        times = []
        i = 1

        while i < len(self.sampleTimestamps):
            times.append(self.sampleTimestamps[i] - self.sampleTimestamps[i - 1])
            i += 1

        return np.sum(np.array(times)) / len(times)

    # Step-length estimation.
    def __stepLength(self, timestamp):
        stepTimestamps, stepIndices, stepAcceMaxMins = \
            step.Step.detect(self.accData)

        if (stepTimestamps[len(stepTimestamps) - 1] == self.lastStepTimestamp):
            return 0

        self.lastStepTimestamps.append(stepTimestamps[len(stepTimestamps) - 1])
        maxStep = stepAcceMaxMins[len(stepAcceMaxMins) - 1][1]
        minStep = stepAcceMaxMins[len(stepAcceMaxMins) - 1][2]
        return self.k * math.pow(maxStep - minStep, 1 / 4)   # Weinberg.

# Class for heading estimation using AHRS.
class AHRSPDR(BasePDR):
    def __init__(self):
        pass

    def heading(self, acceleratorData, magnometerData):
        pass
