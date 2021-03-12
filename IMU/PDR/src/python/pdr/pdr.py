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
        self.k = 0.51
        self.sampleTimestamps = []
        self.sampleCount = 0
        self.accData = []
        self.signals = []

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
        self.accData.append([timestamp, acceleratorData[0], acceleratorData[1], acceleratorData[2]])

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

    # TODO: Find threshold for step detection.
    # Step-length estimation.
    def __stepLength(self, timestamp):
        peak_detection = step.Step.peak_detection(self.accData, 10)
        avg_filter = peak_detection['avgFilter']
        window = self.__step_window(peak_detection['signals'])

        if window == None:
            return 0

        avg_filter_cut = avg_filter     # Temporary
        return self.k * math.pow(np.max(avg_filter_cut) - np.amin(avg_filter_cut), 1 / 4)   # Weinberg.

    # TODO: Finish this.
    # TODO: A window must start with the first 1 in a step and end with the last -1 in the same step.
    # Computes single signal window indices. If step is not complete, return None.
    def __step_window(self, signals):
        return 0

# Class for heading estimation using AHRS.
class AHRSPDR(BasePDR):
    def __init__(self, initial_location):
        super().__init__(initial_location)

    def heading(self, acceleratorData, magnometerData):
        return 0.5
