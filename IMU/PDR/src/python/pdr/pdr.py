import math
import time
import threading
import numpy as np
from pdr import filter
from pdr import util
from pdr import step
from ahrs.filters import Madgwick
from ahrs.common import Quaternion

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
        self.gyroData = []
        self.magnoData = []

    # Getter to initial position.
    def getStartLocation(self):
        return self.start

    # Getter to current position.
    def getCurrentLocation(self, timestamp, acceleratorData, magnometerData, gyroscopeData):
        self.__nextPosition(timestamp, acceleratorData, magnometerData, gyroscopeData)
        return self.current

    # Polymorphic heading estimation.
    def heading(self, acceleratorData, gyroscopeData, magnometerData):
        pass

    # Computes next position.
    def __nextPosition(self, timestamp, acceleratorData, magnometerData, gyroscopeData):
        self.sampleTimestamps.append(time.clock_gettime_ns(time.pthread_getcpuclockid(threading.get_ident())))
        self.sampleCount += 1
        self.accData.append(acceleratorData)
        self.gyroData.append(gyroscopeData)
        self.magnoData.append(magnometerData)

        heading = self.heading(self.accData, self.gyroData, self.magnoData)
        stepLength = self.__stepLength(timestamp)

        if stepLength > 0:
            self.accData = []
            self.gyroData = []
            self.magnoData = []

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

        avg_filter_cut = avg_filter[window[0]:window[1]]
        return self.k * math.pow(np.max(avg_filter_cut) - np.amin(avg_filter_cut), 1 / 4)   # Weinberg.

    # TODO: Finish this.
    # TODO: A window must start with the first 1 in a step and end with the last -1 in the same step.
    # Computes single signal window indices. If step is not complete, return None.
    def __step_window(self, signals):
        start = -1
        negative_met = False
        end = 0

        for i in range(len(signals)):
            if start == -1 and signals[i] == 1:
                start = i

            if start != -1 and signals[i] == -1:
                negative_met = True

            if start != -1 and negative_met and signals[i] != -1:
                return (start, i)

        return None

# Class for heading estimation using AHRS.
class AHRSPDR(BasePDR):
    def __init__(self, initial_location):
        super().__init__(initial_location)

    # Madgwick heading estimation. Units are in meters, not centimeters.
    def heading(self, acceleratorData, gyroscopeData, magnometerData):
        madgwick = Madgwick(gyr = np.array(gyroscopeData), acc = np.array(acceleratorData), frequency = self._sample_rate())
        quaternion = Quaternion(madgwick.Q[-1])
        return quaternion.to_axang()[1]

    # TODO: Use super().sampleTimestamps.
    # Computes sample rate.
    def _sample_rate(self):
        return 100
