import math
import filter
import time
import threading
import numpy as np
import util

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
        self.currentGravity = 0
        self.alpha = 0.9
        self.windowSize = 10
        self.gravityFactors = []
        self.filteredValues = []
        self.sampleTimestamps = []
        self.sampleCount = 0

    # Getter to initial position.
    def getStartLocation(self):
        return self.start

    # Getter to current position.
    def getCurrentLocation(self, acceleratorData, magnometerData, gyroscopeData):
        __nextPosition(acceleratorData, magnometerData, gyroscopeData)
        return self.current

    # Polymorphic heading estimation.
    def heading(self, acceleratorData, magnometerData):
        pass

    # Computes next position.
    def __nextPosition(self, acceleratorData, magnometerData, gyroscopeData):
        self.sampleTimestamps.append(time.clock_gettime_ns(time.pthread_getcpuclockid(threading.get_ident())))
        self.sampleCount += 1
        heading = heading(acceleratorData, magnometerData)
        stepLength = __stepLength(__avgSampleRate, acceleratorData)
        heading = heading(acceleratorData, magnometerData)
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

    # Computes acceleration norm.
    def __acceleration_norm(self, x, y, z):
        return math.sqrt(math.pow(x, 2) + math.pow(y, 2) + math.pow(z, 2))

    # Computes gravity factor used in high-pass filtering to be removed from acceleration.
    def __gravityFactor(self, accelerationX, accelerationY, accelerationZ):
        self.currentGravity = self.alpha * self.currentGravity +
                                (1 - self.alpha) * __accelerationNorm(accelerationX, accelerationY, accelerationZ)
        self.gravityFactors.append(self.currentGravity)
        return self.currentGravity

    # TODO: Next -> step detection.
    # TODO: Consider trying without using filters. Just feed peak detection with raw acceleration data.
    # Step-length estimation.
    def __stepLength(self, sampleRate, acceleratorData):
        accelerationNorm = __acceleration_norm(acceleratorData[0], acceleratorData[1], acceleratorData[2])
        highPass = filter.Filter.highPassFilter(accelerationNorm, sampleRate, self.sampleCount)
        lowPass = filter.Filter.lowPassFilter(accelerationNorm, sampleRate, self.sampleCount)
        filterCombineRectangular = util.rectangular(np.convolve(highPass, lowPass))
        self.filteredValues.append(filterCombineRectangular))

# Class for heading estimation using AHRS.
class AHRSPDR(BasePDR):
    def __init__(self):
        pass

    def heading(self, acceleratorData, magnometerData):
        pass
