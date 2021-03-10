import math

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
        self.lpf = []
        self.hpf = []

    # Getter to initial position.
    def getStartLocation(self):
        return self.start

    # Getter to current position.
    def getCurrentLocation(self, acceleratorData, magnometerData):
        __nextPosition()
        return self.current

    # Polymorphic heading estimation.
    def heading(self, acceleratorData, magnometerData):
        pass

    # Computes next position.
    def __nextPosition(self, acceleratorData, magnometerData, gyroscopeData):
        heading = heading(acceleratorData, magnometerData)
        stepLength = __stepLength()
        heading = heading(acceleratorData, magnometerData)
        x = self.current.getX() + stepLength * math.cos(heading)
        y = self.current.getY() + stepLength * math.sin(heading)
        self.current = Location(x, y)

    # Computes acceleration norm.
    def __acceleration_norm(self, x, y, z):
        return math.sqrt(math.pow(x, 2) + math.pow(y, 2) + math.pow(z, 2))

    # Computes gravity factor used in high-pass filtering to be removed from acceleration.
    def __gravityFactor(self, accelerationX, accelerationY, accelerationZ):
        self.currentGravity = self.alpha * self.currentGravity + (1 - self.alpha) * __accelerationNorm(accelerationX, accelerationY, accelerationZ)
        self.gravityFactors.append(self.currentGravity)
        return self.currentGravity

    # High-pass filter.
    def __hpFilter(self, accelerationX, accelerationY, accelerationZ):
        filtering = __accelerationNorm(accelerationX, accelerationY, accelerationZ) - __gravityFactor(accelerationX, accelerationY, accelerationZ)
        self.hpf.append(filtering)
        return filtering

    # TODO: Paper describes this step poorly:
    # TODO: https://ieeexplore-ieee-org.zorac.aub.aau.dk/document/8250048
    # TODO: We try only storing a single low-pass filter value.
    # Low-pass filter.
    def __lpFilter(self, accelerationX, accelerationY, accelerationZ):
        filtering = __hpFilter(accelerationX, accelerationY, accelerationZ)
        self.lpf.append(filtering)
        return filtering

    # Step-length estimation.
    def __stepLength(self):
        pass

# Class for heading estimation using AHRS.
class AHRSPDR(BasePDR):
    def __init__(self):
        pass

    def heading(self, acceleratorData, magnometerData):
        pass
