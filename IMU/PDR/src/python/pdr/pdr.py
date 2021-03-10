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
        self.current = None

# Class for heading estimation using AHRS.
class AHRSPDR(BasePDR):
    def __init__(self):
        pass

    def heading(self, acceleratorData, magnometerData):
        pass
