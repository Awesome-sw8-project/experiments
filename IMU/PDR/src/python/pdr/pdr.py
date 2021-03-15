import math
import time
import threading
import numpy as np
from pdr import filter
from pdr import util
from pdr import step
from ahrs.filters import Madgwick, Mahony
from ahrs.common import Quaternion

class Location:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

# Base class for PDR estimation with unspecified heading estimation.
class BasePDR:
    def __init__(self, start_location):
        self.start = start_location
        self.current = start_location
        self.k = 0.51
        self.sample_timestamps = []
        self.sample_count = 0
        self.acc_data = []
        self.signals = []
        self.gyro_data = []
        self.magno_data = []

    # Getter to timestamps for derived classes.
    def get_timestamps(self):
        return self.sample_timestamps

    # Getter to initial position.
    def get_start_location(self):
        return self.start

    # Getter to current position.
    def get_current_location(self, timestamp, accelerator_data, magnometer_data, gyroscope_data):
        self.__next_position(timestamp, accelerator_data, magnometer_data, gyroscope_data)
        return self.current

    # Polymorphic heading estimation.
    def heading(self, accelerator_data, gyroscope_data, magnometer_data):
        pass

    # Computes next position.
    def __next_position(self, timestamp, accelerator_data, magnometer_data, gyroscope_data):
#        self.sampleTimestamps.append(time.clock_gettime_ns(time.pthread_getcpuclockid(threading.get_ident())))
        self.sample_timestamps.append(timestamp)
        self.sample_count += 1
        self.acc_data.append(accelerator_data)
        self.gyro_data.append(gyroscope_data)
        self.magno_data.append(magnometer_data)

        heading = self.heading(self.acc_data, self.gyro_data, self.magno_data)
        step_length = self.__step_length(timestamp)

        if step_length > 0:
            self.acc_data = []
            self.gyro_data = []
            self.magno_data = []

        x = self.current.get_x() + step_length * math.cos(heading)
        y = self.current.get_y() + step_length * math.sin(heading)
        self.current = Location(x, y)

    # Computes average sample rate in kHz.
    def __avg_sample_rate(self):
        times = []
        i = 1

        while i < len(self.sample_timestamps):
            times.append(self.sample_timestamps[i] - self.sample_timestamps[i - 1])
            i += 1

        return np.sum(np.array(times)) / len(times)

    # Step-length estimation.
    def __step_length(self, timestamp):
        lag = 2

        if len(self.acc_data) < lag:
            return 0

        peak_detection = step.Step.peak_detection(self.acc_data, 5, lag = lag)
        avg_filter = peak_detection['avgFilter']
        window = self.__step_window(peak_detection['signals'])

        if window == None:
            return 0

        avg_filter_cut = avg_filter[window[0]:window[1]]
        return self.k * math.pow(np.max(avg_filter_cut) - np.amin(avg_filter_cut), 1 / 4)   # Weinberg.

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
    def __init__(self, initial_location, use_madgwick):
        super().__init__(initial_location)
        self.use_madgwick = use_madgwick

    # Madgwick heading estimation. Units are in meters, not centimeters.
    def heading(self, accelerator_data, gyroscope_data, magnometer_data):
        orientation = None

        if self.use_madgwick:
            orientation = Madgwick(gyr = np.array(gyroscope_data), acc = np.array(accelerator_data), frequency = self._sample_rate())

        else:
            orientation = Mahony(gyr = np.array(gyroscope_data), acc = np.array(accelerator_data), frequency = self._sample_rate())

        quaternion = Quaternion(orientation.Q[-1])
        return quaternion.to_axang()[1]

    # Heading estimation by Mahony.
    def mahony_heading(self, accelerator_data, gyroscope_data, magnometer_data):
        mahony = Mahony(gyr = np.array(gyroscope_data), acc = np.array(accelerator_data), frequency = self._sample_rate())
        quaternion = Quaternion(mahony.Q[-1])
        return quaternion.to_axang()[1]

    # Computes sample rate.
    def _sample_rate(self):
        if len(BasePDR.get_timestamps(self)) < 2:
            return 100

        time_period = (BasePDR.get_timestamps(self)[-1] - BasePDR.get_timestamps(self)[0]) / 1000
        return len(BasePDR.get_timestamps(self)) / time_period
