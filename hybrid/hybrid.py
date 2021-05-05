import ml.ml_wrapper as ml
import sys
sys.path.append("../IMU/PDR/src/python")
import pdr.pdr as p

# Base class of hybrid position estimators.
class Hybrid(ml.Estimator):
    def __init__(self, start_location, ml_algorithm, algorithm_label):
        self.heading_type = "madgwick"
        self.pdr = p.AHRSPDR(start_location, heading_type = self.heading_type)
        self.ml = ml.MLWrapper(ml_algorithm, algorithm_label)

    # Re-calibrates PDR.
    def pdr_recalibrate(self, location):
        self.pdr = p.AHRSPDR(location, self.heading_type)

    # Abstract method for position estimation.
    # imu_data is on form: [<TIMESTAMP>, [<ACC_DATA>], [<GYR_DATA>], [MAG_DATA]].
    # antenna_dat is on form: [[<BSSID>, [{<RSSI_VALUES>}]]].
    def next_position(self, imu_data, antenna_data):
        pass

# Hybrid estimator using ML as primary and PDR as support.
class MLPDRHybrid(Hybrid):
    def __init__(self, start_location, ml_algorithm, algorithm_label):
        super().__init__(start_location, ml_algorithm, algorithm_label)
        self.last_position = start_location
        self.using_pdr = False

    # TODO: Maybe we should let PDR run for a minimum number of measurements.
    # TODO: Check BSSID is usable by the SA algorithms (in ML wrapper).
    # Main entry for estimating position.
    def next_position(self, imu_data, antenna_data):
        ml_pos = self.ml.next_position(antenna_data)

        if (ml_pos == None):
            if (not self.using_pdr):
                self.pdr_realibrate(self.last_position[0], self.last_position[1])

            self.using_pdr = True
            imu_pos = self.pdr.get_current_location(imu_data[0], imu_data[1], imu_data[2], imu_data[3])
            self.last_position = [imu_pos.get_x(), imu_pos.get_y(), self.last_position[2]]

            return self.last_position

        self.last_position = ml_pos
        return ml_pos

# Hybrid estimator using PDR as primary and ML as support.
class PDRMLHybrid(Hybrid):
    def __init__(self):
        pass

    # Main entry for estimating position.
    def next_position(self, imu_data, antenna_data):
        pass

# Hybrid estimator using both PDR and ML for averaging estimations.
class AverageHybrid(Hybrid):
    def __init__(self, start_location, ml_algorithm, algorithm_label, recalibrate_limit):
        super().__init__(start_location, ml_algorithm, algorithm_label)
        self.last_floor = 0
        self.recal_limit = recalibrate_limit
        self.estimate_count = 0

    # Main entry for estimating position.
    # Outout is on form [<X>, <Y>, <Z>].
    def next_position(self, imu_data, antenna_data):
        self.estimate_count += 1
        ml_pos = self.ml.next_position(antenna_data)
        pdr_pos = self.pdr.get_current_location(imu_data[0], imu_data[1], imu_data[2], imu_data[3])
        avg_x = pdr_pos.get_x()
        avg_y = pdr_pos.get_y()

        if (ml_pos != None):
            avg_x += ml_pos[0]
            avg_x = avg_x / 2
            avg_y += ml_pos[1]
            avg_y = avg_y / 2
            self.last_floor = ml_pos[2]

            if (self.estimate_count > self.recal_limit):
                self.pdr_recalibrate(p.Location(ml_pos[0], ml_pos[1]))
                self.estimate_count = 0

        return [avg_x, avg_y, self.last_floor]
