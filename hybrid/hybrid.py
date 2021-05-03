import ml.ml_wrapper as ml
import sys
sys.path.append("../IMU/PDR/src/python")
import pdr.pdr as p
sys.path.append("..")
import datapipeline.datapipeline as dp

# Temporary.
sys.path.append("../ML")
import models as m

# Base class of hybrid position estimators.
class Hybrid(ml.Estimator):
    def __init__(self, start_location, ml_algorithm, algorithm_label):
        self.heading_type = "madgwick"
        self.pdr = p.AHRSPDR(start_location, heading_type = self.heading_type)
        self.ml = ml.MLWrapper(ml_algorithm, algorithm_label)

    # Re-calibrates PDR.
    def __pdr_recalibrate(self, location):
        self.pdr = p.AHRSPDR(location, self.heading_type)

    # Abstract method for position estimation.
    # imu_data is on form: [<TIMESTAMP>, [<ACC_DATA>], [<GYR_DATA>], [MAG_DATA]].
    # antenna_dat is on form: [[<BSSID>, [{<RSSI_VALUES>}]]].
    def next_position(self, imu_data, antenna_data):
        pass

# Hybrid estimator using ML as primary and PDR as support.
class MLPDRHybrid(Hybrid):
    def __init__(self):
        pass

    # Main entry for estimating position.
    def next_position(self, imu_data, antenna_data):
        pass

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

    # def get_current_location(self, timestamp, accelerator_data, magnometer_data, gyroscope_data):
    # Main entry for estimating position.
    # Outout is on form [<X>, <Y>, <Z>].
    def next_position(self, imu_data, antenna_data):
        self.estimate_count += 1
        ml_pos = super().ml.next_position(antenna_data)
        pdr_pos = super().pdr.get_current_location(imu_data[0], imu_data[1], imu_data[3], imu_data[2])
        avg_x = pdr_pos.get_x()
        avg_y = pdr_pos.get_y()

        if (ml_pos != None):
            avg_x += ml_pos[0]
            avg_x = avg_x / 2
            avg_y += ml_pos[1]
            avg_y = avg_y / 2
            self.last_floor = ml_pos[2]

            if (self.estimate_count > self.recal_limit):
                super().pdr = p.AHRSPDR(Location(ml_pos[0], ml_pos[1]), heading_type = super().heading_type)

        return [avg_x, avg_y, self.last_floor]


model = m.model02.create_model(5)
ml_est = []
imu_est = []
rssi_iter = rssi_features("TYPE_WIFI", "../../data/data/train", "../../data/data/sample_submission.csv", "./")
imu_iter = imu_data("../../data/data/train")
count = 0

for est in rssi_iter:
    if (count >= 10):
        break

    count += 1
    ml_est.append(est)

count = 0

for est in imu_iter:
    if (count >= 10):
        break

    count += 1
    imu_est.append(est)