import ml.ml_wrapper as ml
import pickle
import os
import sys
sys.path.append("../IMU/PDR/src/python")
import pdr.pdr as p
sys.path.append("..")
import evaluation.evaluator as evaluator, evaluation.eval_output as eo
sys.path.append("..")
import datapipeline.datapipeline as pipe

model_map = {
    "5da138754db8ce0c98bca82f": 1,
    "5da138274db8ce0c98bbd3d2": 2,
    "5d2709d403f801723c32bd39": 3,
    "5dc8cea7659e181adb076a3f": 4,
    "5d2709bb03f801723c32852c": 5
}

# Loads .pickle files with train data.
def gen_for_serialisation(path_to_train):
    site_data_files = [x for x in os.listdir(path_to_train)]
    for file in site_data_files:
        f = open(path_to_train +'/'+file, "rb")
        site, site_data = pickle.load(f)
        f.close()
        yield site, site_data

# Base class of hybrid position estimators.
class Hybrid(ml.Estimator):
    def __init__(self, start_location, algorithm_label):
        self.heading_type = "madgwick"
        self.pdr = p.AHRSPDR(p.Location(start_location[0], start_location[1]), heading_type = self.heading_type)
        self.ml = ml.MLWrapper(algorithm_label)

    # Re-calibrates PDR.
    def pdr_recalibrate(self, location):
        self.pdr = p.AHRSPDR(location, self.heading_type)

    # Abstract method for position estimation.
    # imu_data is on form: [<TIMESTAMP>, [<ACC_DATA>], [<GYR_DATA>], [MAG_DATA]].
    # antenna_dat is on form: [{<BSSID>, [{<RSSI_VALUES>}]}].
    def next_position(self, imu_data, antenna_data):
        pass

# Hybrid estimator using ML as primary and PDR as support.
class MLPDRHybrid(Hybrid):
    def __init__(self, start_location, algorithm_label):
        super().__init__(start_location, algorithm_label)
        self.last_position = start_location
        self.using_pdr = False
        self.time_limit = 1000
        self.prev_timestamp = 0

    # Main entry for estimating position.
    def next_position(self, imu_data, antenna_data):
        pos = self.ml.next_position(antenna_data)
        timestamp = imu_data[0]

        if (timestamp - self.prev_timestamp > self.time_limit or antenna_data == []):
            if (not self.using_pdr):
                self.pdr_recalibrate(p.Location(self.last_position[0], self.last_position[1]))
                self.using_pdr = True

            imu_pos = self.pdr.get_current_location(imu_data[0], imu_data[1], imu_data[2], imu_data[3])
            pos = [imu_pos.get_x(), imu_pos.get_y(), self.last_position[2]]

        self.last_position = pos

        if (len(antenna_data) > 0):
            if (timestamp - self.prev_timestamp <= self.time_limit):
                self.using_pdr = False

            self.prev_timestamp = timestamp

        return pos

    # Sets models index for LightGBM.
    def set_model_index(self, index):
        self.ml.set_model_index(index)

# Hybrid estimator using PDR as primary and ML as support.
class PDRMLHybrid(Hybrid):
    def __init__(self, start_location, algorithm_label):
        super().__init__(start_location, algorithm_label)
        self.pdr_measurement_count = 0
        self.recal_limit = 350
        self.latest_ml_pos = [0, 0, 0]

    # Main entry for estimating position.
    # Also estimates position using ML and stores it in class member, since we might not always
    # get a position estimation from the ML algorithm.
    def next_position(self, imu_data, antenna_data):
        pdr_pos = self.pdr.get_current_location(imu_data[0], imu_data[1], imu_data[2], imu_data[3])
        ml_pos = self.ml.next_position(antenna_data)
        self.pdr_measurement_count += 1

        if (ml_pos != None):
            self.latest_ml_pos = ml_pos

        if (self.measurement_count >= self.recal_limit):
            self.pdr_recalibrate(self.latest_ml_pos)

        return [pdr_pos.get_x(), pdr_pos.get_y(), self.latest_ml_pos[2]]

    # Sets models index for LightGBM.
    def set_model_index(self, index):
        self.ml.set_model_index(index)

# Hybrid estimator using both PDR and ML for averaging estimations.
class AverageHybrid(Hybrid):
    def __init__(self, start_location, algorithm_label, recalibrate_limit):
        super().__init__(start_location, algorithm_label)
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

    # Sets models index for LightGBM.
    def set_model_index(self, index):
        self.ml.set_model_index(index)

if __name__ == "__main__":
    iter = gen_for_serialisation("../../data/data/filtered")
    site_count = 0

    for site in iter:
        if site_count < 0:
            site_count += 1
            continue

        ground_truths = list()
        estimations = list()

        for path_data in site[1]:
            time_mapping = path_data[2]
            keys = list(time_mapping.keys())
            hybrid = MLPDRHybrid(time_mapping[keys[0]][0], "lightgbm")
            hybrid.set_model_index(model_map[site[0]])

            for timestamp in keys:
                if (time_mapping[timestamp][0] == []):      # In case we don't have ground truth.
                    continue

                ground_truths.append(time_mapping[timestamp][0])
                accelerator = time_mapping[timestamp][1][0]
                gyroscope = time_mapping[timestamp][1][2]
                magnometer = time_mapping[timestamp][1][1]
                rssi = None

                if (len(time_mapping[timestamp]) < 3):
                    rssi = []

                else:
                    rssi = time_mapping[timestamp][2]

                pos = hybrid.next_position([int(timestamp), accelerator, magnometer, gyroscope], rssi)
                estimations.append(pos)

        site_count += 1
        eval = evaluator.Evaluator(estimations, ground_truths)
        print("MPE (site " + str(site_count) + "): " + str(eval.get_mpe()))
        print("RMSE (site " + str(site_count) + "): " + str(eval.get_rmse()) + "\n")
        eo.write(eval, "../results/hybrid/average/" + site[0] + ".txt")
