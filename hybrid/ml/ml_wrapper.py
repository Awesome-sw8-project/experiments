from ml.lightgbm_wrapper import LightGBMWrapper

# Position estimation interface.
class Estimator:
    def next_position(self, data):
        pass

# Wrapper of ML algorithms used in this project.
class MLWrapper(Estimator):
    def __init__(self, algorithm_label):
        self.label = algorithm_label
        self.lightgbm_index = 1

    # Estimates next position.
    # Data might not contain appropriate RSSIs with values 0 or -999 depending on being normalized or not.
    # In this case, None is returned.
    # Argument data is a list of RSSI values.
    # Output is on form [<X>, <Y>, <Z>].
    def next_position(self, data):
        if self.label.lower() == "ann":
            return self.__ann_next_position(data)

        elif self.label.lower() == "rnn":
            return self.__rnn_next_position(data)

        elif self.label.lower() == "knn":
            return self.__knn_next_position(data)

        elif self.label.lower() == "lightgbm":
            return self.__lightgbm_next_position(data)

    # ANN next position estimation.
    def __ann_next_position(self, data):
        return self.algorithm.next_position()

    # RNN next position estimation.
    def __rnn_next_position(self, data):
        pass

    # KNN next position estimation.
    def __knn_next_position(self, data):
        pass

    # TODO: Handle when an estimation is not possible.
    # LightGBM next position estimation.
    def __lightgbm_next_position(self, data):
        return LightGBMWrapper.predict(data, self.lightgbm_index)

    # Sets model index for LightGBM.
    def set_model_index(self, index):
        self.lightgbm_index = index
