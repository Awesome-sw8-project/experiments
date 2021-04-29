# Position estimation interface.
class Estimator:
    def next_position(self, data):
        pass

# Wrapper of ML algorithms used in this project.
class MLWrapper(Estimator):
    def __init__(self, ml_algorithm, algorithm_label):
        self.algorithm = ml_algorithm
        self.label = algorithm_label

    # Estimates next position.
    def next_position(self, data):
        if self.algorithm.lower() == "ann":
            return self.__ann_next_position(data)

        elif self.algorithm.lower() == "rnn"
            return self.__rnn_next_position(data)

        elif self.algorithm.lower() == "knn":
            return self.__knn_next_position(data)

        elif self.algorithm.lower() == "lightgbm":
            return self.__lightgbm_next_position(data)

    # ANN next position estimation.
    def __ann_next_position(self, data):
        pass

    # RNN next position estimation.
    def __rnn_next_position(self, data):
        pass

    # KNN next position estimation.
    def __knn_next_position(self, data):
        pass

    # LightGBM next position estimation.
    def __lightgbm_next_position(self, data):
        pass