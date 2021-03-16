import eval_math

class Evaluator:
    # Arguments are file names.
    def __init__(self, evaluation_data, result):
        self.evaluation_data = evaluation_data
        self.result = result

    # Entry method for evaluation results.
    def __eval_results(self):
        self.__eval_positions()
        self.__mpe()

    # Evaluates each estimated position.
    def __eval_positions(self):
        pass

    # Computes mean positioning error.
    def __mpe(self):
        pass

    # Getter to mean positioning error.
    def get_mpe(self):
        return None

    # Getter to positioning error for each waypoint.
    # Return value is map from waypoint to positioning error.
    def get_pe(self):
        return None
