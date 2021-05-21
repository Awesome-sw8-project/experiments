import evaluation.eval_math as em

class Evaluator:
    # Arguments are file names.
    def __init__(self, evaluation_data, actual):
        if len(evaluation_data) != len(actual):
            raise ValueError("Arguments are not of the same size.")

        self.evaluation_data = evaluation_data
        self.ground_truth_data = actual
        self.mpe = 0
        self.pe = []
        self.rmse = (0, 0, 0)
        self.__eval_results()

    # Entry method for evaluation results.
    def __eval_results(self):
        self.__eval_positions()
        self.__mpe()
        self.__rmse()

    # Evaluates each estimated position.
    def __eval_positions(self):
        for i in range(len(self.evaluation_data)):
            self.pe.append(em.pe(self.evaluation_data[i], self.ground_truth_data[i]))

    # Computes mean positioning error.
    def __mpe(self):
        self.mpe = em.mpe(self.evaluation_data, self.ground_truth_data)

    # Computes RMSE.
    def __rmse(self):
        self.rmse = em.rmse(self.evaluation_data, self.ground_truth_data)

    # Getter to mean positioning error.
    def get_mpe(self):
        return self.mpe

    # Getter to positioning error for each estimation.
    def get_pe(self):
        return self.pe

    # Getter to RMSE.
    def get_rmse(self):
        return self.rmse
