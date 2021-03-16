import math

# Container of positioning error calculations.
class Eval:
    # Position error.
    # First argument is estimation list of 3 elements: x, y and floor.
    # Second argument is ground truth list of 3 elements: x, y and floor.
    @staticmethod
    def pe(estimation, ground_truth):
        return math.sqrt(math.pow(estimation[0] - ground_truth[0], 2) + math.pow(estimation[1] - ground_truth[1], 2)) \
                 + 15 * math.fabs(estimation[2] -  ground_truth[2])

    # Mean positopning error.
    # First argument is a list of estimation lists of 3 elements. Inner-most list contains x, y and floor.
    # Second argument is a list of ground truth lists of 3 elements. Inner-most list contains x, y and floor.
    @staticmethod
    def mpe(estimations, ground_truths):
        if len(estimations) != len(ground_truths):
            raise ValueError("Arguments are not of the same size.")

        sum = 0

        for i in range(len(estimations)):
            sum += Eval.pe(estimations[i], ground_truths[i])

        return sum / len(estimations)