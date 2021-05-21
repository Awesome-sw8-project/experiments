import math

# Container of positioning error calculations.

# Position error.
# First argument is estimation list of 3 elements: x, y and floor.
# Second argument is ground truth list of 3 elements: x, y and floor.
def pe(estimation, ground_truth):
    return math.sqrt(math.pow(estimation[0] - ground_truth[0], 2) + math.pow(estimation[1] - ground_truth[1], 2)) \
             + 15 * math.fabs(estimation[2] -  ground_truth[2])

# Mean positopning error.
# First argument is a list of estimation lists of 3 elements. Inner-most list contains x, y and floor.
# Second argument is a list of ground truth lists of 3 elements. Inner-most list contains x, y and floor.
def mpe(estimations, ground_truths):
    if len(estimations) != len(ground_truths):
       raise ValueError("Arguments are not of the same size.")

    sum = 0

    for i in range(len(estimations)):
        sum += pe(estimations[i], ground_truths[i])

    return sum / len(estimations)

# Root Mean Squared Error (RMSE).
# Returns a 3-tuples, one tuple element for each coordinate.
def rmse(estimations, ground_truths):
    if len(estimations) != len(ground_truths):
        raise ValueError("Arguments are not of the same size.")

    sum_x = sum_y = sum_z = 0

    for i in range(len(estimations)):
        sum_x += math.sqrt(math.pow(ground_truths[i][0] - estimations[i][0], 2))
        sum_y += math.sqrt(math.pow(ground_truths[i][1] - estimations[i][1], 2))
        sum_z += math.sqrt(math.pow(ground_truths[i][2] - estimations[i][2], 2))

    return sum_x / len(estimations), sum_y / len(estimations), sum_z / len(estimations)
