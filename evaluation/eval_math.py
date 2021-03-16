import math

# Container of positioning error calculations.
class Eval:
    # Positioning error.
    @staticmethod
    def pe(x_pred, y_pred, x_act, y_act, floor_pred, floor_act):
        return math.sqrt(math.pow(x_pred - x_act, 2) + math.pow(y_pred, y_act, 2)) + 15 * math.fabs(floor_pred - floor_act)

    # Mean positioning error.
    @staticmethod
    def mpe(x_preds, y_preds, x_acts, y_acts, floor_preds, floor_acts):
        if Eval.__check_sizes(x_preds, y_preds, x_acts, y_acts, floor_preds, floor_acts)
            raise ValueError("Arguments are not of the same size.")

        sum = 0

        for i in range(len(x_preds)):
            sum += Eval.pe(x_preds[i], y_preds[i], x_acts[i], y_acts[i], floor_preds[i], floor_acts[i])

        return sum / len(x_preds)

    # Checks MPE list values are of the same size.
    @staticmethod
    def __check_sizes(x_preds, y_preds, x_acts, y_acts, floor_preds, floor_acts):
        s = len(x_preds)

        if len(y_preds) != s or len(x_acts) != s or len(y_acts) != s or len(floor_preds) != s or len(floor_acts) != s:
            return False

        return True
