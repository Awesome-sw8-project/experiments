#from pdr.pdr import BasePDR, AHRSPDR, Location
import sys
sys.path.append("../../../..")
import datapipeline.datapipeline as dp
import evaluation.evaluator as evaluator, evaluation.eval_output as eo

iterator = dp.imu_data("../../../../../data/data/train/")
pdr = None
start = True

for data in iterator:
    pdr_measurements = list()
    ground_truth = list()

    for measure in data[1]:
        if start:
            pdr = AHRSPDR(Location(measure[1][0], measure[1][0]), heading_type = "madgwick")
            start = False
            continue

        next_position = pdr.get_current_location(measure[0], measure[2], measure[3], measure[4])
        pdr_measurement.append([next_position[0], next_position[1], measure[5][2]])
        ground_truth.append(measure[5])

    eval = evaluator.Evaluator(pdr_measurement, ground_truth)
    eo.write(eval, "../../../../results/" + data[0])
