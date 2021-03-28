from pdr.pdr import BasePDR, AHRSPDR, Location
import sys
sys.path.append("../../../..")
import datapipeline.datapipeline as dp
import evaluation.evaluator as evaluator, evaluation.eval_output as eo

pe_sum_mad = 0
pe_sum_mah = 0
pe_sum_ekf = 0
iterations = 1
iterator = dp.imu_data("../../../../../data/data/train/", "../../../../../data/data/sample_submission.csv")
pdr_mad = None
pdr_mah = None
pdr_ekf = None
path_count = 1

for data in iterator:
    i = 0
    pdr_measurements_mad = list()
    pdr_measurements_ekf = list()
    ground_truth = list()
    print("Path count: " + str(path_count))
    path_count += 1

    for measure in data[1]:
        if i == 0:
            pdr_mad = AHRSPDR(Location(measure[1][0], measure[1][1]), heading_type = "madgwick")
            pdr_ekf = AHRSPDR(Location(measure[1][0], measure[1][1]), heading_type = "kalman")
            i += 1
            continue

        next_position_mad = pdr_mad.get_current_location(measure[0], measure[2], measure[3], measure[4])
        next_position_ekf = pdr_ekf.get_current_location(measure[0], measure[2], measure[3], measure[4])
        pdr_measurements_mad.append([next_position_mad.get_x(), next_position_mad.get_y(), measure[5][2]])
        pdr_measurements_ekf.append([next_position_ekf.get_x(), next_position_ekf.get_y(), measure[5][2]])
        ground_truth.append(measure[5])
        i += 1
        iterations += 1

    eval_mad = evaluator.Evaluator(pdr_measurements_mad, ground_truth)
    eval_ekf = evaluator.Evaluator(pdr_measurements_ekf, ground_truth)
    eo.write(eval_mad, "../../../../results/PDR/madgwick/" + data[0])
    eo.write(eval_ekf, "../../../../results/PDR/e_kalman/" + data[0])

    pe_sum_mad += sum(eval_mad.get_pe())
    pe_sum_ekf += sum(eval_ekf.get_pe())
    print(str(pe_sum_ekf / iterations))
    print("Madgwick MPE: " + str(eval_mad.get_mpe()))
    print("E-Kalman MPE: " + str(eval_ekf.get_mpe()))
    print("Measurements: " + str(i))
    print()

eo.write_mpe_avg("../../../../results/PDR/madgwick/", pe_sum_mad / iterations)
eo.write_mpe_avg("../../../../results/PDR/e_kalman", pe_sum_ekf / iterations)
