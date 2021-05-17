import csv
import sys
sys.path.append("../../../../..")
import datapipeline.datapipeline as dp

num_count = 0
num_files = 10850
iterator = dp.imu_data("../../../../../../data/data/train/", "../../../../../../data/data/sample_submission.csv", 0)
prev_time = None

for data in iterator:
    print("Progress: " + str((num_count / num_files) * 100) + "%", end = "\r", flush = True)
    num_count += 1
    i = 0

    with open("../../../../../results/PDR/timing/" + data[0][0:len(data[0]) - 4] + ".csv", "w", newline = '') as file:
        writer = csv.writer(file)
        prev_time = data[1][0]
        writer.writerow(["i", "Time difference"])

        for measure in data[1]:
            timestamp = measure[0]

            if i == 0:
                prev_time = timestamp
                writer.writerow([str(i), "0"])
                i += 1
                continue

            writer.writerow([str(i), str(float((timestamp - prev_time)) / 1000)])
            prev_time = timestamp
            i += 1
