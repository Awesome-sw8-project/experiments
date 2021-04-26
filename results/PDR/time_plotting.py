import numpy as np
import pandas as pd
import csv
import os
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
import plot.cdf as c

# Enables using multi-character delimiter.
def multi_delimiter(delimiter):
    if (len(delimiter) == 1):
        return delimiter

    return delimiter[-1]

# Composes all .csv files into one .csv file.
# Ignores headers assuming all files have a header.
# It also assumes comma-separation and exactly two columns.
def compose_csv(header, dir, file_type, delimiter, out_name, log_progress, skip_lines):
    files = [p for p in os.listdir(dir) if p.endswith("." + file_type)]
    count = 0

    with open(out_name,  "w", newline = '') as out_file:
        writer = csv.writer(out_file)
        writer.writerow(header)

        for file in files:
            count += 1

            if log_progress:
                print("Composing into " + out_name + ": " + str(int((count / len(files)) * 100)) + "%", end = '\r', flush = True)

            with open(dir + "/" + file, "r") as in_file:
                reader = csv.reader(in_file, delimiter = multi_delimiter(delimiter))
                
                for i, line in enumerate(reader):
                    if i > skip_lines:
                        writer.writerow([str(line[0]), str(line[-1])])

if __name__ == "__main__":
    compose_csv(["i", "Time difference"], "timing", "csv", ',', "composed.csv", True)
    compose_csv(["Measurement", "PE"], "madgwick/", "txt", '\t\t', "madgwick_pes.csv", True, 3)
    compose_csv(["Measurement", "PE"], "kalman/", "txt", '\t\t', "kalman_pes.csv", True, 3)

    timings = pd.read_csv("composed.csv")
    plt.boxplot(list(timings['Time difference']), labels = ['Time difference'], sym = '', autorange = True, patch_artist = True)
    plt.savefig("boxplot.png")

    madgwick_pes = pd.read_csv("madgwick_pes.csv")
    kalman_pes = pd.read_csv("kalman_pes.csv")
    madgwick_pe_list = list(madgwick_pes['PE'])
    kalman_pe_list = list(kalman_pes['PE'])
    c.plot(madgwick_pe_list, len(madgwick_pe_list), "Cumulative Distribution Function of Positioning Errors (Madgwick)", "Positioning error (M)")
    c.plot(kalman_pe_list, len(kalman_pe_list), "Cumulative Distribution Function of Positioning Errors (Extended Kalman)", "Positioning error (M)")

    os.remove("composed.csv")
    os.remove("madgwick_pes.csv")
    os.remove("kalman_pes.csv")
