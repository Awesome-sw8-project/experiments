import numpy as np
import pandas as pd
import csv
import os

# Composes all .csv files into one .csv file.
# Ignores headers assuming all files have a header.
# It also assumes comma-separation.
def compose_csv(header, dir, out_name, log_progress):
    files = [p for p in os.listdir(dir) if p.endswith(".csv")]
    count = 0

    with open(out_name,  "w", newline = '') as out_file:
        writer = csv.writer(out_file)
        writer.writerow(header)

        for file in files:
            count += 1

            if log_progress:
                print("Composing " + str(int((count / len(files)) * 100)) + "%...", end = '\r', flush = True)

            with open(dir + "/" + file, "r") as in_file:
                reader = csv.reader(in_file, delimiter = ',')
                start = True

                for i, line in enumerate(reader):
                    if start:
                        start = False
                        continue

                    writer.writerow(line)

if __name__ == "__main__":
    compose_csv(["i", "Time difference"], "timing", "composed.csv", True)
    timings = pd.read_csv("composed.csv")
    boxplot = timings.boxplot(column = ["Time difference"])
