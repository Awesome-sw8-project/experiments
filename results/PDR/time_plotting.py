import numpy as np
import pandas as pd
import csv

# Composes all .csv files into one .csv file.
# Ignores headers assuming all files have a header.
# It also assumes comma-separation.
def compose_csv(header, dir, out_name):
    files = [p for p in os.listdir(dir) if p.endswith(".csv")]

    with open(out_name,  "w", newline = '') as out_file:
        writer = csv.writer(out_file)
        writer.writerow(header)

        for file in files:
            with open(file, "r") as in_file:
                reader = csv.reader(in_file, delimiter = ',')
                start = True

                for i, line in enumerate(reader):
                    if start:
                        start = False
                        continue

                    writer.writerow(line.split(','))

