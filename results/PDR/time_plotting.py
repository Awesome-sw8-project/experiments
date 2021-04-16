import numpy as np
import pandas as pd
import csv

# Composes all .csv files into one .csv file.
# Ignores headers.
def compose_csv(header, dir, out_name):
    files = [p for p in os.listdir(dir) if p.endswith(".csv")]

    with open(out_name,  "w", newline = '') as out_file:
        writer = csv.writer(out_file)
        writer.writerow(header)

        for file in files:
            with open(file, "r") as in_file:
                
