import sites
import csv

# First parameter is ID of site, and second is a list of PEs (that is, a 2D list).
def write_results(id, results, prefixes):
    with open(id + ".csv", "w", newline = '') as file:
        writer = csv.writer(file)
        writer.writerow(["Measurement"] + prefixes)

        for i in range(len(results[0])):
            writer.writerow([i + 1] + __result_row(results, i))

# Returns results row from index.
def __result_row(results, index):
    row = list()

    for i in range(len(results)):
        row.append(results[i][index])

    return row

# Reads array of PEs for a single prefix.
# Returns map from site ID to list of PEs.
def read_results(files):
    result = {}

    for file in files:
        site_id = __site_id(file)
        result[site_id] = list()
        i = 0

        with open(file, "r") as res:
            lines = res.readlines()

            for line in lines:
                if i < 3:
                    i += 1
                    continue

                result[site_id].append(line.split("\t\t\t")[1].rstrip('\n'))

    return result

# Returns site ID from file path.
def __site_id(filepath):
    return filepath[filepath.find('/') + 1:filepath.find('_')]

if __name__ == "__main__":
    files = sites.site_files(["madgwick", "kalman"])
    madgwick_res = read_results(files.get("madgwick"))
    e_kalman_res = read_results(files.get("kalman"))

    if (len(files.get("madgwick")) != len(files.get("kalman"))):
        raise ValueError("Difference in number of files.")

    for i in range(len(e_kalman_res.keys())):
        site_id = list(e_kalman_res.keys())[i]
        write_results(site_id, [madgwick_res.get(site_id), e_kalman_res.get(site_id)], ["madgwick", "e_kalman"])
