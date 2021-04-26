# Retrieves one path from each site.

import os

# Returns map of file name to list of unique site file results.
# List of prefixes determines the result folders.
def site_files(prefixes):
    filepath = ""
    map = {}

    if len(prefixes) < 1:
        filepath = "./"

    else:
        filepath = prefixes[0] + "/"

    files = [p for p in os.listdir(filepath) if p.endswith(".txt")]
    files = __unique_site_files(files)

    # Find unique sites for each prefix.
    for prefix in prefixes:
        map[prefix] = list()

        for file in files:
            map[prefix].append(prefix + "/" + file)

    return map

# Returns list of paths, one for each site.
def __unique_site_files(files):
    return list(set(files))

# Returns site ID from file name.
def site_id(filename):
    return filename[0:24]
