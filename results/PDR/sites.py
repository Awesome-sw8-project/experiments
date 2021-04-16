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
    result = list()

    for file in files:
        if not __has_site(file, result):
            result.append(file)

    return result

# Returns site ID from file name.
def site_id(filename):
    return filename[0:24]

# Checks whether site is already present in list of files.
def __has_site(site_file, site_files):
    for file in site_files:
        if site_id(site_file) == site_id(file):
            return True

    return False
