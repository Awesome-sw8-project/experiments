import pandas as pd, os, json, gc, numpy as np, pickle


def filter_files(files, sites):
    return [p for p in files if p.split("_")[0] in sites]

#get waypoint with lowest timestamp in the list waypoints
def lowest_waypoint(waypoints):
    timestamp = None
    x = y = None
    for e,k in enumerate(waypoints):
        if timestamp == None or int(k[0])<timestamp:
            timestamp = int(k[0])
            x,y = float(k[2]), float(k[3])
    return x,y

#output format of a next call to imu_data
#[filename, [
#               [timestamp, [start_x, start_y], [acc_x, acc_y, acc_z], 
#               [mag_x, mag_y, mag_z], [gyro_x, gyro_y, gyro_z], [x,y,floor]
#               ]
#           ...
#         ] 
#]
#generator/stream instead of a list
def imu_data(filepath, path_to_s_subm):
    #assumes that the data is in a data folder and the file with .txt extension is the dataset. 
    files = [p for p in os.listdir(filepath) if p.endswith(".txt")]
    #outcomment the folslowing command to get data from all sites.
    files = filter_files(files, get_sites_from_sample(path_to_s_subm))
    for file in files:
        imu = list()
        imu_features = list()
        waypoints = list()
        
        f = open(filepath+"/"+file,"r")
        for line in f:
            if len(line)>0 and line[0] == "#":
                continue
            split = line.split("\t")
            if len(split)>1 and (split[1] == "TYPE_ACCELEROMETER" or split[1] == "TYPE_MAGNETIC_FIELD" or split[1] == "TYPE_GYROSCOPE"):
                #split[0] is timestamp, split[1] is the type, while the rest are x,y,z values.
                imu.append([split[0], split[1], split[2], split[3], split[4]])
            elif len(split)>1 and split[1] == "TYPE_WAYPOINT":
                file_waypoints = split[2:]
                for wpt in file_waypoints:
                    wpt_data = wpt.split(",")
                    waypoints.append([int(wpt_data[0]), float(wpt_data[1]), float(wpt_data[2]), float(wpt_data[3])])
        f.close()

        if imu == []:
            yield list()

        df = pd.DataFrame(imu)
        del imu
        #typecasting of values to proper type
        df[0] = df[0].apply(lambda x: int(x))
        df[4] = df[4].apply(lambda x: float(x))
        df[2] = df[2].apply(lambda x: float(x))
        df[3] = df[3].apply(lambda x: float(x))

        #group by timestamp
        grouped = df.groupby(0)
        for time_stamp, group in grouped:
            #find nearest waypoint here
            nearest_wp = find_nearest_wp_index(waypoints,time_stamp)
            group = group.drop_duplicates(subset=1)

            start_x, start_y = lowest_waypoint(waypoints)
            x = float(waypoints[nearest_wp][2])
            y =float(waypoints[nearest_wp][3])
            floor = int(waypoints[nearest_wp][1])
            #group = group.reindex(["TYPE_ACCELEROMETER", "TYPE_MAGNETIC_FIELD", "TYPE_GYROSCOPE"])

            if(len(group.loc[group[1]=="TYPE_ACCELEROMETER"].values) == 0 or
                    len(group.loc[group[1]=="TYPE_MAGNETIC_FIELD"].values) == 0 or
                    len(group.loc[group[1]=="TYPE_GYROSCOPE"].values) == 0):
                continue

            acc_feat = group.loc[group[1]=="TYPE_ACCELEROMETER"].values[0][2:5]
            mag_feat = group.loc[group[1]=="TYPE_MAGNETIC_FIELD"].values[0][2:5]
            gyro_feat = group.loc[group[1]=="TYPE_GYROSCOPE"].values[0][2:5]
            imu_features.append([time_stamp,[start_x,start_y], acc_feat, mag_feat, gyro_feat, [x,y,floor]])
        yield [file,imu_features]