import pandas as pd, os, json, gc, numpy as np, pickle
from collections import Counter

#return generator for rssi data.
def rssi_features_hybrid(rssi_type, data_path, path_to_site_index):
    path_to_data = dict()
    sites = ['5d2709bb03f801723c32852c','5d2709d403f801723c32bd39','5da138274db8ce0c98bbd3d2','5da138754db8ce0c98bca82f','5dc8cea7659e181adb076a3f']
    print(len(sites))
    files = [p for p in os.listdir(data_path) if p.endswith(".txt")]
    for site in sites:
        site_files = [p for p in files if p.startswith(site)]
        #Use index or list of BSSID values for a site.
        index = get_site_index_v2(site, path_to_site_index)
        site_file_to_iterate = sorted(site_files)
        site_file_to_iterate = sorted(site_files[:round(0.5 * len(site_files))])

        with open("../../data/data/paths/{}.pickle".format(site), "wb") as f:
            pickle.dump(site_file_to_iterate, f)

        for path in site_file_to_iterate:
            mapping = dict()
            waypoints = list()
            wifi = list()
            #error are set to ignore as some special chars do not have utf-8 encoding.
            f = open(data_path+"/"+path,"r", errors='ignore')
            for line in f:
                if len(line)>0 and line[0] == "#":
                    continue
                #data is separated by tabs.
                split = line.split("\t")
                if len(split)>1 and split[1] == rssi_type:
                    wifi.append([split[0], split[2], split[3], split[4]])
                elif len(split)>1 and split[1] == "TYPE_WAYPOINT":
                    file_waypoints = split[2:]
                    for wpt in file_waypoints:
                        wpt_data = wpt.split(",")
                        waypoints.append([int(wpt_data[0]), float(wpt_data[1]), float(wpt_data[2]), float(wpt_data[3])])
            f.close()
            if wifi == []:
                continue
            df = pd.DataFrame(wifi)
            del wifi
            gc.collect()
            df[0] = df[0].apply(lambda x: int(x))
            grouped = df.groupby(0)
            for time_stamp, group in grouped:
                #find nearest waypoint here
                nearest_wp = find_nearest_wp_index(waypoints, time_stamp)
                group = group.drop_duplicates(subset=2)

                #ground truth
                x = float(waypoints[nearest_wp][2])
                y =float(waypoints[nearest_wp][3])
                floor = int(waypoints[nearest_wp][1])

                temp = group.iloc[:,2:4]
                feat = temp.set_index(2).reindex(index).replace(np.nan, -999)
                feat = feat.transpose()
                if ((feat.values[0]==-999).sum()== len(feat.values[0])):
                    continue
                else:
                    print("we have data\n")
                mapping[str(time_stamp)] = (feat.values[0],[x,y,floor])
            path_to_data[path] = mapping
        yield site, path_to_data

def imu_data_hybrid(filepath, path_to_path_index):
    sites = ['5d2709bb03f801723c32852c','5d2709d403f801723c32bd39','5da138274db8ce0c98bbd3d2','5da138754db8ce0c98bca82f','5dc8cea7659e181adb076a3f']
    for site in sites:
        path_to_data = dict()
        with open("{}/{}.pickle".format(path_to_path_index,site), "rb") as f:
            site_file_to_iterate = pickle.load(f)

        for file in site_file_to_iterate:
            time_to_data = dict()
            imu = list()
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
                time_to_data[str(time_stamp)] =([start_x,start_y], acc_feat, mag_feat, gyro_feat, [x,y,floor])
            path_to_data[file] = time_to_data
        yield (site, path_to_data)
