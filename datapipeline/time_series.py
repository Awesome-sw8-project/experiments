import pandas as pd, os, json, gc, numpy as np, pickle
from collections import Counter

from datapipeline import *

#necessary dimensions [sample, timestep, features]
#returns training data and ground truth for a site.
def time_rssi_feats_site(site_files, data_path, rssi_type, site, path_to_site_index, path_to_test):
    wifi_features = list()
    ground_truth = list()
    max_timestep = 0
    #Create index or list of BSSID values for a site.
    index = get_site_index(rssi_type, site, site_files, data_path,path_to_test, 1000)
    with open("{}/{}.pickle".format(path_to_site_index, site), "wb") as f:
        pickle.dump(index, f)
    
    for path in site_files:
        time_series,ground_series = list(),list()
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
            time_series.append(feat.values[0])
            ground_series.append([x,y,floor])
        wifi_features.append(time_series)
        ground_truth.append(ground_series)
        if len(time_series) > max_timestep:
            max_timestep = len(time_series)
    return wifi_features, ground_truth, max_timestep

#return generator for rssi data.
def time_rssi_features(rssi_type, data_path, path_to_s_subm, path_to_site_index, path_to_test):
    sites = [p.split("_")[0] for p in os.listdir(data_path) if p.endswith(".txt")]
    
    #remove duplicate site ids
    sites = list(set(sites))
    
    #remove site ids not part of sample submission
    ssub_sites = get_sites_from_sample(path_to_s_subm)
    sites = [p for p in sites if p in ssub_sites]
    print(len(sites))
    files = [p for p in os.listdir(data_path) if p.endswith(".txt")]
    for site in sites:
        site_files = [p for p in files if p.startswith(site)]
        train_data, ground_truth = time_rssi_feats_site(site_files, data_path, rssi_type, site, path_to_site_index, path_to_test)
        yield  site, train_data, ground_truth