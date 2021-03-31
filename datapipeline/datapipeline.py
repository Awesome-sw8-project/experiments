import pandas as pd, os, json, gc, numpy as np, pickle
from collections import Counter

train_path = ''
path_to_sample_submission = ''
path_to_site_indices = ''

def get_sites_from_sample(path_to_sample):
    sub_df = pd.read_csv(path_to_sample)
    sub_df2 = sub_df["site_path_timestamp"].apply(lambda x: pd.Series(x.split("_")))
    sites = sorted(sub_df2[0].value_counts().index.tolist())
    return sites

def filter_files(files, sites):
    return [p for p in files if p.split("_")[0] in sites]


#finds the index of the temporally nearest waypoint in the waypoints list given a timestamp
def find_nearest_wp_index(waypoints, time_stamp):
    dists = list()
    for e,k in enumerate(waypoints):
        dist = abs(int(time_stamp)- int(k[0]))
        dists.append(dist)
    return np.argmin(dists)

#get waypoint with lowest timestamp in the list waypoints
def lowest_waypoint(waypoints):
    timestamp = None
    x = y = None
    for e,k in enumerate(waypoints):
        if timestamp == None or int(k[0])<timestamp:
            timestamp = int(k[0])
            x,y = float(k[2]), float(k[3])
    return x,y


#Get a list of the BSSID values for each node for a particular site. 
# This is used to index the RSSI values such that the same index in each list correpond to the same BSSID.
def get_site_index(rssi_type, site_files, data_path, occ):
    bssids_list = list()
    for site in site_files:
        f = open(data_path+"/"+site, "r", errors='ignore')
        for line in f:
            #omit the metadata in each file
            if line[0] == "#":
                continue
            #data is separated by tabs
            split = line.split("\t")
            if len(split) > 0 and split[1] == rssi_type:
                bssids_list.append(split[3]) #need to find real index.
        f.close()
    #creates a dictionary with the key being bssid and value being the occurance of the bssid value in the dataset.    
    bssids_dict = dict(Counter(bssids_list))
    #filtering of the bssids which are less than 'occ'
    bssids_dict = {bssid:occurence for (bssid,occurence) in bssids_dict.items() if occurence > occ}
    return list(bssids_dict)

#returns training data and ground truth for a site.
def rssi_feats_site(site_files, data_path, rssi_type, site, path_to_site_index):
    wifi_features = list()
    ground_truth = list()
    
    #Create index or list of BSSID values for a site.
    index = get_site_index(rssi_type, site_files, data_path, 1000)
    with open("{path}/{site}.pickle".format(path=path_to_site_index, site=site), "wb") as f:
        pickle.dump(index, f)
    for path in site_files:
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
            wifi_features.append(feat.values[0])
            ground_truth.append([x,y,floor])
    return wifi_features, ground_truth
        
#return generator for rssi data.
def rssi_features(rssi_type, data_path, path_to_s_subm, path_to_site_index):
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
        train_data, ground_truth = rssi_feats_site(site_files, data_path, rssi_type, site, path_to_site_index)
        yield  site, train_data, ground_truth



#returns training data and ground truth for a site.
def test_feats(rssi_type, index, path_to_s_subm):
    ssubm = pd.read_csv(path_to_sample_submission)
    ssubm_df = ssubm["site_path_timestamp"].apply(lambda x: pd.Series(x.split("_")))
    #group by the sites
    ssubm_groups = ssubm_df.groupby(0)    
    for gid0, g0 in ssubm_groups:
        feats = list()
        for gid, g in g0.groupby(1):
            with open("path_to_test .txt".format(), "rb") as f:
                txt. f.readlines()
            wifi = list()
            for line in txt:
                line = line.strip().split()
                if line[1] == rssi_type:
                    wifi.append(line)
            wifi_df = pd.DataFrame(wifi)
            wifi_points = pd.DataFrame(wifi_df.groupby(0).count().index.tolist())

            for timepoint in g.iloc[:,2].tolist():
                deltas = (wifi_points.astype(int)-int(timepoint)).abs()
                min_delta_idx = deltas.values.argmin()
                wifi_block_timestamp = wifi_points.iloc[min_delta_idx].values[0]
                wifi_block = wifi_df[wifi_df[0]== wifi_block_timestamp].drop_duplicates(subset=3)
                feat = wifi_block.set_index(3)[4].reindex(index).fillna(-999)
                feat['site_path_timestamp'] = "{site}_{path}_{timestamp}".format(site=g.iloc[0,0], path=g.iloc[0,1],timestamp=timepoint)
                feats.append(feat)
        feature_df = pd.concat(feats,axis=1).T
        "feature_df.values or to csv

    wifi_features = list()
    ground_truth = list()
    
    for path in site_files:
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
            group = group.drop_duplicates(subset=2)
            
            temp = group.iloc[:,2:4]
            feat = temp.set_index(2).reindex(index).replace(np.nan, -999)
            feat = feat.transpose()
            wifi_features.append(feat.values[0])
    return wifi_features


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


#this function is only included to show how to access array written to file using np
def load_np_to_text(filename):
    return np.loadtxt(filename,delimiter=",")

if __name__ == "__main__":
    #train_gen = rssi_features("TYPE_WIFI",train_path, path_to_sample_submission, path_to_site_indices)
    #rssi_type, data_path, path_to_s_subm, path_to_site_index)
    #for site, train, labels in train_gen:
    #    with open("{site}.pickle".format(site=site), "wb") as f:
    #        pickle.dump((site,train, labels), f)
    #get_sites_from_sample(path_to_sample_submission)
    
    
    #gen = imu_data("~/P8/data/data/train")
    #gen = wifi_features("TYPE_WIFI","~/P8/data/data/train")
    #site, train, labels = next(gen)
    #print(site)
    #print(train)
    #print(labels)
    pass
