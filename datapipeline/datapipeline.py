import pandas as pd
import os
import json
import numpy as np
import gc
from collections import Counter

#finds all bssid values with an occurence over 'occ' and saves it into a bssids.json file. 
# Type specifies which type of data is used, can either be TYPE_WIFI or TYPE_IBEACON
def calculate_all_bssids(occ, type):
    bssids_list = list()
    files = [p for p in os.listdir("data") if p.endswith(".txt")]
    for file in files:
        f = open("data/"+file, "r")
        for line in f:
            #omit the metadata in each file
            if line[0] == "#":
                continue
            split = line.split("\t")
            if len(split) > 0 and split[1] == type:
                bssids_list.append(split[3]) #need to find real index.
        f.close()
    #creates a dictionary with the key being bssid and value being the occurance of the bssid value in the dataset.    
    bssids_dict = dict(Counter(bssids_list))
    #filtering of the bssids which are less than 'occ'
    bssids_dict = {bssid:occurence for (bssid,occurence) in bssids_dict.items() if occurence > occ}
    bssids_list = list(bssids_dict)
    bssids_json = json.dumps(bssids_list)
    f = open(type +"_bssids.json", "w")
    f.write(bssids_json)
    f.close()
    
#get all bssids from a bssids.json file. Can only be used after a call to calculate_all_bssids.
def get_all_bssids(type):
    f = open(type+"_bssids.json", "r")
    lst = json.loads(f.read())
    f.close()
    return lst

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

#TODO:maybe make generator.
def wifi_feature_construction(bssids, type):
    wifi_features = list()
   
    #assumes that the data is in a data folder and the file with .txt extension is the dataset. 
    files = [p for p in os.listdir("data") if p.endswith(".txt")]
    wifi = list()
    for file in files:
        waypoints = list()
        f = open("data/"+file,"r")
        for line in f:
            if len(line)>0 and line[0] == "#":
                continue
            split = line.split("\t")
            if len(split)>1 and split[1] == type:
                #maybe with split 3 instead of split 2 depending on what bssid is.
                wifi.append([split[0], split[2], split[3], split[4]])
            elif len(split)>1 and split[1] == "TYPE_WAYPOINT":
                file_waypoints = split[2:]
                for wpt in file_waypoints:
                    wpt_data = wpt.split(",")
                    waypoints.append([int(wpt_data[0]), float(wpt_data[1]), float(wpt_data[2]), float(wpt_data[3])])
        f.close()

        if not wifi == []:
            df = pd.DataFrame(wifi)
            #val_counts = df[3].value_counts() # this step is only necessary if we would like not include all routers/bssid
            #bssid_count_index = val_counts.index.tolist()
            index = bssids
            df[0] = df[0].apply(lambda x: int(x))
            grouped = df.groupby(0)
            for time_stamp, group in grouped:
                #find nearest waypoint here
                nearest_wp = find_nearest_wp_index(waypoints,time_stamp)
                x = float(waypoints[nearest_wp][2])
                y =float(waypoints[nearest_wp][3])
                floor = int(waypoints[nearest_wp][1])
                temp = group.iloc[:,2:4]
                #temp.drop_duplicates(2, inplace=True) # this might be necessary for dummy data with same bssids
                feat = temp.set_index(2).reindex(index).replace(np.nan, -999)
                #feat.reset_index(drop=True,inplace=True)
                feat = feat.transpose()
                feat_arr = feat.to_numpy()
                feat_arr = np.append(feat_arr,[x,y,floor])
                wifi_features.append(feat_arr)
            del df
        del wifi
        gc.collect()
    #np.savetxt('data.csv',np.asarray(wifi_features),delimiter=',', fmt="%s") #this can be used to write the data to files.
    return wifi_features

def wifi_feats_site(site_files, data_path, rssi_type):
    wifi_features = list()
    ground_truth = list()
    index = 5 # TODO fix this
    for path in site_files:
        waypoints = list()
        wifi = list()
        f = open(data_path+path,"r")
        for line in f:
            if len(line)>0 and line[0] == "#":
                continue
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
        index = bssids #TODO fix this
        df[0] = df[0].apply(lambda x: int(x))
        grouped = df.groupby(0)
        for time_stamp, group in grouped:
            #find nearest waypoint here
            nearest_wp = find_nearest_wp_index(waypoints, time_stamp)
            x = float(waypoints[nearest_wp][2])
            y =float(waypoints[nearest_wp][3])
            floor = int(waypoints[nearest_wp][1])
            temp = group.iloc[:,2:4]
            feat = temp.set_index(2).reindex(index).replace(np.nan, -999)
            #feat.reset_index(drop=True,inplace=True)
            feat = feat.transpose()
            feat_arr = feat.to_numpy()
            feat_arr = np.append(feat_arr)
            wifi_features.append(feat_arr)
            ground_truth.append([x,y,floor])
    return wifi_features, ground_truth
        
#return generator for rssi data.
def wifi_features(rssi_type, data_path):
    sites = [p.split("_")[0] for p in os.listdir(data_path) if p.endswith()]
    sites = list(set(sites))
    files = [p for p in os.listdir(data_path) if p.endswith(".txt")]
    for site in sites:
        site_files = [p for p in files if p.startswith(site)]
        yield wifi_feats_site(site_files, data_path, rssi_type)



#output format of a next call to imu_data
#[filename, [
#               [timestamp, [start_x, start_y], [acc_x, acc_y, acc_z], 
#               [mag_x, mag_y, mag_z], [gyro_x, gyro_y, gyro_z], [x,y,floor]
#               ]
#           ...
#         ] 
#]
#generator/stream instead of a list
def imu_data(filepath):
    #assumes that the data is in a data folder and the file with .txt extension is the dataset. 
    files = [p for p in os.listdir(filepath) if p.endswith(".txt")]
    
    for file in files:
        imu = list()
        imu_features = list()
        waypoints = list()
        
        f = open(filepath+"\\"+file,"r")
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
        df[4] = df[4].apply(lambda x: float(x[:-1]))
        df[2] = df[2].apply(lambda x: float(x))
        df[3] = df[3].apply(lambda x: float(x))
            
        #group by timestamp
        grouped = df.groupby(0)
        for time_stamp, group in grouped:
            #find nearest waypoint here
            nearest_wp = find_nearest_wp_index(waypoints,time_stamp)
            start_x, start_y = lowest_waypoint(waypoints)
                
            x = float(waypoints[nearest_wp][2])
            y =float(waypoints[nearest_wp][3])
            floor = int(waypoints[nearest_wp][1])
            #group = group.reindex(["TYPE_ACCELEROMETER", "TYPE_MAGNETIC_FIELD", "TYPE_GYROSCOPE"])
            acc_feat = group.loc[group[1]=="TYPE_ACCELEROMETER"].to_numpy()[0][2:5]
            mag_feat = group.loc[group[1]=="TYPE_MAGNETIC_FIELD"].to_numpy()[0][2:5]
            gyro_feat = group.loc[group[1]=="TYPE_GYROSCOPE"].to_numpy()[0][2:5]
            imu_features.append([time_stamp,[start_x,start_y], acc_feat, mag_feat, gyro_feat, [x,y,floor]])
        yield [file,imu_features]


#this function is only included to show how to access array written to file using np
def load_np_to_text(filename):
    return np.loadtxt(filename,delimiter=",")

if __name__ == "__main__":
    #gen = imu_data("data")
    
    #calculate_all_bssids(100, "TYPE_WIFI")
    #bssids = get_all_bssids()
    #feat_arr = wifi_feature_construction(bssids, "TYPE_WIFI")
    #print(feat_arr[0][3:6])
    #print(wifi_feature_construction(bssids)
    pass