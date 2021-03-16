import pandas as pd
import os
import json
import numpy as np
import gc
from collections import Counter

basepath = ''
#site_name_path_name
#TODO:check whether a sample_submission.csv file exist in experiments and refer to that file.
#subm_file = pd.read_csv('sample_submission.csv')
#subm_df = subm_file['site_path_timestamp'].apply(lambda x: pd.Series(x.split("_")))
#sites_for_prediction = sorted(subm_df[0].value_counts().index.tolist())
bssid_for_prediction = dict()

#finds all bssid value and saves it into a bssids.json file
def calculate_all_bssids():
    all_bssids = list()
    files = [p for p in os.listdir("data") if p.endswith(".txt")]
    for file in files:
        f = open("data/"+file, "r")
        for line in f:
            if line[0] == "#":
                continue
            split = line.split("\t")
            if len(split) > 0 and split[1] == "TYPE_WIFI":
                all_bssids.append(split[3]) #need to find real index.
        f.close()    
    bssids_dict = dict(Counter(all_bssids))
    #if we need to filter the bssid values. this probably needs readjustment
    bssids_dict = {bssid:occurence for (bssid,occurence) in bssids_dict.items() if occurence > 100}
    all_bssids = list(bssids_dict)
    bssids_json = json.dumps(all_bssids)
    #print(len(all_bssids))
    f = open("bssids.json", "w")
    f.write(bssids_json)
    f.close()
    
#get all bssids from a bssids.json file. Can only be used after a call to calculate_all_bssids.
def get_all_bssids():
    f = open("bssids.json", "r")
    lst = json.loads(f.read())
    f.close()
    return lst

#maybe make it per file write if cannot hold data in main memmory.
def wifi_feature_construction(bssids):
    wifi_features = list()
    
    #assumes that the data is in a data folder. 
    files = [p for p in os.listdir("data") if p.endswith(".txt")]
    #files = [p for p in os.listdir("../data") if p.split('_')[0].startswith(site)]
    wifi = list()
    for file in files:
        waypoints = list()
        f = open("data/"+file,"r")
        for line in f:
            if len(line)>0 and line[0] == "#":
                continue
            split = line.split("\t")
            if(len(split)<=1):
                print(split)
            if len(split)>1 and split[1] == "TYPE_WIFI":
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
                #find true waypoint here
                dists = list()
                for e,k in enumerate(waypoints):
                    dist = abs(int(time_stamp)- int(k[0]))
                    dists.append(dist)
                nearest_wp = np.argmin(dists)
                temp = group.iloc[:,2:4]
                #temp.drop_duplicates(2, inplace=True) # this might be necessary for dummy data with same bssids
                feat = temp.set_index(2).reindex(index).replace(np.nan, -999)
                #TODO: recheck for proper indices
                x = float(waypoints[nearest_wp][2])
                y =float(waypoints[nearest_wp][3])
                floor = int(waypoints[nearest_wp][1])
                """feat["X"] = 
                feat["Y"] = 
                feat["floor"] =""" 
                #feat.reset_index(drop=True,inplace=True)
                feat = feat.transpose()
                feat_arr = feat.to_numpy()
                feat_arr = np.append(feat_arr,[x,y,floor])
                wifi_features.append(feat_arr)
            del df
        del wifi
        gc.collect()
    return wifi_features

#calculate_all_bssids()
bssids = get_all_bssids()
print(wifi_feature_construction(bssids))
#print(wifi_feature_construction(sites_for_prediction,bssids))

if __name__ == "__main__":
    #commands will be written here
    print()
    