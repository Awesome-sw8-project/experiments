import pandas as pd, os, json, gc, numpy as np
from collections import Counter

#finds all bssid values with an occurence over 'occ' and saves it into a bssids.json file. 
# Type specifies which type of data is used, can either be TYPE_WIFI or TYPE_IBEACON
#Deprecated!!!
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
#Deprecated!!!
def get_all_bssids(type):
    f = open(type+"_bssids.json", "r")
    lst = json.loads(f.read())
    f.close()
    return lst

#TODO:maybe make generator.
#Deprecated!!!
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
