#imports
import pandas as pd, os, json, gc, numpy as np, pickle
from collections import Counter

#Path variables
path_to_test = ''
train_data_path = '' #data_path
path_to_s_subm = ''
os.mkdir("./train")
os.mkdir("./test")
os.mkdir("./bssid_index")
path_to_bssid_index = './bssid_index' #path_to_site_index
save_train = './train'
save_test = './test'

#code
def test_bssids(to_test, site):
    files = [f for f in os.listdir(to_test) if f.split("_")[0]==site]
    bssid_list = list()
    for file in files:
        with open("{}/{}".format(to_test,file), "r") as f:
            data = f.readlines()
        for line in data:
            splits = line.split("\t")
            if splits[0] == '#':
                continue
            if splits[1] == 'TYPE_WIFI':
                bssid_list.append(splits[3])
    return bssid_list
#Get a list of the BSSID values for each node for a particular site. 
# This is used to index the RSSI values such that the same index in each list correpond to the same BSSID.
def get_site_index(rssi_type,siteID, site_files, data_path, path_to_test_set, occ):
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
    bssid = list(bssids_dict)
    test_bssid = test_bssids(path_to_test_set, siteID)
    print("test BSSID values for site {} is {}".format(siteID,len(test_bssid)))
    bssid.extend(test_bssid)
    return sorted(list(set(bssid)))

#finds the index of the temporally nearest waypoint in the waypoints list given a timestamp
def find_nearest_wp_index(waypoints, time_stamp):
    dists = list()
    for e,k in enumerate(waypoints):
        dist = abs(int(time_stamp)- int(k[0]))
        dists.append(dist)
    return np.argmin(dists)

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

#main function
#return generator for rssi data.
def time_rssi_features(rssi_type, data_path, path_to_s_subm, path_to_site_index, path_to_test):
    sites = [p.split("_")[0] for p in os.listdir(data_path) if p.endswith(".txt")]
    
    #remove duplicate site ids
    sites = list(sorted(set(sites)))
    
    #remove site ids not part of sample submission
    ssub_sites = get_sites_from_sample(path_to_s_subm)
    sites = [p for p in sites if p in ssub_sites]
    
    files = [p for p in os.listdir(data_path) if p.endswith(".txt")]
    for site in sites:
        site_files = [p for p in files if p.startswith(site)]
        train_data, ground_truth, timestamp = time_rssi_feats_site(site_files, data_path, rssi_type, site, path_to_site_index, path_to_test)
        yield site, train_data, ground_truth

def time_test_feats_pickled(rssi_type, path_to_s_subm,path_to_test, path_to_indices, path_to_save_test):
    ssubm = pd.read_csv(path_to_s_subm)
    #ssubm_df contains site, path and timestamp
    ssubm_df = ssubm["site_path_timestamp"].apply(lambda x: pd.Series(x.split("_")))
    

    #group by the sites
    ssubm_groups = ssubm_df.groupby(0)    
    for gid0, g0 in ssubm_groups:
        
        with open("{path_to_i}/{site}.pickle".format(path_to_i=path_to_indices, site=gid0), "rb") as f:
            index = pickle.load(f)
        
        time_series = list()
        
        for gid, g in g0.groupby(1):
            with open("{path_to_test}/{site}_{path}.txt".format(path_to_test=path_to_test,site =gid0, path = gid), "r") as f:
                txt = f.readlines()
            rssi = list()
            for line in txt:
                line = line.split("\t")
                if line[1] == rssi_type:
                    rssi.append(line)
            rssi_df = pd.DataFrame(rssi)
            del rssi
            gc.collect()
            rssi_points = pd.DataFrame(rssi_df.groupby(0).count().index.tolist())

            for timepoint in g.iloc[:,2].tolist():
                deltas = (rssi_points.astype(int)-int(timepoint)).abs()
                min_delta_idx = deltas.values.argmin()
                feats = list()
                for ind in range(0,min_delta_idx):
                    rssi_block_timestamp = rssi_points.iloc[ind].values[0]
                    rssi_block = rssi_df[rssi_df[0]== rssi_block_timestamp].drop_duplicates(subset=3)
                    feat = rssi_block.set_index(3)[4].reindex(index).fillna(-999)
                    feats.append(feat.values)
                
                site_path_timestamp = "{site}_{path}_{timestamp}".format(site=g.iloc[0,0], path=g.iloc[0,1],timestamp=timepoint)
                time_series.append((feats, site_path_timestamp))
                
        with open("{path_to_save}/{site}.pickle".format(path_to_save=path_to_save_test,site=gid0),"wb") as f:
            pickle.dump(time_series,f)

#execution
time_rssi_features("TYPE_WIFI", train_data_path, path_to_s_subm, path_to_bssid_index, path_to_test)
time_test_feats_pickled("TYPE_WIFI", path_to_s_subm, path_to_test, path_to_bssid_index, save_test)
