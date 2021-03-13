import pandas as pd
import os
import json
import numpy as np
import gc
from collections import Counter

basepath = ''

#TODO:check whether a sample_submission.csv file exist in experiments and refer to that file.
subm_file = pd.read_csv('sample_submission.csv')
subm_df = subm_file['site_path_timestamp'].apply(lambda x: pd.Series(x.split("_")))
sites_for_prediction = sorted(subm_df[0].value_counts().index.tolist())
bssid_for_prediction = dict()

#finds all bssid value and saves it into a bssids.json file
def calculate_all_bssids(sites_for_prediction):
    all_bssids = list()
    for site in sites_for_prediction:
        files = [p for p in os.listdir("data") if p.split('_')[0].startswith(site)]
        for file in files:
            f = open("data/"+file, "r")
            data = json.loads(f.read())
            f.close()
            for msm in data["sensorData"]:
                if msm["Type"] == "TYPE_WIFI":
                    all_bssids.append(msm["bssid"])
    
    
    bssids_dict = dict(Counter(all_bssids))
    #if we need to filter the bssid values.
    #bssids_dict = {bssid:occurence for (key,value) in bssids_dict.items() if occurence > 1000}
    all_bssids = list(bssids_dict)
    bssids_json = json.dumps(all_bssids)
    f = open("bssids.json", "w")
    f.write(bssids_json)
    f.close()
    
#get all bssids from a bssids.json file. Can only be used after a call to calculate_all_bssids.
def get_all_bssids():
    f = open("bssids.json", "r")
    lst = json.loads(f.read())
    f.close()
    return lst


def wifi_feature_construction(sites, bssids):
    wifi_features = list()
    for site in sites:
        #assumes that the data is in a data folder. 
        files = [p for p in os.listdir("data") if p.split('_')[0].startswith(site)]
        #files = [p for p in os.listdir("../data") if p.split('_')[0].startswith(site)]
        wifi = []
        for file in files:
            f = open("data/"+file,"r")
            data = json.loads(f.read())
            f.close()
            for msm in data["sensorData"]:
                if msm["Type"] == "TYPE_WIFI":
                    wifi.append(list(msm.values()))
                
        if not wifi == []:
            df = pd.DataFrame(wifi)
            val_counts = df[3].value_counts() # this step is only necessary if we would like not include all routers/bssid
            bssid_for_prediction[site] = val_counts.index.tolist()
            index = sorted(bssid_for_prediction[site])
            df[0] = df[0].apply(lambda x: int(x))
            grouped = df.groupby(0)
            for _, group in grouped:
                temp = group.iloc[:,3:5]
                temp.drop_duplicates(3, inplace=True)
                feat = temp.set_index(3).reindex(index).replace(np.nan, -999)
                wifi_features.append(feat[4].to_numpy())
            del df
        del wifi
        gc.collect()
    return wifi_features

bssids = get_all_bssids()
print(wifi_feature_construction(sites_for_prediction,bssids))

if __name__ == "__main__":
    #commands will be written here
    print()
    