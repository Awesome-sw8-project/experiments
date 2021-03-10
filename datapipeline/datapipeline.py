import pandas as pd

import os
import json
import numpy as np

basepath = ''

subm_file = pd.read_csv('sample_submission.csv')
subm_df = subm_file['site_path_timestamp'].apply(lambda x: pd.Series(x.split("_")))
sites_for_prediction = sorted(subm_df[0].value_counts().index.tolist())
bssid_for_prediction = dict()
dfs = []
for site in sites_for_prediction:
    #assumes that the data is in a data folder. Possibly needs to explicitly typecast site to string (str())
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
            print(group)
    
    
    





if __name__ == "__main__":
    #commands will be written here
    print()
    