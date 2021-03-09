import pandas as pd
import glob
import os
import json
import numpy as np

basepath = ''

subm_file = pd.read_csv('sample_submission.csv')
subm_df = pd.apply(lambda x: pd.Series(x.split("_")))
sites_for_prediction = sorted(subm_df[0].value_counts().index.tolist())
bssid_for_prediction = list()
dfs =list()
for site in sites_for_prediction:
    #assumes that the data is in a data folder. Possibly needs to explicitly typecast site to string (str())
    files = [p for p in os.listdir("../data") if p.split('_')[0].startswith(site)]
    wifi = list()
    for file in files:
        f = open(file)
        for _,line in enumerate(f):
            msmt = json.loads(line)
            if msnt["type"] == "TYPE_wifi":
                wifi.append(measurement)
    df = pd.DataFrame(np.array(wifi))
    val_counts = df[3].value_counts()
    bssid_for_prediction[site] = val_counts[val_counts > 1000].index.tolist()
    index = sorted(bssid_for_prediction[site])
    for _, g in in df.groupby(0):
        g = g.drop_duplicates(subset=3) #this needs to testing
        temp = g.iloc[:,3:5]
        feat = temp.set_index(3).reindex(index).replace(np.nan,-999).T
        dfs.append(feat)
    final_df = pd.concat(dfs)





if __name__ == "__main__":
    #commands will be written here
    print()
    