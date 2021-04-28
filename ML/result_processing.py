import os,pickle, math
import matplotlib.pyplot as plt, numpy as np
import seaborn as sns, pandas as pd
from functools import reduce

from visualise_data import unpickle_hist, get_data

def get_mean_site(path_results,site):
    files = [x for x in os.listdir(path_results) if x.split("_")[0] == site]
    y_results = [i for i in files if i.split("_")[3].split(".")[0] == "ys"]
    hists = [unpickle_hist(path_results,x) for x in y_results]
    epochs = max([len(x["loss"]) for x in hists])
    l, vl = get_data(hists, epochs-1)
    return l, vl

def get_mean(path_results):
    sites = [x.split("_")[0] for x in os.listdir(path_results)]
    sites = list(set(sites))
    mean_l, mean_v_l= list(),list()
    for site in sites:
        l,vl = get_mean_site(path_results, site)
        mean_l.append(l)
        mean_v_l.append(vl)
    mean_mean_l = reduce(lambda a,b:a+b, mean_l)/len(mean_l)
    mean_mean_vl = reduce(lambda a,b:a+b, mean_v_l)/len(mean_v_l)
    print("path is : {}".format(path_results))
    print("The mean of training loss MSE is : {}\nThe mean of the validation MSE is: {}".format(mean_mean_l,mean_mean_vl))
    #print(mean_l)
    #print(mean_v_l)
    return mean_mean_l,mean_mean_vl
if __name__ == "__main__":
    get_mean("")