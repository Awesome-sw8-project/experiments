import os
from functools import reduce
from ML.visualise_data import unpickle_hist

def get_mean_loss(hists):
    div = 0
    loss = 0
    val_loss = 0
    for hist in hists:
        if len(hist["loss"]) > len(hist["loss"])-1:
            div = div +1
            loss = loss + hist["loss"][len(hist["loss"])-1]
            val_loss = val_loss + hist["val_loss"][len(hist["loss"])-1]
    return loss/div, val_loss/div

def get_mean_site(path_results,site, value):
    files = [x for x in os.listdir(path_results) if x.split("_")[0] == site]
    y_results = [i for i in files if i.split("_")[3].split(".")[0] == value]
    if (len(y_results) == 0):
        return 10000000,1000000, True
    hists = [unpickle_hist(path_results,x) for x in y_results]
    l,vl = get_mean_loss(hists)
    return l, vl, False

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
    return mean_mean_l,mean_mean_vl

###For get mean best k.
def get_worst_loss(hists):
    b_hist = None
    loss = None
    n_loss = None
    for hist in hists:
        if n_loss == None:
            loss = hist["loss"][len(hist["loss"])-1]
            n_loss = float(hist["val_loss"][len(hist["val_loss"])-1])
            b_hist = hist
        elif n_loss < hist["val_loss"][len(hist["val_loss"])-1]:
            loss = hist["loss"][len(hist["loss"])-1]
            n_loss = hist["val_loss"][len(hist["val_loss"])-1]
            b_hist = hist
    if n_loss == None:
        print(hists[0])
        exit()
    return loss, n_loss, b_hist

def get_best_loss(hists):
    b_hist = None
    loss = 0
    n_loss = None
    for hist in hists:

        if n_loss == None:
            loss = hist["loss"][len(hist["loss"])-1]
            n_loss = float(hist["val_loss"][len(hist["val_loss"])-1])
            b_hist = hist
        elif n_loss > hist["val_loss"][len(hist["val_loss"])-1]:
            loss = hist["loss"][len(hist["loss"])-1]
            n_loss = hist["val_loss"][len(hist["val_loss"])-1]
            b_hist = hist
    if n_loss == None:
        print(hists[0])
        exit()
    return loss, n_loss, b_hist

def get_best_site(path_results, site, value):
    files = [x for x in os.listdir(path_results) if x.split("_")[0] == site]
    y_results = [i for i in files if i.split("_")[3].split(".")[0] == value]
    
    if (len(y_results) == 0):
        return 10000000,1000000, True
    
    hists = [unpickle_hist(path_results,x) for x in y_results]
    l, vl,_ = get_best_loss(hists)
    
    return l, vl, False

def get_worst_site(path_results, site, value):
    files = [x for x in os.listdir(path_results) if x.split("_")[0] == site]
    y_results = [i for i in files if i.split("_")[3].split(".")[0] == value]
    
    if (len(y_results) == 0):
        return 10000000,1000000, True
    
    hists = [unpickle_hist(path_results,x) for x in y_results]
    l, vl,_ = get_worst_loss(hists)
    
    return l, vl, False

def get_mean_k(path_results, option="mean", value="floors"):
    sites = [x.split("_")[0] for x in os.listdir(path_results)]
    sites = list(set(sites))
    mean_l, mean_v_l= list(),list()
    less_sites = 0
    for site in sites:
        if(option == "mean"):
            l,vl, no_data = get_mean_site(path_results, site, value)
        elif(option == "best"):
            l,vl, no_data = get_best_site(path_results, site, value)
        elif(option=="worst"):
            l,vl, no_data = get_worst_site(path_results, site, value)
        
        if no_data:
            less_sites += 1
        else:
            mean_l.append(l)
            mean_v_l.append(vl)
    mean_mean_l = reduce(lambda a,b:a+b, mean_l)/(len(mean_l)-less_sites)
    mean_mean_vl = reduce(lambda a,b:a+b, mean_v_l)/(len(mean_v_l) - less_sites)
    print("path is : {}".format(path_results))
    print("The mean of training loss MSE is : {}\nThe mean of the validation MSE is: {}".format(mean_mean_l,mean_mean_vl))
    
    return mean_mean_l,mean_mean_vl

if __name__ == "__main__":
    pass
    
