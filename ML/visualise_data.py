import os,pickle
import matplotlib.pyplot as plt, numpy as np

def unpickle_hist(path,file):
    with open("{path}/{file}".format(path=path,file=file), "rb") as f:
        hist = pickle.load(f)
    return hist

def get_data(hists, index):
    div = 0
    loss = 0
    min_loss = 1000000
    max_loss = -999
    val_loss = 0
    v_min_loss = 1000000
    v_max_loss = -999
    for hist in hists:
        if len(hist["loss"]) > index:
            div = div +1
            loss = loss + hist["loss"][index]
            if min_loss > hist["loss"][index]:
                min_loss = hist["loss"][index]
            if max_loss < hist["loss"][index]:
                max_loss = hist["loss"][index]
            val_loss = val_loss + hist["val_loss"][index]
            if v_min_loss > hist["val_loss"][index]:
                v_min_loss = hist["val_loss"][index]
            if v_max_loss < hist["val_loss"][index]:
                v_max_loss = hist["val_loss"][index]
    return loss/div,min_loss,max_loss, val_loss/div, v_min_loss,v_max_loss


def visualise(path_results, site, path_to_save):
    files = [x for x in os.listdir(path_results) if x.split("_")[0] == site]
    x_results = [i for i in files if i.split("_")[3].split(".")[0] == "xs"]
    vis_helper2(path_results, x_results, "{site}_x".format(site=site), "{path}/X".format(path=path_to_save))
    

def vis_helper(path, files, title):
    hists = [unpickle_hist(path,x) for x in files]
    epochs = max([len(x["loss"]) for x in hists])
    loss = list()
    val_loss = list()
    for x in range(0,epochs):
        l, vl = get_data(hists, x)
        loss.append(l)
        val_loss.append(vl)
    epochs = range(1, epochs + 1)
    plt.plot(epochs, loss, 'bo-', label="loss")
    plt.plot(epochs, val_loss, 'go-', label="validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Mean squared Error")
    plt.title(title)
    plt.legend()
    plt.savefig("{path}/{title}".format(path="C:/Users/Abiram Mohanaraj/Desktop/data/results",title=title))
    plt.close()

def vis_helper2(path, files, title, saveto):
    hists = [unpickle_hist(path,x) for x in files]
    epochs = max([len(x["loss"]) for x in hists])
    loss = list()
    min_loss = list()
    max_loss = list()
    val_loss = list()
    val_min = list()
    val_max = list()
    
    for x in range(0,epochs):
        #loss/div,min_loss,max_loss, val_loss/div, v_min_loss,v_max_loss
        l,l_min,l_max, v,v_min,v_max = get_data(hists, x)
        loss.append(l)
        min_loss.append(l_min)
        max_loss.append(l_max)
        val_loss.append(v)
        val_min.append(v_min)
        val_max.append(v_max)
    epochs = range(1, epochs + 1)
    plt.plot(epochs, loss, 'bo-', label="Training loss")
    plt.fill_between(epochs,max_loss, loss, color='blue', alpha=0.9)
    plt.fill_between(epochs,loss,min_loss, color='blue',alpha=0.9)

    plt.plot(epochs, val_loss, 'yo-', label="validation loss")
    plt.fill_between(epochs,val_max,val_loss, color='lightyellow',alpha=0.9)
    plt.fill_between(epochs,val_loss,min_loss, color='lightyellow',alpha=0.9)
    plt.xlabel("Epoch")
    plt.ylabel("Mean squared Error")
    plt.title(title)
    plt.legend()
    plt.savefig("{path}/{title}".format(path=saveto,title=title))
    plt.close()

if "__main__" == __name__:
    sites = [x.split("_")[0] for x in os.listdir(path)]
    for site in sites: 
        visualise(path, site, path_to_save)