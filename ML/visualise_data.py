import os, pickle
import matplotlib.pyplot as plt

def unpickle_hist(path,file):
    with open("{path}/{file}".format(path=path,file=file), "rb") as f:
        hist = pickle.load(f)
    return hist

def get_data(hists, index):
    div = 0
    loss = 0
    val_loss = 0
    for hist in hists:
        if len(hist["loss"]) > index:
            div = div +1
            loss = loss + hist["loss"][index]
            val_loss = val_loss + hist["val_loss"][index]
    return loss/div, val_loss/div

def get_data2(hists, index):
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


def visualise(path_results, site, path_to_save, skip=0):
    files = [x for x in os.listdir(path_results) if x.split("_")[0] == site]
    x_results = [i for i in files if i.split("_")[3].split(".")[0] == "xs"]
    vis_helper(path_results, x_results, "{site}_x".format(site=site), "{path}/X".format(path=path_to_save), skip=skip)
    x_results = [i for i in files if i.split("_")[3].split(".")[0] == "ys"]
    vis_helper(path_results, x_results, "{site}_y".format(site=site), "{path}/Y".format(path=path_to_save), skip=skip)
    x_results = [i for i in files if i.split("_")[3].split(".")[0] == "floors"]
    vis_helper(path_results, x_results, "{site}_floor".format(site=site), "{path}/Floor".format(path=path_to_save), skip=skip)
    
def print_vl(path_results, site):
    files = [x for x in os.listdir(path_results) if x.split("_")[0] == site]
    x_results = [i for i in files if i.split("_")[3].split(".")[0] == "ys"]
    v_l_helper(path_results, x_results)

def v_l_helper(path, files):
    hists = [unpickle_hist(path,x) for x in files]
    epochs = max([len(x["loss"]) for x in hists])
    loss = list()
    val_loss = list()
    for x in range(0,epochs):
        l, vl = get_data(hists, x)
        loss.append(l)
        val_loss.append(vl)

def vis_helper(path, files, title, saveto, skip=0):
    hists = [unpickle_hist(path,x) for x in files]
    epochs = max([len(x["loss"]) for x in hists])
    
    loss = list()
    val_loss = list()
    for x in range(0,epochs):
        l, vl = get_data(hists, x)
        loss.append(l)
        val_loss.append(vl)
    epochs = [x for x in range(1,epochs+1)]
    plt.plot(epochs[skip:], loss[skip:], 'bo-', label="Training loss")
    plt.plot(epochs[skip:], val_loss[skip:], 'go-', label="Validation loss")
    if skip == 0:
        plt.xlabel("Epoch")
        plt.ylabel("Mean squared Error")
    else:
        plt.legend()
    plt.savefig("{}/{}".format(saveto,title),bbox_inches='tight')
    plt.close()


if "__main__" == __name__:
    path = ""
    path_to_save = ''
    sites = [x.split("_")[0] for x in os.listdir(path)]
    sites = sorted(list(set(sites)))
    for site in sites:
        print(site)
        visualise(path, site, path_to_save, skip=0)
