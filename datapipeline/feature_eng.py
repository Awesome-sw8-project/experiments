import os, pickle, numpy as np

def create_new_train(path_to_train, path_to_save):
    files = [f for f in os.listdir(path_to_train)]
    for file in files:
        with open("{}/{}".format(path_to_train, file), "rb") as f:
            site, train, label = pickle.load(f)
        new_train, new_label = list(), list()
        total = 0
        for (train_sample, label_x) in zip(train, label):
            if not ((train_sample == -999).sum() == len(train_sample)):
                new_train.append(train_sample)
                new_label.append(label_x)
        with open("{}/{}".format(path_to_save,file), "wb") as f:
            pickle.dump((site,new_train,new_label), f)
        print("New data for site: {}".format(site))
    print("done")

def normalise_value(x, min_val, max_val):
    x = np.asarray(x).astype(np.int)
    return (x-min_val)/(max_val-min_val)


def min_max_normalise(p_t_train, p_t_test, site, save_train,save_test):
    with open("{}/{}.pickle".format(p_t_train,site),"rb") as f:
        siteID, train, label = pickle.load(f)
    flat_train = np.asarray(train).astype(np.int).flatten()
    min_val = np.amin(flat_train)
    max_val = np.amax(flat_train)
    del flat_train
    train = np.asarray(train).astype(np.int)
    new_train = [normalise_value(x, min_val,max_val) for x in train]
    with open("{}/{}.pickle".format(save_train,siteID),"wb") as f:
        pickle.dump((site,new_train,label), f)
    
    #test data normalisation
    with open("{}/{}.pickle".format(p_t_test,site),"rb") as f:
        test_data = pickle.load(f)
    n_test_data = [(normalise_value(feat, min_val,max_val), stamp) for (feat, stamp) in test_data]
    with open("{}/{}.pickl".format(save_test,site), "wb") as f:
        pickle.dump(n_test_data, f)
    print("Feature Engineered for site: {}".format(site))

def normalise_for_sites(pt_train, pt_test, save_train, save_test):
    sites = [site.split(".")[0] for site in os.listdir(pt_train)]
    for site in sites:
        min_max_normalise(pt_train,pt_test, site,save_train,save_test)

if __name__ == "__main__":
    normalise_for_sites(pt_train, pt_test, save_tr, save_te)