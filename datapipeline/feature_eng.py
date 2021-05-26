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
            else:
                total += 1
        with open("{}/{}".format(path_to_save,file), "wb") as f:
            pickle.dump((site,new_train,new_label), f)
        print("New data for site: {}, missing values: {}\n".format(site,total))
    print("Fin")

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

def evaluation_min_max_normalise(p_t_train, p_t_test, site, save_train,save_test):
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
        siteID, train, label = pickle.load(f)
    train = np.asarray(train).astype(np.int)
    new_train = [normalise_value(x, min_val,max_val) for x in train]
    with open("{}/{}.pickl".format(save_test,site), "wb") as f:
        pickle.dump((site,new_train,label), f)
    print("Feature Engineered for site: {}".format(site))

def normalise_for_sites(pt_train, pt_test, save_train, save_test):
    sites = [site.split(".")[0] for site in os.listdir(pt_train)]
    for site in sites:
        min_max_normalise(pt_train,pt_test, site,save_train,save_test)


def eval_normalise_for_sites(pt_train, pt_test, save_train, save_test):
    sites = [site.split(".")[0] for site in os.listdir(pt_train)]
    for site in sites:
        evaluation_min_max_normalise(pt_train,pt_test, site,save_train,save_test)

def even_train(path_to_train, path_to_save):
    train_files = [x for x in os.listdir(path_to_train)]
    lst = list()
    for file in train_files:
        with open("{}/{}".format(path_to_train,file),"rb") as f:
            site, train, ground = pickle.load(f)
        print("Original data size : {}\n".format(len(train)))
        floor_map = dict()
        count = 0
        for x in ground:
            lst.append(x[2])
            floor_map[x[2]] = floor_map.get(x[2], []) + [count]  
            count += 1
        
        indices_to_be_moved = list()
        min_floors = 99999999
        for floor in floor_map:
            if len(floor_map[floor])<min_floors:
                min_floors =len(floor_map[floor])
        for floor in floor_map:
            if floor == 6:
                print(len(floor_map[floor][:min_floors]))
            indices_to_be_moved.extend(floor_map[floor][:min_floors])
        indices_to_be_moved = set(indices_to_be_moved)
        
        sub_train,sub_ground = list(),list()
        
        for i, (x,y) in enumerate(zip(train,ground)):
            if i in indices_to_be_moved:
                sub_train.append(x)
                sub_ground.append(y)
        print("New data size : {}\n".format(len(sub_train)))
        
        with open("{}/{}.pickle".format(path_to_save,site),"wb") as f:
            pickle.dump((site, sub_train,sub_ground),f)
        
def check_no_rssi(train_path):
    files = [f for f in os.listdir(train_path)]
    for file in files:
        with open("{}/{}".format(train_path, file), "rb") as f:
            site, train, label = pickle.load(f)
        total = 0
        for (train_sample, label_x) in zip(train, label):
            if ((train_sample == -999).sum() == len(train_sample)):
                total+=1
        print("Site: {} has {} with not rssi\n".format(site,total))

def get_floors(path):
    val_files = [x for x in os.listdir(path)]
    floors =list()
    for file in val_files:
        with open("{}/{}".format(path,file),"rb") as f:
            _, _, ground = pickle.load(f)
        for x in ground:
            floors.append(x[2])
    print(set(floors))

def get_second_half(path, save_path, val):
    train_files = [x for x in os.listdir(path)]
    for file in train_files:
        with open("{}/{}".format(path,file),"rb") as f:
            site, train, ground = pickle.load(f)
        
        with open("{}/{}.pickle".format(save_path,site),"wb") as f:
            print("New file size : {}".format(len(train[round(0.5*len(train)):round(0.5*len(train))+val])))
            pickle.dump((site, train[round(0.5*len(train)):round(0.5*len(train))+val],ground[round(0.5*len(train)):round(0.5*len(train))+val]),f)
if __name__ == "__main__":
    
    pass
