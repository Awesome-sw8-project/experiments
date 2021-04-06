import os, pickle

def get_x_y_floor(truth_array):
    xs = list()
    ys = list()
    floors = list()
    for lst in truth_array:
        xs.append(lst[0])
        ys.append(lst[1])
        floors.append(lst[2])
    return xs,ys,floors

def gen_for_serialisation(path_to_train):
    site_data_files = [x for x in os.listdir(path_to_train)]
    for file in site_data_files:
        #../input/indoor-positioning-traindata/train_data
        f = open(path_to_train +'/'+file, "rb")
        site, train, ground = pickle.load(f)
        f.close()
        yield site,train,ground

def get_data_for_test(pt_test,site):
    with open("{pt_test}/{site}.pickle".format(pt_test=pt_test,site=site), "rb") as f:
        test_data = pickle.load(f)
    
