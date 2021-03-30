import matplotlib.pyplot as plt
import pickle, os


#Cannot be used anyway
def get_train_val_values(metric, path):
    train = list()
    val = list()
    val_count = 0
    data_files = [x for x in os.listdir(path)]
    for data in data_files:
        with open("{path}/{data}".format(path=path, data=data), "rb") as f:
            hist = pickle.load(f)
            if len(train) == 0:
                train.extend(hist[metric])
                val.extend(hist["val_{met}".format(met=metric)])
            else:
                for x in range(len(hist[metric])):
                    train[x] = train[x] + hist[metric][x]
                    val_count = val_count + 1
                for x in range(len(hist["val_{met}".format(met=metric)])):
                    val[x] = val[x] + hist["val_{met}".format(met=metric)][x]
    train = [x/val_count for x in train]
    val = [x/val_count for x in val]
    return train, val

           
    
#list of train values, caption of train values, train
def plot(train, train_label, val, val_label, title):
   epochs = range(1, len(train) + 1)

   plt.plot(epochs, train, 'bo-', label=train_label)
   plt.plot(epochs, val, 'go-', label=val_label)
   plt.xlabel("Epoch")
   plt.ylabel("Mean squared Error")
   plt.title(title)
   plt.legend()
   #plt.figure()
   plt.savefig("{title}.png".format(title=title))
   #plt.show()
   

train, val = get_train_val_values("loss","C:/Users/Abiram Mohanaraj/Documents/GitHub/experiments/results/NN02")
plot(train,"training loss", val, "validation loss", "only recorded on last fold")
