import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Main entry function for CDF plotting.
def plot(data, size, title, x_label):
    x = np.sort(data)
    y = np.arange(size) / float(size)

    plt.xlabel(x_label)
    plt.ylabel("Distribution")
    plt.title(title)
    plt.plot(x, y, label = 'CDF', marker = 'o')
    plt.savefig("./" + title + ".png")
    plt.close()
