import numpy as np
import matplotlib.pyplot as plt
import json

with open("features.json", mode="r") as stream:
    data = json.load(stream)

def plot_scatter(epoch_data):
    epoch_data = np.array(epoch_data)
    scatter = plt.scatter(epoch_data[:, 0], epoch_data[:, 1], c=epoch_data[:, 2])
    plt.legend(*scatter.legend_elements())
    # label=np.array([str(i) for i in range(10)])[epoch_data[:, 2].astype(int)]


#plot_scatter(data[0])
#plt.show()
plot_scatter(data[1])
plt.show()
