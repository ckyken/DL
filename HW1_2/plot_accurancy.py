import matplotlib.pyplot as plt
#import matplotlib.ticker as ticker
import numpy as np
import json

# %matplotlib inline

with open("HW2_Training_accurancy.json", mode="r") as stream:
    data_train = json.load(stream)

with open("HW2_Testing_accurancy.json", mode="r") as stream:
    data_test = json.load(stream)

plt.figure()
plt.plot(list(range(1, len(data_train) + 1)), data_train)
plt.plot(list(range(1, len(data_test)+1)), data_test)
plt.title('Accurancy')
plt.xlabel('Epochs')
plt.ylabel('Accurancy')
plt.show()
