import matplotlib.pyplot as plt
#import matplotlib.ticker as ticker
import numpy as np
import json

# %matplotlib inline

with open("HW2_cross_entropy_loss.json", mode="r") as stream:
    data = json.load(stream)

plt.figure()
plt.plot(list(range(1, len(data)+1)), data)
plt.title('Learning_curve')
plt.xlabel('Epochs')
plt.ylabel('cross entropy')
plt.show()
