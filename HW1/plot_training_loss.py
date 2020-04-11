import matplotlib.pyplot as plt
#import matplotlib.ticker as ticker
import numpy as np
import json

# %matplotlib inline

with open("Training_loss-1.json", mode="r") as stream:
    data = json.load(stream)

plt.figure()
plt.plot(list(range(1, len(data)+1)), data)
plt.title('Training_loss')
plt.xlabel('Epochs')
plt.ylabel('Training_loss')
plt.show()
