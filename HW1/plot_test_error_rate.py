import matplotlib.pyplot as plt
#import matplotlib.ticker as ticker
import numpy as np
import json

# %matplotlib inline

with open("Test_error_rate-1.json", mode="r") as stream:
    data = json.load(stream)

plt.figure()
plt.plot(list(range(1, len(data)+1)), data)
plt.title('Test_error_rate')
plt.xlabel('Epochs')
plt.ylabel('Test_error_rate')
plt.show()
