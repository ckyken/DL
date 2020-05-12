import pandas as pd
import numpy as np
import statistics
import matplotlib.pylab as plt
import math
from sklearn.model_selection import train_test_split

origin_data = pd.read_csv('original_covid_19.csv',  usecols=lambda column: column not in [
                          "Lat", "Long"], skiprows=[1, 2])
origin_data_numonly = pd.read_csv('original_covid_19.csv',  usecols=lambda column: column not in [
                                  "Lat", "Long", "Unnamed: 0"], skiprows=[1, 2])

country = origin_data['Unnamed: 0']

cor = []
for i in range(len(country) - 1):
    list1 = [0] * (i + 1)
    for j in range(i + 1, len(country), 1):
        cov_mat = np.cov(
            origin_data_numonly.iloc[i], origin_data_numonly.iloc[j])
        c = cov_mat[0, 1] / (math.sqrt(cov_mat[0, 0])
                             * math.sqrt(cov_mat[1, 1]))
        list1.append(c)
    cor.append(list1)
last = [0] * 185
cor.append(last)
cor_mat = np.matrix(cor)

dat = cor_mat.T

fig_size = 20
fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=100)
plt.imshow(dat, interpolation='none')
ax = plt.gca()
ax.set_xticks(np.arange(0, 184, 1))
ax.set_yticks(np.arange(0, 184, 1))
ax.set_yticklabels(country)
ax.set_xticklabels(country)

ax.imshow(dat, interpolation='none')


# clb = plt.colorbar()
# clb.set_label('label', labelpad=-40, y=1.05, rotation=0)

plt.show()
