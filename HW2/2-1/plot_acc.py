import ipdb
import matplotlib.pyplot as plt
import numpy as np
import json

# with open("train_acc_LSTM.json", mode='r') as stream:
#     data_train = json.load(stream)
# with open("test_acc_LSTM.json", mode='r') as stream:
#     data_test = json.load(stream)
# with open("train_acc_GRU.json", mode='r') as stream:
#     data_train = json.load(stream)
# with open("test_acc_GRU.json", mode='r') as stream:
#     data_test = json.load(stream)
with open("train_acc_RNN.json", mode='r') as stream:
    data_train = json.load(stream)
with open("test_acc_RNN.json", mode='r') as stream:
    data_test = json.load(stream)

image_1 = plt.figure()

iteration = list()
for i in range(10):
    iteration.append((i + 1) * 165)
# ipdb.set_trace()
plt.plot(iteration, data_train, label='train')
plt.plot(iteration, data_test, label='test')
plt.legend()
# plt.title('LSTM_Accurancy')
# plt.title('GRU_Accurancy')
plt.title('RNN_Accurancy')
plt.xlabel('iteration')
plt.ylabel('Accurancy')
# plt.savefig('LSTM.png')
# plt.savefig('GRU.png')
plt.savefig('RNN.png')
plt.show()
