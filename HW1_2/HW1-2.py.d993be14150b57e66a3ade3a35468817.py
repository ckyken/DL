import pandas as pd
from skimage.io import imread
from skimage import io, data
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from skimage.transform import resize
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout, Softmax
from torch.optim import Adam, SGD

trainData = pd.read_csv('problem2-CNN/train.csv')
testData = pd.read_csv('problem2-CNN/test.csv')

img_num = trainData['filename']
img_num_test = testData['filename']
Y_train_data = trainData['label']
Y_test_data = testData['label']

train_data = []

for i in tqdm(range(len(img_num))):
    filename = trainData['filename'][i]
    image_path = 'problem2-CNN/images/' + str(filename)
    img = imread(image_path)
    if img.shape[-1] > 3:
        # discard alpha channel
        img = img[:, :, :3]
    xmin = trainData['xmin'][i]
    xmax = trainData['xmax'][i]
    ymin = trainData['ymin'][i]
    ymax = trainData['ymax'][i]
    #   切割與resize
    partial_img_temp = img[ymin:ymax, xmin:xmax, :]
    partial_img = resize(partial_img_temp, (64, 64))

    partial_img = np.reshape(partial_img, (1, 3, 64, 64))
    #   貌似normalize過惹
        # 解決data unbalabce
    if Y_train_data[i] == "bad":
        for i in range(4):
          train_data.append(partial_img)
    elif Y_train_data[i] == "none":
        for i in range(20):
            train_data.append(partial_img)
        # 解決data unbalabce
    else:
        train_data.append(partial_img)
#     io.imshow(partial_img)
#     io.show()

test_data = []

for i in tqdm(range(len(img_num_test))):
    filename = testData['filename'][i]
    image_path = 'problem2-CNN/images/' + str(filename)
    img = imread(image_path)
    if img.shape[-1] > 3:
        # discard alpha channel
        img = img[:, :, :3]
    xmin = testData['xmin'][i]
    xmax = testData['xmax'][i]
    ymin = testData['ymin'][i]
    ymax = testData['ymax'][i]
    #   切割與resize
    partial_img_temp = img[ymin:ymax, xmin:xmax, :]
    partial_img = resize(partial_img_temp, (64, 64))
    partial_img = np.reshape(partial_img, (1, 3, 64, 64))
    #   貌似normalize過惹
    test_data.append(partial_img)


def evaluation(predicts, golds):
    correct = 0
    total = len(predicts)
    for predict, gold in zip(predicts, golds):
        if predict == gold:
            correct += 1
    return correct / total

 class Net(nn.Module):
        def __init__(self, D_in = 4 * 16 * 16, H = 100, D_out = 3):
            super(Net, self).__init__()
            self.CNN = nn.Sequential(
                    # Defining a 2D convolution layer
                Conv2d(3, 4, kernel_size=3, stride=1, padding=1),
                BatchNorm2d(4),
                ReLU(inplace=True),
                MaxPool2d(kernel_size=2, stride=2),
                    # Defining another 2D convolution layer
                Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
                BatchNorm2d(4),
                ReLU(inplace=True),
                MaxPool2d(kernel_size=2, stride=2),

            )
            self.DNN = nn.Sequential(
                nn.Linear(D_in, H),
                nn.ReLU(),
                nn.Linear(H, D_out),

            )
#             self.softmax = nn.Softmax

        def forward(self, x):
            x = self.CNN(x)
            x = x.view(x.shape[0], -1)
            x = self.DNN(x)
#             x = self.softmax(x)
            return x

loss_func = nn.CrossEntropyLoss(reduction='sum')
model = Net()
optimizer = Adam(model.parameters(), lr=0.0001)

Y_train_data_list = []
for i in range(len(Y_train_data)):
    if Y_train_data[i] == "good":
        Y_train_data_list.append(np.array([0]))
    elif Y_train_data[i] == "bad":
        for i in range(4):
            Y_train_data_list.append(np.array([1]))
    else:
        for i in range(20):
            Y_train_data_list.append(np.array([2]))
Y_train = np.vstack(Y_train_data_list)

Y_train = np.reshape(Y_train, (-1))

Y_test_data_list = []
for i in range(len(Y_test_data)):
    if Y_test_data[i] == "good":
        Y_test_data_list.append(np.array([0]))
    elif Y_test_data[i] == "bad":
        Y_test_data_list.append(np.array([1]))
    else:
        Y_test_data_list.append(np.array([2]))
Y_test = np.vstack(Y_test_data_list)

Y_test = np.reshape(Y_test, (-1))

# 因為用append所以是很多np.array, 所以用vstack改成單一np.array
train_data = np.vstack(train_data)
test_data = np.vstack(test_data)

def train_epoch(model, train_data, Y_train, test_data, Y_test, batch_size = 64, epoch = 10):
    model.train()
    test_data = torch.from_numpy(test_data).float()
    Y_test = torch.from_numpy(Y_test)
    loss_append = []
    train_accur = []
    for i in range(epoch):
        permutation = np.random.permutation(train_data.shape[0])
        train_data_shuffled = train_data[permutation, :, :, :]
        Y_train_shuffled = Y_train[permutation]
        epoch_loss = 0
        batch_num = len(train_data) // batch_size
        predicts = []
        predicts_test = []
        golds = []


        for j in range(batch_num):
            model.train(mode=True)
            begin = j * batch_size
            end = min(begin + batch_size, train_data.shape[0] - 1)
            X = train_data_shuffled[begin:end, :, :, :]
#             print(X[0])
            Y = Y_train_shuffled[begin:end]
            m_batch = end - begin

            # converting validation images into torch format
            X = torch.from_numpy(X).float()
            Y = torch.from_numpy(Y)



            # X_afterCNN = CNN(X)
            # X_afterCNN = X_afterCNN.view(X.shape[0], -1)
            # X_afterDNN = DNN(X_afterCNN)

            X_afterDNN = model(X)
            y_hat = torch.softmax(X_afterDNN, dim=1)

            # ------test data-------------
            model.eval()

            test_result_temp = model(test_data)
            test_result = torch.softmax(test_result_temp, dim=1)
            # print(X_afterDNN)
            # print(Y)

            loss = loss_func(y_hat, Y)
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

#             print(y_hat)
#             print(test_result)


            golds += Y.tolist()
            predicts += torch.argmax(y_hat, dim=1).tolist()

#         print("predict : ", predicts)

        predicts_test += torch.argmax(test_result, dim=1).tolist()
        loss_append.append(epoch_loss)
        train_accur.append(evaluation(predicts, golds))

#         print("gold : ", golds)
#         print(predicts)
#         print(Y_train)
#         golds += np.argmax(Y_train, axis=).tolist()
        print("epoch : ", i + 1, "loss : ", epoch_loss / train_data.shape[0], "train_accur : "
              , evaluation(predicts, golds), "test_accur : ", evaluation(predicts_test, Y_test))
    torch.save(model.state_dict(), problem2-CNN/state_dict/model_state_dict)
    torch.save(optimizer.state_dict(),)

    with open("HW2_cross_entropy_loss.json", mode="w") as stream:
        json.dump(TrainError, stream)
    with open("HW2_Training_accurancy.json", mode="w") as stream:
        json.dump(train_accur, stream)



