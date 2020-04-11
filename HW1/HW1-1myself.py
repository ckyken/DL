# import argparse
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

digits = 10

test = np.load('test.npz')
train = np.load('train.npz')

y_test = test['label']
x_test = test['image']
y_train = train['label']
x_train = train['image']

x_train = x_train.reshape(-1, x_train.shape[1] * x_train.shape[2])
x_test = x_test.reshape(-1, x_test.shape[1] * x_test.shape[2])
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

X = np.vstack((x_train, x_test))
y = np.vstack((y_train, y_test))

examples = y.shape[0]
y = y.reshape(1, examples)
Y_new = np.eye(digits)[y.astype('int32')]
Y_new = Y_new.T.reshape(digits, examples)
# print(y)

m = x_train.shape[0]
X_train, X_test = X[:m].T, X[m:].T
Y_train, Y_test = Y_new[:, :m], Y_new[:, m:]

shuffle_index = np.random.permutation(m)
X_train, Y_train = X_train[:, shuffle_index], Y_train[:, shuffle_index]


def sigmoid(z):
    s = 1. / (1. + np.exp(-z))
    return s


def sigmoid_gradient(z):
    #     # To prevent from overflow
    #     z = np.clip(z, 1e-15, 1 - 1e-15)
    s = sigmoid(z) * (1 - sigmoid(z))
    return s


def cross_entropy(Y, Y_hat):
    L_sum = np.sum(np.multiply(Y, np.log(Y_hat)))
    m = Y.shape[1]
    L = -(1./m) * L_sum

    return L


def cross_entropy_gradient(Y, Y_hat):
    L = Y_hat - Y
    return L


def softmax(z):
    s = np.exp(z) / np.sum(np.exp(z), axis=0)
    return s


def evaluation(predicts, golds):
    correct = 0
    total = len(predicts)
    assert len(predicts) == len(golds)
    for predict, gold in zip(predicts, golds):
        if predict == gold:
            correct += 1
    accurancy = correct / total
    return accurancy


class Layer:
    def __init__(self, input_, output):
        self.input = input_
        self.output = output  # number of layer node
        self.W = np.random.randn(
            self.output, self.input) * np.sqrt(1. / self.input)
        self.b = np.zeros((self.output, 1)) * np.sqrt(1. / self.input)

    def forward(self, last_layer):
        self.last_layer = last_layer
        layer_output = np.matmul(self.W, self.last_layer) + self.b
#         layer_output = sigmoid(layer_output_temp)
        return layer_output

    def back_propagation(self, CE_gradientorgradient, m_batch, learning_rate):
        W_temp = self.W
        W_gradient = (1. / m_batch) * \
            np.matmul(CE_gradientorgradient, self.last_layer.T)
        b_gradient = (1. / m_batch) * \
            np.sum(CE_gradientorgradient, axis=1, keepdims=True)
        self.W_new = self.W - learning_rate * W_gradient
        self.b_new = self.b - learning_rate * b_gradient
        self.W = self.W_new
        self.b = self.b_new
        gradient_temp = np.matmul(W_temp.T, CE_gradientorgradient)
        return gradient_temp


hiddenlayer1 = Layer(784, 400)
hiddenlayer2 = Layer(400, 400)
outputlayer = Layer(400, 10)


def SGD_train_epoch(X_train, Y_train, batch_size=64, epoch=10, learning_rate=0.03):
    TrainError = []
    TestError = []
    features = []
    training_loss = []

    for i in range(epoch):

        # shuffle training set
        permutation = np.random.permutation(X_train.shape[1])
        X_train_shuffled = X_train[:, permutation]
        Y_train_shuffled = Y_train[:, permutation]

        batch_num = len(X_train) // batch_size
        predicts = []
        golds = []
        predicts_test = []
        golds_test = []
        epoch_loss = 0

        latent_features = []  # (output_node, label)

        if i + 1 in (20, 80):
            store_latent_feature = True
        else:
            store_latent_feature = False

        for j in range(batch_num):
            begin = j * batch_size
            end = min(begin + batch_size, X_train.shape[1] - 1)
            X = X_train[:, begin:end]
            Y = Y_train[:, begin:end]
            m_batch = end - begin

            output1_temp = hiddenlayer1.forward(X)
            output1 = sigmoid(output1_temp)
            output2_temp = hiddenlayer2.forward(output1)
            # if store_latent_feature:
            #     latent_features.extend(
            #         [np.hstack([output_node, np.argmax(y_temp, axis=0)]).tolist() for output_node, y_temp in zip(np.reshape(output2_temp, (-1, 2)), np.reshape(Y, (-1, 10)))])
            # import ipdb
            # ipdb.set_trace()
            output2 = sigmoid(output2_temp)

            y_hat_temp = outputlayer.forward(output2)
            y_hat = softmax(y_hat_temp)
            # print(y_hat)

            predicts += np.argmax(y_hat, axis=0).tolist()
            golds += np.argmax(Y, axis=0).tolist()

            loss = cross_entropy(Y, y_hat)
            epoch_loss += loss
            gradient = cross_entropy_gradient(Y, y_hat)

            back_output1 = outputlayer.back_propagation(
                gradient, m_batch, learning_rate)
            back_output2_temp = sigmoid_gradient(output2_temp) * back_output1
            back_output2 = hiddenlayer2.back_propagation(
                back_output2_temp, m_batch, learning_rate)
            back_output3_temp = sigmoid_gradient(output1_temp) * back_output2
            back_output3 = hiddenlayer1.back_propagation(
                back_output3_temp, m_batch, learning_rate)

            # ---------test data-----------

            output1_temp = hiddenlayer1.forward(X_test)
            output1 = sigmoid(output1_temp)
            output2_temp = hiddenlayer2.forward(output1)
            output2 = sigmoid(output2_temp)
            y_hat_temp = outputlayer.forward(output2)
            y_hat = softmax(y_hat_temp)
#             print(y_hat.shape)
            # print(y_hat)
#             print(Y_test.shape)
            predicts_test += np.argmax(y_hat, axis=0).tolist()
            golds_test += np.argmax(Y_test, axis=0).tolist()

        if store_latent_feature:
            features.append(latent_features)
#             loss_test = cross_entropy(Y_test, y_hat)

        print('Epoch : ', i + 1, 'training_loss = ', epoch_loss / len(Y_train), 'train_accur = ',
              evaluation(predicts, golds), 'test_accur = ', evaluation(predicts_test, golds_test))

        TrainError.append(1 - evaluation(predicts, golds))
        TestError.append(1 - evaluation(predicts_test, golds_test))
        training_loss.append((epoch_loss / len(Y_train)))

    # with open("features.json", mode="w") as stream:
    #     json.dump(features, stream)

    # with open("predicts_test.json", mode="w") as stream:
    #     json.dump(predicts_test, stream)

    # with open("golds_test.json", mode="w") as stream:
    #     json.dump(golds_test, stream)
    # with open("Train_error_rate.json", mode="w") as stream:
    #     json.dump(TrainError, stream)

    # with open("Test_error_rate.json", mode="w") as stream:
    #     json.dump(TestError, stream)

    with open("Training_loss.json", mode="w") as stream:
        json.dump(training_loss, stream)


SGD_train_epoch(X_train, Y_train, batch_size=64, epoch=500, learning_rate=0.03)
