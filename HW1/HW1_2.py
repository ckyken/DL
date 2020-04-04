

# load MNIST data
import ipdb
import argparse
import numpy as np
import h5py
import pickle as pkl
import json

parser = argparse.ArgumentParser()

# hyperparameters setting
parser.add_argument('--lr', type=float, default=0.5, help='learning rate')
parser.add_argument('--epochs', type=int, default=50,
                    help='number of epochs to train')
parser.add_argument('--n_x', type=int, default=784, help='number of inputs')
parser.add_argument('--n_h', type=int, default=64,
                    help='number of hidden units')
parser.add_argument('--n_h2', type=int, default=2,
                    help='number of hidden units')
parser.add_argument('--beta', type=float, default=0.9,
                    help='parameter for momentum')
parser.add_argument('--batch_size', type=int,
                    default=64, help='input batch size')

# parse the arguments
opt = parser.parse_args()


digits = 10
params = {"W1": np.random.randn(opt.n_h, opt.n_x) * np.sqrt(1. / opt.n_x),
          "b1": np.zeros((opt.n_h, 1)) * np.sqrt(1. / opt.n_x),
          "W2": np.random.randn(opt.n_h2, opt.n_h) * np.sqrt(1. / opt.n_h),
          "b2": np.zeros((opt.n_h2, 1)) * np.sqrt(1. / opt.n_h),
          "W3": np.random.randn(digits, opt.n_h2) * np.sqrt(1. / opt.n_h2),
          "b3": np.zeros((digits, 1)) * np.sqrt(1. / opt.n_h2)}
WEIGHT_FILE = 'weight.pkl'


def get_data():
    # MNIST_data = h5py.File("MNISTdata.hdf5", 'r')
    # x_train = np.float32(MNIST_data['x_train'][:])
    # y_train = np.int32(np.array(MNIST_data['y_train'][:, 0])).reshape(-1, 1)
    # x_test = np.float32(MNIST_data['x_test'][:])
    # y_test = np.int32(np.array(MNIST_data['y_test'][:, 0])).reshape(-1, 1)
    # MNIST_data.close()

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

    # stack together for next step
    X = np.vstack((x_train, x_test))
    y = np.vstack((y_train, y_test))

    # one-hot encoding

    examples = y.shape[0]
    y = y.reshape(1, examples)
    Y_new = np.eye(digits)[y.astype('int32')]
    Y_new = Y_new.T.reshape(digits, examples)

    # number of training set
    # m = 12000
    m = x_train.shape[0]
    m_test = X.shape[0] - m
    X_train, X_test = X[:m].T, X[m:].T
    Y_train, Y_test = Y_new[:, :m], Y_new[:, m:]

    # shuffle training set
    shuffle_index = np.random.permutation(m)
    X_train, Y_train = X_train[:, shuffle_index], Y_train[:, shuffle_index]

    return X_train, Y_train, X_test, Y_test


def sigmoid(z):
    """
    sigmoid activation function.

    inputs: z
    outputs: sigmoid(z)
    """
    s = 1. / (1. + np.exp(-z))
    return s


def compute_loss(Y, Y_hat):
    """
    compute loss function
    """
    L_sum = np.sum(np.multiply(Y, np.log(Y_hat)))
    m = Y.shape[1]
    L = -(1./m) * L_sum

    return L


def evaluation(predicts, golds):
    correct = 0
    total = len(predicts)
    for predict, gold in zip(predicts, golds):
        if predict == gold:
            correct += 1
    return correct / total


def feed_forward(X, params):
    """
    feed forward network: 2 - layer neural net

    inputs:
        params: dictionay a dictionary contains all the weights and biases

    return:
        cache: dictionay a dictionary contains all the fully connected units and activations
    """
    cache = {}

    # Z1 = W1.dot(x) + b1
    cache["Z1"] = np.matmul(params["W1"], X) + params["b1"]

    # A1 = sigmoid(Z1)
    cache["A1"] = sigmoid(cache["Z1"])

    # Z2 = W2.dot(A1) + b2
    cache["Z2"] = np.matmul(params["W2"], cache["A1"]) + params["b2"]

    # A2 = sigmoid(Z2)
    cache["A2"] = sigmoid(cache["Z2"])

    # Z3 = W3.dot(A2) + b3
    cache["Z3"] = np.matmul(params["W3"], cache["A2"]) + params["b3"]

    # A3 = softmax(Z3)
    cache["A3"] = np.exp(cache["Z3"]) / np.sum(np.exp(cache["Z3"]), axis=0)

    return cache


def back_propagate(X, Y, params, cache, m_batch):
    """
    back propagation

    inputs:
        params: dictionay a dictionary contains all the weights and biases
        cache: dictionay a dictionary contains all the fully connected units and activations

    return:
        grads: dictionay a dictionary contains the gradients of corresponding weights and biases
    """
    # error at last layer
    dZ3 = cache["A3"] - Y

    # gradients at last layer (Py2 need 1. to transform to float)
    dW3 = (1. / m_batch) * np.matmul(dZ3, cache["A2"].T)
    db3 = (1. / m_batch) * np.sum(dZ3, axis=1, keepdims=True)

    # ---

    # back propgate through second layer
    dA2 = np.matmul(params["W3"].T, dZ3)
    dZ2 = dA2 * sigmoid(cache["Z2"]) * (1 - sigmoid(cache["Z2"]))

    # gradients at second layer (Py2 need 1. to transform to float)
    dW2 = (1. / m_batch) * np.matmul(dZ2, cache["A1"])
    db2 = (1. / m_batch) * np.sum(dZ2, axis=1, keepdims=True)

    # ---

    # back propgate through first layer
    dA1 = np.matmul(params["W2"].T, dZ2)
    dZ1 = dA1 * sigmoid(cache["Z1"]) * (1 - sigmoid(cache["Z1"]))

    # gradients at first layer (Py2 need 1. to transform to float)
    dW1 = (1. / m_batch) * np.matmul(dZ1, X.T)
    db1 = (1. / m_batch) * np.sum(dZ1, axis=1, keepdims=True)

    grads = {"dW1": dW1, "db1": db1, "dW2": dW2,
             "db2": db2, "dW3": dW3, "db3": db3}

    return grads


if __name__ == "__main__":

    X_train, Y_train, X_test, Y_test = get_data()

    TrainError = []
    TestError = []

    # training
    for i in range(opt.epochs):

        # shuffle training set
        permutation = np.random.permutation(X_train.shape[1])
        X_train_shuffled = X_train[:, permutation]
        Y_train_shuffled = Y_train[:, permutation]

        batch_num = len(X_train) // opt.batch_size
        predicts = []
        golds = []
        predicts_test = []
        golds_test = []

        for j in range(batch_num):

            # get mini-batch
            begin = j * opt.batch_size
            end = min(begin + opt.batch_size, X_train.shape[1] - 1)
            X = X_train_shuffled[:, begin:end]
            Y = Y_train_shuffled[:, begin:end]
            m_batch = end - begin

            # forward and backward
            cache = feed_forward(X, params)
            grads = back_propagate(X, Y, params, cache, m_batch)

            dW1 = grads["dW1"]
            db1 = grads["db1"]
            dW2 = grads["dW2"]
            db2 = grads["db2"]
            dW3 = grads["dW3"]
            db3 = grads["db3"]

            # with momentum (optional)
            dW1 = (opt.beta * dW1 + (1. - opt.beta) * grads["dW1"])
            db1 = (opt.beta * db1 + (1. - opt.beta) * grads["db1"])
            dW2 = (opt.beta * dW2 + (1. - opt.beta) * grads["dW2"])
            db2 = (opt.beta * db2 + (1. - opt.beta) * grads["db2"])
            dW3 = (opt.beta * dW3 + (1. - opt.beta) * grads["dW3"])
            db3 = (opt.beta * db3 + (1. - opt.beta) * grads["db3"])

            # gradient descent
            params["W1"] = params["W1"] - opt.lr * dW1
            params["b1"] = params["b1"] - opt.lr * db1
            params["W2"] = params["W2"] - opt.lr * dW2
            params["b2"] = params["b2"] - opt.lr * db2
            params["W3"] = params["W3"] - opt.lr * dW3
            params["b3"] = params["b3"] - opt.lr * db3

        # forward pass on training set
        cache = feed_forward(X_train, params)
        train_loss = compute_loss(Y_train, cache["A3"])

        # evaluate_train
        predicts += np.argmax(cache["A3"], axis=0).tolist()
        golds += np.argmax(Y_train, axis=0).tolist()

        # forward pass on test set
        cache = feed_forward(X_test, params)
        test_loss = compute_loss(Y_test, cache["A3"])

        # evaluate_test
        predicts_test += np.argmax(cache["A2"], axis=0).tolist()
        golds_test += np.argmax(Y_test, axis=0).tolist()

        print("Epoch {}: training loss = {}, test loss = {}, Train_accur = {}".format(
            i + 1, train_loss, test_loss, evaluation(predicts, golds)))

        TrainError.append(1 - evaluation(predicts, golds))
        TestError.append(1 - evaluation(predicts_test, golds_test))

    with open(WEIGHT_FILE, 'wb') as stream:
        print("Saving weights")
        pkl.dump(params, stream)

    with open("Train_error_rate.json", mode="w") as stream:
        json.dump(TrainError, stream)

    with open("Test_error_rate.json", mode="w") as stream:
        json.dump(TestError, stream)
