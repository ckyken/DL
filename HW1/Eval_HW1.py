from HW1_2 import feed_forward, get_data, WEIGHT_FILE, evaluation
import pickle as pkl
import numpy as np

if __name__ == "__main__":

    _, _, X_test, Y_test = get_data()

    with open(WEIGHT_FILE, 'rb') as stream:
        params = pkl.load(stream)

    cache = feed_forward(X_test, params)
    print('Evaluation on test:', evaluation(np.argmax(
        cache["A3"], axis=0).tolist(), np.argmax(Y_test, axis=0).tolist()))
