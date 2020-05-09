import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import math
from model import Classifier
from typing import List, Set
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from pygal.maps.world import COUNTRIES, World


INTERVAL = 20  # L
INPUT_SIZE = 1  # only one input number for each time step
HIDDEN_SIZE = 128
RANDOM_SEED = 87
# RNN_MODULE = nn.RNN
RNN_MODULE = nn.LSTM
# RNN_MODULE = nn.GRU


def load_data(data_path: str = 'covid_19.csv'):
    """ load table
    columns: countries
    index: date
    """
#     origin_data = pd.read_csv('covid_19.csv',  usecols=lambda column: column not in ["Lat", "Long"], skiprows=[1, 2], index_col = "Country")
#     origin_data_numonly = pd.read_csv('covid_19.csv',  usecols=lambda column: column not in ["Lat", "Long", "Unnamed: 0"], skiprows=[1, 2])

    origin_data = pd.read_csv('covid_19.csv',  usecols=lambda column: column not in [
                              "Lat", "Long"], index_col="Country")

    return origin_data


def get_country_set(table: pd.DataFrame, threshold: float = 0.9) -> Set[str]:
    """ compute corrilation and collect high correlation countries """
    country = table.index

    country_set = set()
    for i in range(len(table) - 1):
        for j in range(i + 1, len(table), 1):
            cov_mat = np.cov(
                table.iloc[i], table.iloc[j])
            c = cov_mat[0, 1] / (math.sqrt(cov_mat[0, 0])
                                 * math.sqrt(cov_mat[1, 1]))
            if c > 0.9:
                country_set.add(country[i])
                country_set.add(country[j])

    return country_set


def data_preprocessing(table: pd.DataFrame, country_set: Set[str], test_split: float = 0.1):
    """
    split train test set

    return a numpy array which each row are the data difference of each country
    """

    country_to_drop = set(table.index) - country_set
    table.drop(country_to_drop, inplace=True)
    diff_table = table.diff(axis=1).dropna(axis=1)
    all_data = np.array(diff_table)

    X_train, X_test = train_test_split(
        all_data, test_size=test_split, random_state=RANDOM_SEED, shuffle=False)

    return X_train, X_test, all_data, diff_table.index


def evaluation(gold: List[bool], predict: List[bool]):
    """ calculate accuracy """

    correct = 0
    total = len(gold)
    for y, y_hat in zip(gold, predict):
        if y == y_hat:
            correct += 1
    return correct / total


def train(model: nn.Module, loss_func, optimizer, data: np.array, test_data: np.array, epochs: int = 10, batch_size: int = 16, interval: int = INTERVAL):
    countries, days = data.shape
    iteration = 0
    loss = []
    for epoch in range(epochs):
        epoch_loss = 0
        predict = []
        gold = []

        for start in range(0, countries, batch_size):
            end = min(start + batch_size, len(data))
            iteration += end - start

            batch_data = data[start:end]

            for interval_start in range(days - interval):
                x = batch_data[:, interval_start: interval_start + interval]
                y = batch_data[:, interval_start + interval] > 0
                gold.extend(y.tolist())

                x = torch.from_numpy(x).permute(1, 0).unsqueeze(-1).float()
                y = torch.from_numpy(y).float()

                y_hat = model(x)

                predict.extend((y_hat.detach().numpy() > 0.5).tolist())

                optimizer.zero_grad()

                loss = loss_func(y_hat, y)
                epoch_loss += loss.item()
                loss.backward()

                optimizer.step()

        accuracy = evaluation(gold, predict)
        # print("epoch", epoch + 1, "loss:",
        #       epoch_loss / countries, "acc:", accuracy)

        model.eval()
        prob_predict, test_acc = test(model, test_data)
        model.train()

        print("epoch:", epoch + 1, "iteration:", iteration, "loss:",
              epoch_loss / countries, "acc:", accuracy, "acc_test:", test_acc)


def test(model: nn.Module, data: np.array, epochs: int = 10, interval: int = INTERVAL):
    prob_predict = []
    countries, days = data.shape
    iteration = 0
    for epoch in range(epochs):
        predict = []
        gold = []

        for interval_start in range(days - interval):
            x = data[:, interval_start: interval_start + interval]
            y = data[:, interval_start + interval] > 0
            gold.extend(y.tolist())

            x = torch.from_numpy(x).permute(1, 0).unsqueeze(-1).float()
            y = torch.from_numpy(y).float()

            y_hat = model(x)
            prob_predict.extend(y_hat.tolist())

            predict.extend((y_hat.detach().numpy() > 0.5).tolist())

    accuracy = evaluation(gold, predict)
    # print("epoch", epoch + 1, "loss:",
    #       epoch_loss / countries, "acc:", accuracy)

    return prob_predict, accuracy


def get_country_code(country_name: str):
    if country_name == 'Taiwan*':
        # country_name = 'Taiwan (Republic of China)'
        country_name = 'Taiwan, Province of China'  # fuck you
    elif country_name == 'Bolivia':
        country_name = 'Bolivia, Plurinational State of'
    elif country_name == 'Brunei':
        country_name = 'Brunei Darussalam'
    elif country_name == 'Burma':
        country_name = 'Myanmar'
    elif country_name == 'Cabo Verde':
        country_name = 'Cape Verde'
    elif country_name == 'Congo (Kinshasa)':
        country_name = 'Congo, the Democratic Republic of the'
    elif country_name == 'Congo (Brazzaville)':
        country_name = 'Congo'
    elif country_name == 'Czechia':
        country_name = 'Czech Republic'
    elif country_name == 'Eswatini':
        country_name = 'Swaziland'
    elif country_name == 'Holy See':
        country_name = 'Holy See (Vatican City State)'
    elif country_name == 'Iran':
        country_name = 'SIran, Islamic Republic of'
    elif country_name == 'Korea, South':
        country_name = 'Korea, Republic of'
    elif country_name == 'Laos':
        country_name = 'Lao People’s Democratic Republic'
    elif country_name == 'Libya':
        country_name = 'Libyan Arab Jamahiriya'
    elif country_name == 'Moldova':
        country_name = 'Moldova, Republic of'
    elif country_name == 'North Macedonia':
        country_name = 'Macedonia, the former Yugoslav Republic of'
    elif country_name == 'Russia':
        country_name = 'Russian Federation'
    elif country_name == 'South Sudan':
        country_name = 'Sudan'
    elif country_name == 'Syria':
        country_name = 'Syrian Arab Republic'
    elif country_name == 'Tanzania':
        country_name = 'Tanzania, United Republic of'
    elif country_name == 'Venezuela':
        country_name = 'Venezuela, Bolivarian Republic of'
    elif country_name == 'Vietnam':
        country_name = 'Viet Nam'
    elif country_name == 'West Bank and Gaza':
        country_name = 'Palestine, State of'
    elif country_name == 'US':
        country_name = 'United States'

    for code, name in COUNTRIES.items():
        if name == country_name:
            return code

    return country_name


def draw_word_map(prob_predict: List[float], countries: List[str], output_file: str = 'word_map.svg'):
    """
    > 0.5 accending => prob
    < 0.5 descending => 1 - prob
    """
    worldmap_chart = World()
    worldmap_chart.title = 'Motherfucker'
    accending = {}
    descending = {}
    for prob, country in zip(prob_predict, countries):
        country_code = get_country_code(country)
        if not country_code:
            continue

        if prob > 0.5:
            accending[country_code] = prob
        else:
            descending[country_code] = prob

    worldmap_chart.add('accending', accending)
    worldmap_chart.add('descending', descending)
    worldmap_chart.render_to_file(output_file)


if __name__ == "__main__":
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    loss_func = nn.BCELoss()
    model = Classifier(INPUT_SIZE, HIDDEN_SIZE, RNN_MODULE)
    optimizer = Adam(model.parameters(), lr=0.00001)

    table = load_data()
    country_set = get_country_set(table)

    train_data, test_data, all_data, countries = data_preprocessing(
        table, country_set)

    train(model, loss_func, optimizer, train_data, test_data)
    model.eval()
    prob_predict, _ = test(model, all_data, interval=all_data.shape[1]-1)
    draw_word_map(prob_predict, countries)
