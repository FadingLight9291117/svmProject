import random
from pathlib import Path
import json

from libs.metrics import metric, get_metrics

import numpy as np
from libs.wordvector import sentence_vec
from sklearn import svm

TRAIN_TEST_SPLIT = 0.8


def data_split(rate=TRAIN_TEST_SPLIT, shuffle=True):
    origin_data_path = Path('./data/data.json')

    print(origin_data_path)

    data = None
    with origin_data_path.open('r', encoding='utf-8') as f:
        data = json.load(f)

    if shuffle:
        random.shuffle(data)
    train_len = int(len(data) * rate)
    train_data = data[:train_len]
    test_data = data[train_len:]
    return train_data, test_data


def dataloader():
    train_data, test_data = data_split()

    get_sentence_vec = sentence_vec()

    def gen_data(data):
        for i, v in enumerate(data):
            sentence = v['text']
            label = v['sentiment']
            label = 1 if label == 'positive' else 0
            sentence_vec = get_sentence_vec(sentence)
            yield sentence_vec, label

    return gen_data(train_data), gen_data(test_data)


def get_train_data():
    train_data, _ = dataloader()
    train_data = list(train_data)

    # data
    X = np.zeros((len(train_data), train_data[0][0].size))
    y = np.zeros((len(train_data)))

    for i, v in enumerate(train_data):
        X[i] += v[0]
        y[i] += v[1]

    return X, y


def get_test_data():
    _, test_data = dataloader()
    test_data = list(test_data)

    # data
    X = np.zeros((len(test_data), test_data[0][0].size))
    y = np.zeros((len(test_data)))

    for i, v in enumerate(test_data):
        X[i] += v[0]
        y[i] += v[1]

    return X, y


def train():
    # svm train
    train_X, train_y = get_train_data()
    clf = svm.SVC(kernel='rbf', verbose=True)
    clf.fit(train_X, train_y)

    # test
    test_X, test_y = get_test_data()
    score_ploy = clf.score(train_X, train_y)

    print(score_ploy)

    pred = clf.predict(test_X)

    return get_metrics(*metric(pred, test_y))


if __name__ == '__main__':
    metrics = train()
