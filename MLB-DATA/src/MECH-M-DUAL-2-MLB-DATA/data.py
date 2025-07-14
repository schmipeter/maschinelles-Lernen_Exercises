import numpy as np
import logging
import etl


def load_cats_vs_dogs():
    cats_w = etl.load("catData_w.mat")
    dogs_w = etl.load("dogData_w.mat")

    n = min(cats_w.shape[0], dogs_w.shape[0])  # 20 pro Klasse verfügbar
    n_test = 5  # min. 1 für Score-Aufruf
    n_train = n - n_test  # 15  Train-Bilder/Klasse

    X_train = np.concatenate((cats_w[:n_train], dogs_w[:n_train]))
    y_train = np.repeat(np.array([1, -1]), n_train)

    X_test = np.concatenate((cats_w[n_train:n], dogs_w[n_train:n]))
    y_test = np.repeat(np.array([1, -1]), n_test)

    logging.debug(f"train={n_train},  test={n_test}  pro Klasse")
    return X_train, y_train, X_test, y_test
