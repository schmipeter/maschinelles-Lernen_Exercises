import numpy as np
import scipy
import logging
from MECH_M_DUAL_2_MLB_DATA.myio import check_or_download


def load_cats_vs_dogs():
    cats_w = scipy.io.loadmat(check_or_download(
        "https://github.com/dynamicslab/databook_python/"
        "raw/refs/heads/master/DATA/catData_w.mat").absolute())["cat_wave"]

    dogs_w = scipy.io.loadmat(check_or_download(
        "https://github.com/dynamicslab/databook_python/"
        "raw/refs/heads/master/DATA/dogData_w.mat").absolute())["dog_wave"]

    X_train = np.concatenate((cats_w[:60, :], dogs_w[:60, :]))
    y_train = np.repeat(np.array([1, -1]), 60)
    X_test = np.concatenate((cats_w[60:80, :], dogs_w[60:80, :]))
    y_test = np.repeat(np.array([1, -1]), 20)
    logging.debug("Loaded the data with Split of 60 to 20 per category.")

    return X_train, y_train, X_test, y_test
