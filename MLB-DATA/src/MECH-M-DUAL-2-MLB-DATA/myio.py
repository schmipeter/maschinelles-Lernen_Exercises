import skops.io as sio
from pickle import dump, load
from skl2onnx import to_onnx
import sklearn
import onnxruntime as ort
import pathlib
import logging
import numpy as np


def save_onnx(clf: sklearn.utils._bunch.Bunch,
              file: pathlib.Path, x, protocol: int = 5) -> None:
    filename = file.with_suffix(".onnx")
    logging.debug(f"Save onnx to pickle file {filename}")
    onx = to_onnx(clf, x.astype(np.int64))
    with open(filename, "wb") as f:
        f.write(onx.SerializeToString())


def save_pickle(clf: sklearn.utils._bunch.Bunch,
                file: pathlib.Path, x, protocol: int = 5) -> None:
    filename = file.with_suffix(".pkl")
    logging.debug(f"Save clf to pickle file {filename}")
    with open(filename, "wb") as f:
        dump(clf, f, protocol=protocol)


def save_skops(clf: sklearn.utils._bunch.Bunch,
               file: pathlib.Path, x, protocol: int = 5) -> None:
    filename = file.with_suffix(".skops")
    logging.debug(f"Save clf to skops file {filename}")
    sio.dump(clf, filename)


def load_onnx(file: pathlib.Path) -> ort.InferenceSession:
    filename = file.with_suffix(".onnx")
    logging.debug(f"Load clf from onnx file {filename}")
    model = ort.InferenceSession(filename)
    return model


def load_pickle(file: pathlib.Path) -> sklearn.utils._bunch.Bunch:
    filename = file.with_suffix(".pkl")
    logging.debug(f"Load clf from pickle file {filename}")
    with open(filename, "rb") as f:
        clf = load(f)
    return clf


def load_skops(file: pathlib.Path) -> sklearn.utils._bunch.Bunch:
    filename = file.with_suffix(".skops")
    logging.debug(f"Load clf from skops file {filename}")
    unknown_types = sio.get_untrusted_types(file=filename)
    # investigate the contents of unknown_types, and only load if you trust
    # everything you see.
    for i, a in enumerate(unknown_types):
        logging.warning(f"Unknown type at {i} is {a}.")

    clf = sio.load(filename, trusted=unknown_types)
    return clf
