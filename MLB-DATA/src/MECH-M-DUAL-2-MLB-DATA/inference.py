#!/home/pekandolf/.local/bin/pdm run
from data import load_cats_vs_dogs
from myio import load_skops as load
import logging
from pathlib import Path

logging.basicConfig(format='%(levelname)s:%(name)s: %(message)s',
                    level=logging.DEBUG)
_, _, X_test, y_test = load_cats_vs_dogs()

file = Path(".") / "models" / "model"

logging.debug("Load classifier")
voting_clf = load(file)

# not Working for onnx
logging.debug("Score classifier")
score = voting_clf.score(X_test, y_test)

logging.info(f"We have a hard voting score of {score}")
