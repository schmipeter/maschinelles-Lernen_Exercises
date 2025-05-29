"""inference.py
===============
Lädt das zuvor trainierte Modell (models/model.skops) und bewertet es
auf dem Cats‑vs‑Dogs‑Testset.
"""

from pathlib import Path
import logging

from MECH_M_DUAL_2_MLB_DATA.data import load_cats_vs_dogs
from MECH_M_DUAL_2_MLB_DATA.myio import load_skops as load

logging.basicConfig(level=logging.DEBUG,
                    format="%(levelname)s:%(name)s: %(message)s")


def main() -> None:
    X_train, y_train, X_test, y_test = load_cats_vs_dogs()

    model_path = Path("models/model.skops")
    logging.debug("Load classifier from %s", model_path)
    clf = load(model_path)

    score = clf.score(X_test, y_test)
    logging.info("We have a hard voting score of %s", score)


if __name__ == "__main__":
    main()
