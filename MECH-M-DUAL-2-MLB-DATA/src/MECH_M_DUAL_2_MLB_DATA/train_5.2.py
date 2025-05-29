from MECH_M_DUAL_2_MLB_DATA.git_safeguards import require_clean_repo
from MECH_M_DUAL_2_MLB_DATA import build_model          # ← neu!

from data import load_cats_vs_dogs
from myio import save_skops as save

import logging
from pathlib import Path

logging.basicConfig(format="%(levelname)s:%(name)s: %(message)s",
                    level=logging.DEBUG)


@require_clean_repo(check_remote=True)
def main() -> None:
    X_train, y_train, X_test, y_test = load_cats_vs_dogs()

    logging.debug("Create model from YAML")
    voting_clf = build_model()          # ← Modell/Pipeline aus params.yaml

    logging.debug("Train classifier")
    voting_clf.fit(X_train, y_train)

    logging.debug("Score classifier")
    score = voting_clf.score(X_test, y_test)
    logging.info(f"We have a hard voting score of {score}")

    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    save(voting_clf, model_dir / "model", X_train[:1])


if __name__ == "__main__":
    main()