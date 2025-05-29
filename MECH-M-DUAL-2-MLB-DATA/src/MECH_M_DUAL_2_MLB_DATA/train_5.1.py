# src/MECH-M-DUAL-2-MLB-DATA/train.py
#
# Modelltraining + Persistenz – nur wenn das Git-Repo sauber ist
# --------------------------------------------------------------
# Hinweis:
#   * Der Dekorator @require_clean_repo verhindert den Start,
#     sobald uncommittete Änderungen oder untracked Dateien existieren
#     (bzw. wenn der Branch hinter Upstream liegt, weil check_remote=True).
#   * Aufgerufen wird das Training weiterhin via:
#         pdm run src/MECH-M-DUAL-2-MLB-DATA/train.py
# ---------------------------------------------------------------------

from MECH_M_DUAL_2_MLB_DATA.git_safeguards import require_clean_repo

from model import get_model
from data import load_cats_vs_dogs
from myio import save_skops as save

import logging
from pathlib import Path


logging.basicConfig(
    format="%(levelname)s:%(name)s: %(message)s",
    level=logging.DEBUG,
)

# ---------------------------------------------------------------------
# Hauptfunktion
# ---------------------------------------------------------------------

@require_clean_repo(check_remote=True)          # <— Git-Schutz aktivieren
def main() -> None:
    """
    Lädt die Katzen-vs-Hunde-Daten, trainiert das Voting-Ensemble
    und legt das Modell als *.skops* Datei im Ordner *models/* ab.
    """

    # Daten laden
    X_train, y_train, X_test, y_test = load_cats_vs_dogs()

    # Modell erstellen
    voting_clf = get_model()

    logging.debug("Train classifier")
    voting_clf.fit(X_train, y_train)

    logging.debug("Score classifier")
    score = voting_clf.score(X_test, y_test)
    logging.info(f"We have a hard voting score of {score}")

    # Persistenz
    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)

    model_file = model_dir / "model"         # .skops wird von save() angehängt
    save(voting_clf, model_file, X_train[:1])


# ---------------------------------------------------------------------
# Skriptein­stieg
# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()