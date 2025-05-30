"""
Exercise 5.3 (advanced) – Mehrschrittiges Training mit Hyper-Sweep.

* iteriert über verschiedene PCA-n_components-Werte
* baut pro Durchgang EIN neues Modell auf Basis derselben params.yaml
* loggt pro Durchgang einen Punkt und ruft anschließend live.next_step()
  → dadurch entstehen richtige Kurven in dvclive / dvc plots
* speichert das zuletzt gebaute Modell als Artefakt
"""

from pathlib import Path
import logging
from copy import deepcopy
from typing import List

from dvclive import Live
from omegaconf import OmegaConf

from MECH_M_DUAL_2_MLB_DATA.data import load_cats_vs_dogs
from MECH_M_DUAL_2_MLB_DATA import build_model           # nimmt OmegaConf
from MECH_M_DUAL_2_MLB_DATA.myio import save_skops       # (clf, path, x)

logging.basicConfig(level=logging.DEBUG,
                    format="%(levelname)s:%(name)s: %(message)s")


def sweep_components(n_list: List[int]) -> None:
    """Trainiert für jede Komponentenzahl ein neues Modell und loggt Scores."""
    # ---------------------------------------------------------------- data
    X_train, y_train, X_test, y_test = load_cats_vs_dogs()

    # Ursprungs-Konfiguration laden
    base_cfg = OmegaConf.load("params.yaml")

    with Live(dir="dvclive", resume=True) as live:
        for step, n in enumerate(n_list):
            logging.info("Step %d  –  PCA n_components=%d", step, n)

            # --------------------------------------------------------- Model
            cfg = deepcopy(base_cfg)
            cfg.PCA.init_args.n_components = n
            clf = build_model(cfg)         # Factory nimmt die veränderte cfg

            clf.fit(X_train, y_train)

            # ------------------------------------------------------- Logging
            live.log_metric("trainscore", clf.score(X_train, y_train))
            live.log_metric("testscore",  clf.score(X_test,  y_test))
            live.log_param("n_components", n)   # erscheint als Spalte in dvc exp
            live.next_step()                    # ← erzeugt neuen Plot-Punkt

        # letztes Modell sichern (oder: bestes auswählen)
        out = Path("dvclive/artifacts")
        out.mkdir(parents=True, exist_ok=True)
        save_skops(clf, out / f"model_pca{n_list[-1]}.skops", X_train[:1])


def main() -> None:
    # Beispiel-Sweep: 5, 10, 20, 30, 41 Komponenten
    sweep_components([5, 10, 20, 30, 41])


if __name__ == "__main__":
    main()