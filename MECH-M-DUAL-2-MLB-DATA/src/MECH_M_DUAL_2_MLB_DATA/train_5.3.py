from pathlib import Path
import logging
from dvclive import Live

from MECH_M_DUAL_2_MLB_DATA.data import load_cats_vs_dogs
from MECH_M_DUAL_2_MLB_DATA import build_model          # YAML-Factory
from MECH_M_DUAL_2_MLB_DATA.myio import save_skops

logging.basicConfig(level=logging.DEBUG,
                    format="%(levelname)s:%(name)s: %(message)s")

def main() -> None:
    Xtr, ytr, Xte, yte = load_cats_vs_dogs()
    clf = build_model()

    with Live(dir="dvclive", resume=True) as live:
        clf.fit(Xtr, ytr)
        live.log_metric("trainscore", clf.score(Xtr, ytr))
        live.log_metric("testscore",  clf.score(Xte, yte))

        out = Path("dvclive/artifacts")
        out.mkdir(parents=True, exist_ok=True)
        save_skops(clf, out / "model.skops", Xtr[:1])   # ein Sample reicht

if __name__ == "__main__":
    main()