from omegaconf import OmegaConf
from pathlib import Path
import logging

from dvclive import Live
from MECH_M_DUAL_2_MLB_DATA.data import load_cats_vs_dogs
from MECH_M_DUAL_2_MLB_DATA.model_factory import build_model   # deine neue Factory

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")

def sweep_components(components):
    """führt mehrere PCA-Varianten (partial-fit) in einem Lauf durch"""
    Xtr, ytr, Xte, yte = load_cats_vs_dogs()

    for step, n in enumerate(components):
        log.info("Step %d  –  PCA n_components=%s", step, n)

        cfg = OmegaConf.load("params.yaml")

        # <- ACHTUNG:   Models.PCA  statt  PCA
        cfg.Models.PCA.init_args.n_components = n

        clf = build_model(cfg)          # Factory akzeptiert OmegaConf-Objekt

        with Live(dir="dvclive_advanced", resume=True) as live:
            clf.fit(Xtr, ytr)
            live.log_metric("trainscore", clf.score(Xtr, ytr))
            live.log_metric("testscore",  clf.score(Xte, yte))
            live.next_step()            # dvclive + DVC → ein Punkt pro Step

def main():
    sweep_components([5, 10, 20, 30, 41])

if __name__ == "__main__":
    main()