# src/MECH_M_DUAL_2_MLB_DATA/model_factory.py
from __future__ import annotations
from pathlib import Path
from importlib import import_module
from typing import Any, Dict

from omegaconf import OmegaConf, DictConfig
from sklearn.pipeline import Pipeline

import logging
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------

def _import_class(qualname: str):
    """ 'sklearn.ensemble.RandomForestClassifier'  →  Klasse """
    mod, cls = qualname.rsplit(".", 1)
    return getattr(import_module(mod), cls)

def _instantiate(name: str, cfg_section: DictConfig, registry: dict[str, Any]):
    """
    - importiert die Klasse
    - baut optionale **kwargs  aus init_args
    - baut ein optionales *positional* Argument „estimators“
      (benötigt z. B. VotingClassifier)
    """
    cls = _import_class(cfg_section.type)
    kwargs = OmegaConf.to_container(cfg_section.get("init_args", {}), resolve=True)

    pos_args: list[Any] = []
    if "estimators" in cfg_section:                       # <-- NEU
        missing = [e for e in cfg_section.estimators if e not in registry]
        if missing:
            raise KeyError(
                f"Estimator(s) {missing!r} erst nach {name!r} definieren "
                "oder Reihenfolge in params.yaml korrigieren."
            )
        est_list = [(e, registry[e]) for e in cfg_section.estimators]
        pos_args.append(est_list)                         # erste Position

    obj = cls(*pos_args, **kwargs)                        # ✔ jetzt korrekt
    registry[name] = obj
    log.debug("Instanziiert %-25s -> %s", name, obj)
# ---------------------------------------------------------------------------

def _pipeline_from_cfg(cfg: DictConfig) -> Pipeline:
    """Erzeugt eine sklearn-Pipeline aus vollständigem OmegaConf."""
    # 1) alle Modelle instanziieren und merken
    registry: dict[str, Any] = {}
    for name, section in cfg.Models.items():
        _instantiate(name, section, registry)

    # 2) Schritte gemäß Pipeline-Reihenfolge zusammensetzen
    steps = [(name, registry[name]) for name in cfg.Pipeline]
    pipe = Pipeline(steps)
    log.debug("Fertige Pipeline: %s", pipe)
    return pipe

# ---------------------------------------------------------------------------

def build_model(config: str | Path | DictConfig | Dict[str, Any] | None = None) -> Pipeline:
    """
    * **ohne Argument**   →  params.yaml im Projekt
    * **Pfad/Dateiname** →  YAML laden
    * **OmegaConf-Objekt / dict** →  direkt verwenden
    """
    if config is None:
        config = "params.yaml"

    # Fall 1: Pfad/Dateiname
    if isinstance(config, (str, Path)):
        cfg = OmegaConf.load(str(config))             # type: ignore[arg-type]
    # Fall 2: schon OmegaConf / dict
    else:
        cfg = OmegaConf.create(config)

    return _pipeline_from_cfg(cfg)