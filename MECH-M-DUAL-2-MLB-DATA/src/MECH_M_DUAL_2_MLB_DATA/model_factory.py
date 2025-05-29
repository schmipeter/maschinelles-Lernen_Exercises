"""
model_factory.py
================
Erzeugt ein Modell oder eine Pipeline *ausschließlich* aus einer YAML-Datei.
* Reihenfolge im YAML spielt KEINE Rolle.
* Container-Klassen (VotingClassifier u. a.) holen fehlende Unter-Modelle
  automatisch nach (“on-demand instantiation”).

Beispiel–Verwendung
-------------------
from MECH_M_DUAL_2_MLB_DATA import build_model
model = build_model()                     # liest params.yaml im Projekt-Root
"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any, Dict, List

from omegaconf import OmegaConf
from sklearn.pipeline import Pipeline

__all__ = ["build_model"]


# ---------------------------------------------------------------------------#
# Hilfsfunktionen                                                            #
# ---------------------------------------------------------------------------#
def _load_class(dotted_path: str):
    """Lädt Klasse per 'paket.modul.Klassename'."""
    module_name, cls_name = dotted_path.rsplit(".", 1)
    module = import_module(module_name)
    return getattr(module, cls_name)


def _is_container(section: "OmegaConf") -> bool:
    """Container-Klassen besitzen meist eine 'estimators'-Liste."""
    return "estimators" in section


def _instantiate(
    name: str,
    cfg: "OmegaConf",
    registry: Dict[str, Any],
    cfg_root: "OmegaConf",
):
    """Erzeugt Objekt laut *cfg* – erzeugt fehlende Unter-Modelle automatisch."""
    if name in registry:  # bereits vorhanden
        return registry[name]

    cls = _load_class(cfg.type)
    kwargs = OmegaConf.to_container(cfg.get("init_args", {}), resolve=True)

    # Container?  → erst Unter-Modelle sicherstellen
    if "estimators" in cfg:
        for sub in cfg.estimators:
            if sub not in registry:
                _instantiate(sub, cfg_root[sub], registry, cfg_root)
        kwargs["estimators"] = [(e, registry[e]) for e in cfg.estimators]

    obj = cls(**kwargs)
    registry[name] = obj
    return obj


# ---------------------------------------------------------------------------#
# Öffentliche Fabrikfunktion                                                 #
# ---------------------------------------------------------------------------#
def build_model(config_path: str | Path = "params.yaml"):
    """
    Baut Modell bzw. Pipeline gem. YAML-Konfiguration.

    * config_path: Pfad zur YAML (relativ/absolut). Default: params.yaml im
      aktuellen Verzeichnis.
    * Gibt `sklearn.pipeline.Pipeline` zurück, falls ein 'Pipeline'-Schlüssel
      existiert, sonst das unter 'model' referenzierte oder erste Objekt.
    """
    cfg = OmegaConf.load(str(config_path))
    registry: Dict[str, Any] = {}

    # Phase 1 – alle Nicht-Container instanziieren
    for name, section in cfg.items():
        if name == "Pipeline" or _is_container(section):
            continue
        _instantiate(name, section, registry, cfg)

    # Phase 2 – Container (holt Untermodelle on-demand)
    for name, section in cfg.items():
        if name == "Pipeline" or not _is_container(section):
            continue
        _instantiate(name, section, registry, cfg)

    # Pipeline bauen?
    if "Pipeline" in cfg:
        steps = [(n, registry[n]) for n in cfg.Pipeline]
        return Pipeline(steps)

    # Einzelmodell zurückgeben
    if "model" in cfg:
        return registry[str(cfg.model)]
    # Fallback: erster Key (ohne Pipeline)
    return registry[next(iter(k for k in cfg if k != "Pipeline"))]