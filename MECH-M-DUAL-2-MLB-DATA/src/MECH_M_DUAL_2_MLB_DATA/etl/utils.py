"""
Kleine Hilfsfunktionen, die von mehreren ETL-Schritten benötigt werden.
"""

from __future__ import annotations

import math
import numpy as np


def vector_to_image(vec: np.ndarray) -> np.ndarray:
    """
    Wandelt einen 1-D-Vektor (L,) zurück in ein quadratisches Graustufen-Bild
    der Form (H, W).

    * L muss eine Quadratzahl sein (z. B. 1024 = 32×32).  
    * Bei anderen Längen wird eine ValueError geworfen.
    """
    L: int = vec.size
    s: int = int(math.isqrt(L))
    if s * s != L:
        raise ValueError(
            f"Vektorlaenge {L} ist keine Quadratzahl – Format unbekannt."
        )
    return vec.reshape(s, s)