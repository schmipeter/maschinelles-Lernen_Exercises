from pathlib import Path
from .extract import extract_and_split
from .transform import ImageTransform
from .load import load, ImageDataset

__all__ = [
    "extract_and_split",
    "ImageTransform",
    "load",
    "ImageDataset",
]


def build_from_mat(
    mat_file: str | Path,
    out_root: str | Path = "data/raw",
    test_ratio: float = 0.2,
    seed: int = 42,
    img_format: str = "png",
) -> None:
    """Einmaliges ETL: legt data/raw/<cat|dog>/<train|test>/ an."""
    extract_and_split(
        Path(mat_file),
        Path(out_root),
        test_ratio=test_ratio,
        seed=seed,
        image_format=img_format,
    )
