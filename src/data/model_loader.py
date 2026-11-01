import re
from functools import lru_cache
from pathlib import Path
from typing import List

import numpy as np

from src.data.models import Model

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "datasets"


def list_available_models() -> List[str]:
    return sorted(path.stem for path in DATA_DIR.glob("*.m"))


def _split_numbers(text: str) -> List[str]:
    return [token for token in re.split(r"[,\s]+", text.strip()) if token]


def _parse_numeric_sequence(block: str) -> np.ndarray:
    values = _split_numbers(block.replace("\n", " ").replace("\r", " "))
    return np.array([float(v) for v in values], dtype=float)


def _parse_matrix(block: str) -> np.ndarray:
    rows = [[float(v) for v in _split_numbers(row)] for row in block.strip().split(";") if row.strip()]
    return np.array(rows, dtype=float)


def _extract_block(content: str, name: str) -> str:
    pattern = rf"\b{name}\b\s*=\s*\[(.*?)\];"
    match = re.search(pattern, content, flags=re.DOTALL)
    if not match:
        raise ValueError(f"Missing block: {name}")
    return match.group(1)


def _extract_scalar(content: str, name: str) -> int:
    pattern = rf"\b{name}\b\s*=\s*(\d+);"
    match = re.search(pattern, content)
    if not match:
        raise ValueError(f"Missing scalar: {name}")
    return int(match.group(1))


def _reshape_inputs(I: int, J: int, data: dict):
    return (
        data["cij"].reshape(I, J),
        data["aij"].reshape(I, J),
        data["bi"].reshape(I),
        data["X"].reshape(I),
        data["Y"].reshape(I),
        data["XX"].reshape(J),
        data["YY"].reshape(J),
    )


def _compute_distance_matrix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.sqrt((x[:, None] - x[None, :]) ** 2 + (y[:, None] - y[None, :]) ** 2)


@lru_cache
def load_model(name: str) -> Model:
    path = DATA_DIR / f"{name}.m"
    if not path.exists():
        raise FileNotFoundError(f"Dataset '{name}' not found in {DATA_DIR}")

    content = path.read_text(encoding="utf-8")

    I = _extract_scalar(content, "I")
    J = _extract_scalar(content, "J")

    raw = {
        "cij": _parse_matrix(_extract_block(content, "cij")),
        "aij": _parse_matrix(_extract_block(content, "aij")),
        "bi": _parse_numeric_sequence(_extract_block(content, "bi")),
        "X": _parse_numeric_sequence(_extract_block(content, "X")),
        "Y": _parse_numeric_sequence(_extract_block(content, "Y")),
        "XX": _parse_numeric_sequence(_extract_block(content, "XX")),
        "YY": _parse_numeric_sequence(_extract_block(content, "YY")),
    }

    cij, aij, bi, X, Y, XX, YY = _reshape_inputs(I, J, raw)

    DIS = _compute_distance_matrix(X, Y)
    F = _compute_distance_matrix(XX, YY)

    return Model(
        I=I,
        J=J,
        cij=cij,
        aij=aij,
        bi=bi,
        DIS=DIS,
        F=F,
    )
