from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "code",
    "preferredLabel_esp",
    "altLabels_esp",
    "description_esp",
}


def load_esco_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo ESCO: {path}")

    last_error: UnicodeDecodeError | None = None
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            frame = pd.read_csv(
                path,
                sep=";",
                encoding=encoding,
                dtype={"code": str},
                keep_default_na=False,
            )
            break
        except UnicodeDecodeError as error:
            last_error = error
    else:
        raise ValueError("No se pudo determinar la codificación del CSV") from last_error

    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {', '.join(sorted(missing))}")

    frame = frame.copy()
    for column in REQUIRED_COLUMNS:
        frame[column] = frame[column].fillna("").astype(str).str.strip()
    frame = frame[frame["description_esp"].ne("") & frame["preferredLabel_esp"].ne("")]
    return frame.reset_index(drop=True)
