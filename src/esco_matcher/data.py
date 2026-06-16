from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_COLUMNS = {
    "code",
    "preferredLabel_esp",
    "altLabels_esp",
    "description_esp",
}


def _clean_required_columns(frame: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {', '.join(sorted(missing))}")

    frame = frame.copy()
    for column in REQUIRED_COLUMNS:
        frame[column] = frame[column].fillna("").astype(str).str.strip()
    frame = frame[frame["description_esp"].ne("") & frame["preferredLabel_esp"].ne("")]
    return frame.reset_index(drop=True)


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

    return _clean_required_columns(frame)


def _is_esco_leaf(node: Any) -> bool:
    return isinstance(node, dict) and isinstance(node.get("esco"), dict)


def _metadata_text(metadata: dict[str, Any] | None, key: str) -> str:
    if not isinstance(metadata, dict):
        return ""
    return str(metadata.get(key, "")).strip()


def _joined_codes(
    esco_code: str,
    ciuo08_code: str,
    ciuo08_cl_codes: list[str],
) -> str:
    parts = [f"ESCO:{esco_code}"]
    if ciuo08_code:
        parts.append(f"CIUO08:{ciuo08_code}")
    if ciuo08_cl_codes:
        parts.append(f"CIUO08_CL:{' > '.join(ciuo08_cl_codes)}")
    return " | ".join(parts)


def load_enriched_json(path: str | Path) -> pd.DataFrame:
    """Flatten the enriched CIUO/ESCO hierarchy into one row per ESCO occupation."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo JSON enriquecido: {path}")

    with path.open(encoding="utf-8") as file:
        data = json.load(file)

    records: list[dict[str, Any]] = []

    def walk(
        node: Any,
        ciuo08_cl_path: list[dict[str, str]],
        ciuo08: dict[str, str] | None,
    ) -> None:
        if not isinstance(node, dict):
            return

        current_cl_path = ciuo08_cl_path
        if isinstance(node.get("ciuo08_cl"), dict):
            cl_node = node["ciuo08_cl"]
            current_cl_path = [
                *ciuo08_cl_path,
                {
                    "code": _metadata_text(cl_node, "code"),
                    "label": _metadata_text(cl_node, "glosa"),
                    "description": _metadata_text(cl_node, "descripcion"),
                },
            ]

        current_ciuo08 = ciuo08
        if isinstance(node.get("ciuo08"), dict):
            current_ciuo08 = {
                "code": _metadata_text(node["ciuo08"], "code"),
                "label": _metadata_text(node["ciuo08"], "glosa"),
            }

        if _is_esco_leaf(node):
            esco = node["esco"]
            esco_code = _metadata_text(esco, "code")
            if not esco_code:
                # The JSON stores the ESCO code as the parent key, so it is added
                # by the recursion before this branch.
                esco_code = _metadata_text(node, "_esco_code")
            ciuo08_cl_codes = [item["code"] for item in current_cl_path if item["code"]]
            ciuo08_cl_labels = [
                item["label"] for item in current_cl_path if item["label"]
            ]
            ciuo08_code = _metadata_text(current_ciuo08, "code")
            records.append(
                {
                    "code": esco_code,
                    "preferredLabel_esp": _metadata_text(
                        esco, "preferredLabel_esp"
                    ),
                    "altLabels_esp": _metadata_text(esco, "altLabels_esp"),
                    "description_esp": _metadata_text(esco, "description_esp"),
                    "conceptUri": _metadata_text(esco, "conceptUri"),
                    "preferredLabel_eng": _metadata_text(
                        esco, "preferredLabel_eng"
                    ),
                    "altLabels_eng": _metadata_text(esco, "altLabels_eng"),
                    "description_eng": _metadata_text(esco, "description_eng"),
                    "ciuo08_code": ciuo08_code,
                    "ciuo08_label": _metadata_text(current_ciuo08, "label"),
                    "ciuo08_cl_codes": " > ".join(ciuo08_cl_codes),
                    "ciuo08_cl_labels": " > ".join(ciuo08_cl_labels),
                    "ciuo08_cl_path": " > ".join(
                        f"{item['code']} {item['label']}".strip()
                        for item in current_cl_path
                        if item["code"] or item["label"]
                    ),
                    "joined_codes": _joined_codes(
                        esco_code, ciuo08_code, ciuo08_cl_codes
                    ),
                }
            )

        for key, child in node.items():
            if key in {"ciuo08_cl", "ciuo08", "esco", "_esco_code"}:
                continue
            if isinstance(child, dict):
                child_with_code = child
                if _is_esco_leaf(child):
                    child_with_code = {**child, "_esco_code": str(key)}
                walk(child_with_code, current_cl_path, current_ciuo08)

    walk(data, [], None)
    frame = pd.DataFrame.from_records(records)
    frame = _clean_required_columns(frame)
    if frame["code"].duplicated().any():
        duplicates = ", ".join(frame.loc[frame["code"].duplicated(), "code"].head(5))
        raise ValueError(f"El JSON contiene códigos ESCO duplicados: {duplicates}")
    return frame


def load_occupation_data(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".json":
        return load_enriched_json(path)
    return load_esco_csv(path)
