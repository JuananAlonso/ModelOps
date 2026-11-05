"""Lectura validada de CSV y adaptación a tareas tabulares supervisadas."""

import csv
import io
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from modelops.training.data import TabularDataset


@dataclass(frozen=True)
class CsvOptions:
    delimiter: str = ","
    encoding: str = "utf-8-sig"
    decimal: str = "."

    def __post_init__(self) -> None:
        if len(self.delimiter) != 1 or self.delimiter in '\r\n"':
            raise ValueError(
                "El separador debe ser un carácter distinto de comillas/fin de línea."
            )
        if self.decimal not in (".", ",") or self.decimal == self.delimiter:
            raise ValueError(
                "El decimal debe ser punto o coma y distinto del separador."
            )


def read_csv(path: Path, options: CsvOptions | None = None) -> pd.DataFrame:
    options = options or CsvOptions()
    content = path.read_text(encoding=options.encoding)
    reader = csv.reader(io.StringIO(content), delimiter=options.delimiter, strict=True)
    header = next(reader, [])
    if not header or any(not name.strip() for name in header):
        raise ValueError("El CSV requiere una cabecera con nombres no vacíos.")
    duplicates = sorted({name for name in header if header.count(name) > 1})
    if duplicates:
        raise ValueError(f"Cabeceras duplicadas: {duplicates}. Renombra las columnas.")
    for row in reader:
        if row and len(row) != len(header):
            raise ValueError(
                f"Fila CSV {reader.line_num}: usa {len(header)} campos "
                f"separados por {options.delimiter!r}."
            )
    # Una sola interpretación para validar y leer; conservar etiquetas como 01.
    return pd.read_csv(
        io.StringIO(content), sep=options.delimiter, dtype=str, skip_blank_lines=True
    )


def numeric_values(values: pd.Series, decimal: str = ".") -> pd.Series:
    if decimal != "." and not pd.api.types.is_numeric_dtype(values):
        values = values.str.replace(decimal, ".", regex=False)
    return pd.to_numeric(values, errors="coerce")


def numeric_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Conversión estricta conservada para consumidores del contrato v1."""
    if not len(frame.columns):
        raise ValueError("No quedan variables de entrada; revisa --target y --exclude.")
    converted = pd.DataFrame(index=frame.index)
    for name in frame.columns:
        values = numeric_values(frame[name])
        if not np.isfinite(values.to_numpy(dtype=np.float64)).all():
            raise ValueError(
                f"La columna {name!r} contiene ausentes, texto o valores no finitos. "
                "Corrígela o exclúyela del entrenamiento con --exclude."
            )
        converted[name] = values.astype(np.float64)
    return converted


def csv_dataset(
    path: Path,
    target: str,
    exclude: list[str],
    *,
    task: str = "classification",
    options: CsvOptions | None = None,
    categorical: list[str] | None = None,
    numeric: list[str] | None = None,
) -> "TabularDataset":
    # Import local para mantener independiente el lector usado por inferencia.
    from modelops.training.data import TabularDataset

    options = options or CsvOptions()
    frame = read_csv(path, options)
    if target not in frame.columns:
        raise ValueError(
            f"Objetivo {target!r} inexistente. "
            f"Elige --target entre {list(frame.columns)}."
        )
    unknown = sorted(set(exclude) - set(frame.columns))
    if unknown or target in exclude:
        raise ValueError(
            f"Exclusiones inválidas: {unknown or [target]}. "
            "No excluyas el objetivo y usa nombres existentes."
        )
    features = frame.drop(columns=[target, *set(exclude)]).copy()
    if not len(features.columns):
        raise ValueError("No quedan variables de entrada; revisa --target y --exclude.")
    categorical, numeric = set(categorical or ()), set(numeric or ())
    if (categorical | numeric) - set(features.columns) or categorical & numeric:
        raise ValueError(
            "--categorical/--numeric requieren variables existentes y sin solapamiento."
        )
    categories = []
    for name in features:
        values = features[name]
        converted = numeric_values(values, options.decimal)
        is_numeric = values.notna().eq(converted.notna()).all()
        if name not in categorical and (name in numeric or is_numeric):
            if (values.notna() & converted.isna()).any():
                raise ValueError(
                    f"La columna numérica {name!r} contiene texto inválido."
                )
            if np.isinf(converted.to_numpy(dtype=float)).any():
                raise ValueError(f"La columna {name!r} contiene valores no finitos.")
            features[name] = converted.astype(float)
        else:
            categories.append(name)
            # sklearn trabaja con strings y np.nan, no pd.NA/None.
            features[name] = values.astype(object).where(values.notna(), np.nan)
    labels = frame[target]
    if labels.isna().any():
        raise ValueError("El objetivo tiene ausentes; completa o elimina esas filas.")
    if task == "classification":
        classes = tuple(sorted(labels.unique().tolist()))
        if len(classes) < 2:
            raise ValueError(
                "El objetivo necesita al menos dos clases. Revisa --target."
            )
        encoded = labels.map({label: i for i, label in enumerate(classes)}).astype(
            "int64"
        )
    elif task == "regression":
        classes = ()
        encoded = numeric_values(labels, options.decimal).astype(float)
        if not np.isfinite(encoded.to_numpy()).all():
            raise ValueError(
                "El objetivo de regresión debe ser numérico finito, sin ausentes."
            )
    else:
        raise ValueError("La tarea debe ser classification o regression.")
    encoded.name = target
    return TabularDataset(
        name=f"csv:{path.name}",
        features=features,
        target=encoded,
        feature_names=tuple(features.columns),
        target_names=classes,
        target_name=target,
        task=task,
        categorical_features=tuple(categories),
        allow_missing=True,
        source_options=asdict(options),
    )


def inspect_csv(path: Path, options: CsvOptions | None = None) -> dict:
    options = options or CsvOptions()
    frame = read_csv(path, options)
    columns, issues = [], []
    for name in frame:
        values = frame[name]
        converted = numeric_values(values, options.decimal)
        numeric = values.notna().eq(converted.notna()).all() and values.notna().any()
        unique = int(values.nunique())
        counts = values.value_counts()
        columns.append(
            {
                "name": name,
                "dtype": str(converted.dtype) if numeric else "string",
                "missing": int(values.isna().sum()),
                "unique": unique,
                "numeric_finite": bool(
                    numeric and np.isfinite(converted.to_numpy(dtype=float)).all()
                ),
                "binary_candidate": bool(values.notna().all() and unique == 2),
                "classification_candidate": bool(
                    values.notna().all() and unique >= 2 and counts.min() >= 2
                ),
                "regression_candidate": bool(
                    numeric and np.isfinite(converted.to_numpy(dtype=float)).all()
                ),
                "class_counts": {str(k): int(v) for k, v in counts.items()}
                if 0 < unique <= 20
                else None,
            }
        )
        if not values.notna().any():
            issues.append(
                f"{name!r}: columna vacía; se imputará si es variable, "
                "no puede ser objetivo."
            )
        if unique == len(frame) and len(frame):
            issues.append(
                f"{name!r}: todos los valores son únicos; "
                "revisa si es un identificador y usa --exclude."
            )
        if numeric and np.isinf(converted.to_numpy(dtype=float)).any():
            issues.append(
                f"{name!r}: contiene infinitos; corrígelos antes de entrenar."
            )
        if 1 < unique <= 20 and counts.min() < 2:
            issues.append(
                f"{name!r}: una clase tiene menos de dos filas; "
                "añade ejemplos para estratificar."
            )
    if len(frame) < 6:
        issues.append(
            "Con test_size=0.2 hacen falta al menos 6 filas "
            "para dejar dos filas en test."
        )
    if len(frame.columns) < 2:
        issues.append("No quedarían variables tras elegir el objetivo; añade columnas.")
    return {
        "rows": len(frame),
        "columns": columns,
        "issues": issues,
        "csv_options": asdict(options),
    }
