"""CLI no interactiva para aprendizaje supervisado sobre CSV propios."""

import argparse
import csv
import json
import os
import pickle
import shlex
import shutil
import sys
import textwrap
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd
from lightgbm.basic import LightGBMError
from mlflow.exceptions import MlflowException

from modelops.csv_data import CsvOptions, csv_dataset, inspect_csv, read_csv
from modelops.serving.config import ServingConfig
from modelops.serving.contracts import ModelIdentity
from modelops.serving.predictor import Predictor
from modelops.tracking.adapter import load_registered_pipeline, track_result
from modelops.tracking.config import resolve_config
from modelops.training.artifacts import persist_result
from modelops.training.config import TrainingConfig
from modelops.training.runner import train_dataset

ALGORITHMS = {
    "logistic-regression": "baseline",
    "linear-regression": "baseline",
    "lightgbm": "lightgbm",
}


def _format_inspection(output: dict, terminal_width: int | None = None) -> str:
    """Renderizar columnas alineadas y ajustar nombres sin ocultarlos."""
    headers = [
        "Columna",
        "Tipo",
        "Ausentes",
        "Distintos",
        "Clasificación",
        "Regresión",
    ]
    rows = [
        [
            str(column["name"]),
            str(column["dtype"]),
            str(column["missing"]),
            str(column["unique"]),
            "sí" if column["classification_candidate"] else "no",
            "sí" if column["regression_candidate"] else "no",
        ]
        for column in output["columns"]
    ]
    available = terminal_width or shutil.get_terminal_size(fallback=(100, 24)).columns
    fixed = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(1, len(headers))
    ]
    # Reservar el espacio restante para el nombre; nunca truncar cabeceras.
    separators = 3 * (len(headers) - 1)
    name_width = max(8, available - separators - sum(fixed))
    name_width = min(
        name_width, max(8, max(len(headers[0]), *(len(row[0]) for row in rows)))
    )
    widths = [name_width, *fixed]

    def render(values: list[str]) -> list[str]:
        first, *rest = values
        pieces = textwrap.wrap(
            first,
            width=name_width,
            break_long_words=True,
            break_on_hyphens=False,
            subsequent_indent="↳ ",
        ) or [""]
        lines = []
        for piece in pieces:
            cells = [piece, *rest]
            cells = [
                value.ljust(width) for value, width in zip(cells, widths, strict=True)
            ]
            lines.append(" │ ".join(cells).rstrip())
        return lines

    lines = render(headers)
    lines.append("─┼─".join("─" * width for width in widths))
    for row in rows:
        lines.extend(render(row))
    lines.insert(0, f"Filas: {output['rows']} · Columnas: {len(rows)}")
    for issue in output["issues"]:
        lines.append(f"Revisar: {issue}")
    return "\n".join(lines)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        description="Inspecciona, entrena y predice CSV de clasificación o regresión."
    )
    commands = root.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser(
        "inspect",
        help="Revisar tipos, ausentes y posibles objetivos.",
        epilog="Ejemplo: modelops inspect clientes.csv",
    )
    train = commands.add_parser(
        "train",
        help="Entrenar y registrar una versión sin cambiar la API.",
        epilog=(
            "Ejemplo: modelops train clientes.csv --target churn "
            "--algorithm lightgbm --name churn --exclude id_cliente"
        ),
    )
    predict = commands.add_parser(
        "predict",
        help="Predecir con una versión explícita registrada.",
        epilog=(
            "Ejemplo: modelops predict nuevos_clientes.csv --model churn --version 1"
        ),
    )
    for command in (inspect, train, predict):
        command.add_argument(
            "csv", type=Path, help="CSV con cabecera (UTF-8 y coma por defecto)."
        )
        command.add_argument(
            "--delimiter",
            default=",",
            help="Separador de un carácter, por ejemplo ';'. Usa 'tab' para tabulador.",
        )
        command.add_argument(
            "--encoding", default="utf-8-sig", help="Codificación del CSV (utf-8-sig)."
        )
        command.add_argument(
            "--decimal", default=".", choices=[".", ","], help="Separador decimal (.)."
        )
        command.add_argument(
            "--json", action="store_true", help="Salida JSON sin progreso en stdout."
        )
    train.add_argument("--target", required=True)
    train.add_argument(
        "--task",
        choices=["classification", "regression"],
        default="classification",
        help="Clasificación binaria/multiclase (por defecto) o regresión numérica.",
    )
    train.add_argument(
        "--categorical",
        action="append",
        default=[],
        help="Forzar variable categórica; se puede repetir (p. ej. código postal).",
    )
    train.add_argument(
        "--numeric",
        action="append",
        default=[],
        help="Forzar variable numérica y rechazar texto inválido; se puede repetir.",
    )
    train.add_argument("--algorithm", required=True, choices=ALGORITHMS)
    train.add_argument("--name", required=True, help="Nombre exacto en el Registry.")
    train.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Columna a ignorar; se puede repetir.",
    )
    train.add_argument(
        "--seed", type=int, default=42, help="Semilla de partición y modelo (42)."
    )
    train.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fracción de test estratificado (0.2).",
    )
    train.add_argument("--output-dir", type=Path, default=Path("artifacts/training"))
    predict.add_argument("--model", required=True)
    predict.add_argument(
        "--version",
        required=True,
        help="Versión decimal positiva; sin latest ni alias.",
    )
    for command in (train, predict):
        command.add_argument("--storage-dir", type=Path)
        command.add_argument("--tracking-uri")
        command.add_argument("--experiment-name")
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    local = None
    stage = "entrada"
    try:
        # Algunas bibliotecas escriben progreso en stdout: reservarlo para el resultado.
        with redirect_stdout(sys.stderr):
            options = CsvOptions(
                "\t" if args.delimiter == "tab" else args.delimiter,
                args.encoding,
                args.decimal,
            )
            if args.command == "inspect":
                output = inspect_csv(args.csv, options)
            else:
                experiment_name = args.experiment_name
                if (
                    args.command == "train"
                    and experiment_name is None
                    and "MLFLOW_EXPERIMENT_NAME" not in os.environ
                ):
                    experiment_name = f"modelops-{args.algorithm}-{args.name}"
                config = resolve_config(
                    args.storage_dir,
                    tracking_uri=args.tracking_uri,
                    experiment_name=experiment_name,
                )
                name = args.name if args.command == "train" else args.model
                selected = ServingConfig(
                    config, name, "1" if args.command == "train" else args.version
                )
                if args.command == "train":
                    if (
                        args.algorithm == "logistic-regression"
                        and args.task != "classification"
                    ) or (
                        args.algorithm == "linear-regression"
                        and args.task != "regression"
                    ):
                        raise ValueError(
                            "Algoritmo incompatible con --task: "
                            "usa logistic-regression "
                            "para clasificación, linear-regression para regresión "
                            "o lightgbm para ambas."
                        )
                    candidate = ALGORITHMS[args.algorithm]
                    result = train_dataset(
                        csv_dataset(
                            args.csv,
                            args.target,
                            args.exclude,
                            task=args.task,
                            options=options,
                            categorical=args.categorical,
                            numeric=args.numeric,
                        ),
                        TrainingConfig(args.seed, args.test_size),
                        candidates=(candidate,),
                    )
                    stage = "persistencia"
                    local = persist_result(
                        result, args.output_dir, project_root=Path.cwd()
                    )
                    stage = "registro"
                    tracked = track_result(
                        result, local, config, registered_names={candidate: name}
                    )
                    reference = tracked.candidates[candidate]
                    output = {
                        "metrics": result.metrics[candidate],
                        "task": result.dataset.task,
                        "experiment_name": config.experiment_name,
                        "feature_names": list(result.dataset.feature_names),
                        "categorical_features": list(
                            result.dataset.categorical_features
                        ),
                        "name": reference.name,
                        "version": reference.version,
                        "local_run_dir": str(local),
                        "class_names": list(result.dataset.target_names),
                        "predict_command": shlex.join(
                            [
                                "uv",
                                "run",
                                "--locked",
                                "modelops",
                                "predict",
                                "nuevos_clientes.csv",
                                *(
                                    ["--delimiter", args.delimiter]
                                    if args.delimiter != ","
                                    else []
                                ),
                                *(
                                    ["--encoding", args.encoding]
                                    if args.encoding != "utf-8-sig"
                                    else []
                                ),
                                *(
                                    ["--decimal", args.decimal]
                                    if args.decimal != "."
                                    else []
                                ),
                                "--model",
                                name,
                                "--version",
                                reference.version,
                                "--tracking-uri",
                                config.tracking_uri,
                            ]
                        ),
                    }
                else:
                    stage = "carga del modelo"
                    predictor = Predictor(
                        load_registered_pipeline(config, name, selected.model_version)
                    )
                    stage = "entrada de predicción"
                    features = predictor.contract.parse_csv_features(
                        read_csv(args.csv, options), options.decimal
                    )
                    response = predictor.response(
                        features,
                        ModelIdentity(
                            name=name,
                            version=selected.model_version,
                            uri=f"models:/{name}/{selected.model_version}",
                        ),
                    )
                    output = response.model_dump(mode="json")
    except (
        ValueError,
        OSError,
        UnicodeError,
        csv.Error,
        pd.errors.ParserError,
        MlflowException,
        pickle.UnpicklingError,
        EOFError,
        ImportError,
        AttributeError,
        LightGBMError,
        LookupError,
    ) as error:
        print(f"Error de {stage}: {error}", file=sys.stderr)
        if local is not None:
            print(f"Artefactos locales conservados: {local}", file=sys.stderr)
        print(
            "Revisa el CSV y las opciones con modelops "
            + args.command
            + " --help; comprueba nombre, versión y configuración MLflow "
            "si falla el modelo o el registro.",
            file=sys.stderr,
        )
        return 2
    if args.json:
        print(json.dumps(output, indent=2, allow_nan=False))
    elif args.command == "inspect":
        print(_format_inspection(output))
    elif args.command == "train":
        print(f"Registrado: {output['name']} versión {output['version']}")
        print(f"Experimento MLflow: {output['experiment_name']}")
        print(f"Tarea: {output['task']}")
        if len(output["class_names"]) == 2:
            print(f"Clase positiva 1 = {output['class_names'][1]}")
        elif output["class_names"]:
            print("Clases: " + json.dumps(output["class_names"], ensure_ascii=False))
        print("Métricas: " + json.dumps(output["metrics"]))
        print(f"Artefactos locales: {output['local_run_dir']}")
        print("Prueba con un CSV de solo variables de entrada:")
        print(output["predict_command"])
    elif output.get("task") == "regression":
        print(f"Modelo: {output['model']['name']} versión {output['model']['version']}")
        print("fila\tvalor predicho")
        for index, row in enumerate(output["predictions"], 1):
            print(f"{index}\t{row['value']}")
    else:
        print(f"Modelo: {output['model']['name']} versión {output['model']['version']}")
        print(
            "fila\tclase original\t"
            + "\t".join(f"P({name})" for name in output["class_names"])
        )
        for index, row in enumerate(output["predictions"], 1):
            original = output["class_names"][output["classes"].index(row["label"])]
            print(
                f"{index}\t{original}\t"
                + "\t".join(str(value) for value in row["probabilities"])
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
