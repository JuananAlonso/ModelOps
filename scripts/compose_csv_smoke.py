"""Auditoría CSV con Docker Compose aislado; solo requiere Python estándar y Docker."""

import argparse
import csv
import json
import math
import os
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from uuid import uuid4


def equivalent(left, right):
    if isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equivalent(left[key], right[key])
    elif isinstance(left, list):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            equivalent(a, b)
    elif isinstance(left, float):
        assert math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-12)
    else:
        assert left == right


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report", type=Path, default=Path("artifacts/e2e/csv-compose.json")
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    project = "modelops-csv-" + uuid4().hex[:10]
    environment = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("MODELOPS_", "MLFLOW_"))
    }
    environment.update(
        MODELOPS_API_PORT="0",
        MODELOPS_MLFLOW_PORT="0",
        POSTGRES_PASSWORD="csv_audit_only",
    )
    report = {"status": "failed", "project": project, "cases": [], "cleanup": "pending"}

    def compose(*parts, check=True):
        result = subprocess.run(
            ["docker", "compose", "--project-name", project, *parts],
            cwd=root,
            env=environment,
            capture_output=True,
            text=True,
            timeout=600 if parts[0] == "build" else 180,
        )
        if check and result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
        return result

    def request(path, body=None):
        port = compose("port", "api", "8000").stdout.strip()
        req = Request(
            "http://" + port + path,
            data=json.dumps(body).encode() if body else None,
            headers={"Content-Type": "application/json"},
        )
        with urlopen(req, timeout=15) as response:
            return json.load(response)

    def cli(folder, *parts, check=True):
        return compose(
            "run",
            "--rm",
            "--no-deps",
            "-v",
            f"{folder}:/csv:ro",
            "train",
            "env",
            "-u",
            "MLFLOW_EXPERIMENT_NAME",
            "modelops",
            *parts,
            check=check,
        )

    def invalid(folder, *parts):
        result = cli(folder, *parts, check=False)
        assert result.returncode == 2, result.stdout + result.stderr
        assert not result.stdout.strip() and "Traceback" not in result.stderr

    try:
        compose("build", "api")
        compose("up", "-d", "--wait", "--wait-timeout", "120", "mlflow")
        cases = [
            (
                "binary-lgbm",
                "clientes.csv",
                "nuevos_clientes.csv",
                "churn",
                "lightgbm",
                ["--exclude", "id_cliente"],
            ),
            (
                "binary-logistic",
                "pedidos.csv",
                "nuevos_pedidos.csv",
                "devuelto",
                "logistic-regression",
                [],
            ),
            (
                "multi-logistic",
                "segmentos.csv",
                "nuevos_segmentos.csv",
                "segmento",
                "logistic-regression",
                ["--exclude", "id", "--categorical", "codigo_postal"],
            ),
            (
                "multi-lgbm",
                "segmentos.csv",
                "nuevos_segmentos.csv",
                "segmento",
                "lightgbm",
                ["--exclude", "id", "--categorical", "codigo_postal"],
            ),
            (
                "regression-linear",
                "viviendas.csv",
                "nuevas_viviendas.csv",
                "precio",
                "linear-regression",
                [
                    "--exclude",
                    "id",
                    "--task",
                    "regression",
                    "--delimiter",
                    ";",
                    "--decimal",
                    ",",
                ],
            ),
            (
                "regression-lgbm",
                "viviendas.csv",
                "nuevas_viviendas.csv",
                "precio",
                "lightgbm",
                [
                    "--exclude",
                    "id",
                    "--task",
                    "regression",
                    "--delimiter",
                    ";",
                    "--decimal",
                    ",",
                ],
            ),
        ]
        with TemporaryDirectory(prefix="modelops csv ") as temporary:
            folder = Path(temporary)
            # El usuario de la imagen (UID 10001) debe poder leer el bind mount.
            folder.chmod(0o755)
            for source in (root / "examples/csv").glob("*.csv"):
                (folder / source.name).write_bytes(source.read_bytes())
            # Otro esquema, caracteres especiales, Latin-1, tabulador y coma decimal.
            with (folder / "externo.csv").open(
                "w", encoding="latin-1", newline=""
            ) as stream:
                writer = csv.writer(stream, delimiter="\t")
                writer.writerow(["importe [EUR]", "ciudad", "vacia", "clase"])
                for i in range(90):
                    writer.writerow(
                        [
                            "" if i % 7 == 0 else f"{i},5",
                            "" if i % 11 == 0 else ["León", "Cádiz", "Bilbao"][i % 3],
                            "",
                            ["01", "1", "otro"][i % 3],
                        ]
                    )
            (folder / "nuevos_externos.csv").write_text(
                "ciudad\tvacia\timporte [EUR]\nCiudad nueva\t\t\n\t\t5,5\n",
                encoding="latin-1",
            )
            cases.append(
                (
                    "externo",
                    "externo.csv",
                    "nuevos_externos.csv",
                    "clase",
                    "lightgbm",
                    ["--delimiter", "tab", "--decimal", ",", "--encoding", "latin-1"],
                )
            )
            for name, source, new, target, algorithm, extra in cases:
                dialect = [
                    value
                    for i, value in enumerate(extra)
                    if value in ("--delimiter", "--decimal", "--encoding")
                    or i
                    and extra[i - 1] in ("--delimiter", "--decimal", "--encoding")
                ]
                inspected = json.loads(
                    cli(folder, "inspect", f"/csv/{source}", *dialect, "--json").stdout
                )
                assert inspected["rows"] > 0
                human = cli(folder, "inspect", f"/csv/{source}", *dialect).stdout
                assert "│" in human and "\t" not in human
                train = [
                    "train",
                    f"/csv/{source}",
                    "--target",
                    target,
                    "--algorithm",
                    algorithm,
                    "--name",
                    name,
                    "--output-dir",
                    "/tmp/training",
                    *extra,
                    "--json",
                ]
                trained = json.loads(cli(folder, *train).stdout)
                assert trained["version"] == "1"
                assert trained["experiment_name"] == f"modelops-{algorithm}-{name}"
                predicted = json.loads(
                    cli(
                        folder,
                        "predict",
                        f"/csv/{new}",
                        "--model",
                        name,
                        "--version",
                        "1",
                        *dialect,
                        "--json",
                    ).stdout
                )
                environment.update(MODELOPS_MODEL_NAME=name, MODELOPS_MODEL_VERSION="1")
                compose("up", "-d", "--no-deps", "--force-recreate", "--wait", "api")
                schema = request("/model/schema")
                with (folder / new).open(
                    encoding="latin-1" if name == "externo" else "utf-8-sig"
                ) as stream:
                    rows = list(
                        csv.DictReader(
                            stream,
                            delimiter="\t"
                            if name == "externo"
                            else ";"
                            if name.startswith("regression")
                            else ",",
                        )
                    )
                for row in rows:
                    for key, value in row.items():
                        row[key] = (
                            (value or None)
                            if key in schema["categorical_features"]
                            else float(value.replace(",", "."))
                            if value
                            else None
                        )
                response = request("/predict", {"instances": rows})
                equivalent(predicted, response)
                bad = {**rows[0], "extra": 0}
                try:
                    request("/predict", {"instances": [bad]})
                    raise AssertionError("Se esperaba 422")
                except HTTPError as error:
                    assert error.code == 422
                assert request("/health/ready")["status"] == "ready"
                invalid(
                    folder,
                    "predict",
                    f"/csv/{source}",
                    "--model",
                    name,
                    "--version",
                    "1",
                    *dialect,
                    "--json",
                )
                if name == "externo":
                    second = json.loads(cli(folder, *train).stdout)
                    assert second["version"] == "2"
                    equivalent(request("/predict", {"instances": rows}), predicted)
                    invalid(
                        folder,
                        "predict",
                        f"/csv/{new}",
                        "--model",
                        name,
                        "--version",
                        "999",
                        *dialect,
                        "--json",
                    )
                report["cases"].append(
                    {
                        "name": name,
                        "task": trained["task"],
                        "algorithm": algorithm,
                        "train_rows": inspected["rows"],
                        "predicted_rows": len(rows),
                        "cli_http_parity": "passed",
                        "metrics": trained["metrics"],
                    }
                )
                print(f"passed: {name}", flush=True)
            invalid(
                folder,
                "train",
                "/csv/clientes.csv",
                "--target",
                "churn",
                "--exclude",
                "churn",
                "--algorithm",
                "lightgbm",
                "--name",
                "invalid",
                "--json",
            )
            # Recrear la pila conservando volúmenes: versión 1 sigue seleccionada.
            compose("down")
            compose("up", "-d", "--wait", "--wait-timeout", "120", "api")
            equivalent(request("/predict", {"instances": rows}), predicted)
            report.update(
                status="passed", persistence="passed", version_pinning="passed"
            )
    except Exception as error:
        report["error"] = str(error)
        raise
    finally:
        result = compose("down", "--volumes", "--remove-orphans", check=False)
        report["cleanup"] = "passed" if result.returncode == 0 else "failed"
        if result.returncode:
            report["status"] = "failed"
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    if report["cleanup"] != "passed":
        raise RuntimeError("No se pudieron limpiar los recursos de la auditoría.")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
