"""Smoke real y aislado: uv run --locked python scripts/compose_smoke.py.

Solo elimina los contenedores y volúmenes de su proyecto aleatorio de prueba.
"""

import argparse
import json
import os
import platform
import statistics
import subprocess
import time
from importlib.metadata import version
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from uuid import uuid4

import numpy as np

from modelops.tracking.adapter import load_registered_pipeline, make_client
from modelops.tracking.config import resolve_config
from modelops.training.data import load_demo_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--report", type=Path, help="Guardar informe JSON informativo.")
    args = parser.parse_args()
    project = f"modelops-smoke-{uuid4().hex[:10]}"
    root = Path(__file__).resolve().parents[1]
    report = {
        "status": "failed",
        "cleanup": "pending",
        "project": project,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "mlflow": version("mlflow"),
        "batch_rows": 5,
        "latency_samples": 20,
    }
    environment = {
        **os.environ,
        "MODELOPS_API_PORT": "0",
        "MODELOPS_MLFLOW_PORT": "0",
        "MODELOPS_MODEL_NAME": "modelops-baseline",
        "MODELOPS_MODEL_VERSION": "1",
        "POSTGRES_PASSWORD": "smoke_local_only",
    }

    def compose(*arguments: str, capture: bool = False) -> str:
        result = subprocess.run(
            ["docker", "compose", "--project-name", project, *arguments],
            cwd=root,
            env=environment,
            check=True,
            text=True,
            stdout=subprocess.PIPE if capture else None,
            timeout=600 if arguments[0] == "build" else 180,
        )
        return result.stdout or ""

    def endpoint(service: str, port: str) -> str:
        return "http://" + compose("port", service, port, capture=True).strip()

    def get_json(url: str, body: dict | None = None) -> dict:
        request = Request(
            url,
            data=json.dumps(body).encode() if body is not None else None,
            headers={"Content-Type": "application/json"},
        )
        with urlopen(request, timeout=15) as response:
            return json.load(response)

    def wait_live(base: str) -> None:
        deadline = time.monotonic() + 60
        while True:
            try:
                assert get_json(base + "/health/live")["status"] == "alive"
                return
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError("La API no responde a liveness") from None
                time.sleep(0.2)

    def expect_error(base: str, status: int, body: dict | None = None) -> None:
        try:
            get_json(base + ("/predict" if body is not None else "/health/ready"), body)
            raise AssertionError(f"Se esperaba HTTP {status}")
        except HTTPError as error:
            assert error.code == status
            text = error.read().decode()
            assert "Traceback" not in text and "postgresql" not in text

    try:
        if not args.skip_build:
            compose("build")
        compose("up", "-d", "--wait", "--wait-timeout", "120", "mlflow")
        # Sin modelo: vivo, pero no preparado.
        compose("up", "-d", "api")
        base = endpoint("api", "8000")
        wait_live(base)
        expect_error(base, 503)
        output = compose("run", "--rm", "-T", "train", capture=True)
        # Compose y las bibliotecas pueden anteponer mensajes al resultado CLI.
        start = output.index('{\n  "local_run_dir"')
        training, _ = json.JSONDecoder().raw_decode(output[start:])
        reference = training["candidates"]["baseline"]
        assert reference["version"] == "1"
        assert reference["source_uri"].startswith("mlflow-artifacts:/")
        started = time.perf_counter()
        compose("up", "-d", "--no-deps", "--force-recreate", "--wait", "api")
        report["api_start_ready_ms"] = round((time.perf_counter() - started) * 1000, 2)
        features = load_demo_dataset().features.iloc[:5]
        first = get_json(
            endpoint("api", "8000") + "/predict",
            {"instances": features.to_dict("records")},
        )
        config = resolve_config(tracking_uri=endpoint("mlflow", "5000"))
        pipeline = load_registered_pipeline(config, reference["name"], "1")
        np.testing.assert_array_equal(
            [row["label"] for row in first["predictions"]], pipeline.predict(features)
        )
        np.testing.assert_allclose(
            [row["probabilities"] for row in first["predictions"]],
            pipeline.predict_proba(features),
            rtol=1e-9,
            atol=1e-12,
        )
        base = endpoint("api", "8000")
        body = {"instances": features.to_dict("records")}
        expect_error(base, 422, {"instances": []})
        expect_error(base, 422, {"instances": [{"mean radius": 1}]})
        samples = []
        for _ in range(report["latency_samples"]):
            request = Request(
                base + "/predict",
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"},
            )
            started = time.perf_counter()
            with urlopen(request, timeout=15) as response:
                raw = response.read()
            samples.append((time.perf_counter() - started) * 1000)
            assert json.loads(raw) == first
        report["latency_p50_ms"] = round(statistics.median(samples), 2)
        report["latency_p95_ms"] = round(sorted(samples)[18], 2)
        report["response_bytes"] = len(raw)
        # El objeto cargado sigue siendo utilizable sin acceso al Registry.
        compose("stop", "mlflow")
        assert get_json(base + "/predict", body) == first
        # Un proceso nuevo, sin backend, permanece vivo pero no preparado.
        compose("up", "-d", "--no-deps", "--force-recreate", "api")
        unavailable = endpoint("api", "8000")
        wait_live(unavailable)
        expect_error(unavailable, 503)
        expect_error(unavailable, 503, body)
        # Recrea todos los procesos y las redes conservando ambos volúmenes.
        compose("down")
        started = time.perf_counter()
        compose("up", "-d", "--wait", "--wait-timeout", "120")
        report["stack_restart_ready_ms"] = round(
            (time.perf_counter() - started) * 1000, 2
        )
        second = get_json(
            endpoint("api", "8000") + "/predict",
            {"instances": features.to_dict("records")},
        )
        assert first == second
        client = make_client(resolve_config(tracking_uri=endpoint("mlflow", "5000")))
        assert len(client.search_model_versions(f"name='{reference['name']}'")) == 1
        report.update(
            {
                "status": "passed",
                "compose_smoke": "passed",
                "rows": 5,
                "version": "1",
                "persistence": "passed",
            }
        )
    finally:
        try:
            if report["status"] != "passed":
                compose("ps")
                compose("logs", "--tail", "80", "api")
        finally:
            try:
                compose("down", "--volumes", "--remove-orphans")
                report["cleanup"] = "passed"
            except Exception:
                report["status"] = "failed"
                report["cleanup"] = "failed"
                raise
            finally:
                if args.report:
                    args.report.parent.mkdir(parents=True, exist_ok=True)
                    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
