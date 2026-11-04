"""Smoke real y aislado: uv run --locked python scripts/compose_smoke.py.

Solo elimina los contenedores y volúmenes de su proyecto aleatorio de prueba.
"""

import argparse
import json
import os
import subprocess
import time
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
    args = parser.parse_args()
    project = f"modelops-smoke-{uuid4().hex[:10]}"
    root = Path(__file__).resolve().parents[1]
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

    try:
        if not args.skip_build:
            compose("build")
        compose("up", "-d", "--wait", "--wait-timeout", "120", "mlflow")
        # Sin modelo: vivo, pero no preparado.
        compose("up", "-d", "api")
        base = endpoint("api", "8000")
        deadline = time.monotonic() + 45
        while True:
            try:
                assert get_json(base + "/health/live")["status"] == "alive"
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError("La API no responde a liveness") from None
                time.sleep(0.2)
        try:
            get_json(base + "/health/ready")
            raise AssertionError("No debe estar preparado antes del primer modelo")
        except HTTPError as error:
            assert error.code == 503
        output = compose("run", "--rm", "-T", "train", capture=True)
        # Compose y las bibliotecas pueden anteponer mensajes al resultado CLI.
        start = output.index('{\n  "local_run_dir"')
        training, _ = json.JSONDecoder().raw_decode(output[start:])
        reference = training["candidates"]["baseline"]
        assert reference["version"] == "1"
        assert reference["source_uri"].startswith("mlflow-artifacts:/")
        compose("up", "-d", "--no-deps", "--force-recreate", "--wait", "api")
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
        # Recrea todos los procesos y las redes conservando ambos volúmenes.
        compose("down")
        compose("up", "-d", "--wait", "--wait-timeout", "120")
        second = get_json(
            endpoint("api", "8000") + "/predict",
            {"instances": features.to_dict("records")},
        )
        assert first == second
        client = make_client(resolve_config(tracking_uri=endpoint("mlflow", "5000")))
        assert len(client.search_model_versions(f"name='{reference['name']}'")) == 1
        print(
            json.dumps(
                {
                    "compose_smoke": "passed",
                    "rows": 5,
                    "version": "1",
                    "persistence": "passed",
                }
            )
        )
    finally:
        compose("down", "--volumes", "--remove-orphans")


if __name__ == "__main__":
    main()
