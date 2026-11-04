"""Servidor de desarrollo local; entrenamiento se ejecuta en otro proceso."""

import argparse
from pathlib import Path

import uvicorn

from modelops.serving.app import create_app
from modelops.serving.config import resolve_serving_config
from modelops.tracking.config import resolve_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inferencia de ejemplo, sin uso clínico."
    )
    parser.add_argument("--model-name")
    parser.add_argument("--model-version")
    parser.add_argument("--storage-dir", type=Path)
    parser.add_argument("--tracking-uri")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    try:
        if not 1 <= args.port <= 65535:
            raise ValueError("El puerto debe estar entre 1 y 65535.")
        config = resolve_serving_config(
            tracking=resolve_config(args.storage_dir, tracking_uri=args.tracking_uri),
            model_name=args.model_name,
            model_version=args.model_version,
        )
    except ValueError as error:
        parser.exit(2, f"Error de configuración de serving: {error}\n")
    uvicorn.run(create_app(config), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
