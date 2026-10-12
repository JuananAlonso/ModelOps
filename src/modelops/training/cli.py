"""Entrada CLI fina: cargar, entrenar, persistir y presentar resultados."""

import argparse
import json
from pathlib import Path

from modelops.training.artifacts import persist_result
from modelops.training.config import TrainingConfig
from modelops.training.data import load_demo_dataset
from modelops.training.runner import train_dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ejemplo técnico de clasificación binaria, sin uso clínico."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/training"))
    args = parser.parse_args(argv)
    try:
        config = TrainingConfig(seed=args.seed, test_size=args.test_size)
        result = train_dataset(load_demo_dataset(), config)
        run_dir = persist_result(result, args.output_dir, project_root=Path.cwd())
    except (ValueError, OSError) as error:
        parser.exit(2, f"Error de entrenamiento: {error}\n")
    print(json.dumps({"run_dir": str(run_dir), "metrics": result.metrics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
