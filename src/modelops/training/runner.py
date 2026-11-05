"""Orquestación de ML en memoria, sin persistencia ni tracking externos."""

from dataclasses import dataclass

from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from modelops.model_contract import ModelContract
from modelops.training.config import TrainingConfig
from modelops.training.data import TabularDataset
from modelops.training.evaluation import evaluate_model
from modelops.training.models import make_models
from modelops.training.splitting import DataSplit, split_dataset
from modelops.training.validation import dataset_fingerprint


@dataclass(frozen=True)
class TrainingResult:
    config: TrainingConfig
    dataset: TabularDataset
    dataset_sha256: str
    split: DataSplit
    models: dict[str, Pipeline]
    metrics: dict[str, dict[str, float | None]]


def train_dataset(
    dataset: TabularDataset,
    config: TrainingConfig,
    *,
    candidates: tuple[str, ...] = ("baseline", "lightgbm"),
) -> TrainingResult:
    """Ajustar los candidatos solicitados sobre train y medir una vez sobre test."""
    fingerprint = dataset_fingerprint(dataset)
    split = split_dataset(dataset, config)
    if (
        not candidates
        or len(set(candidates)) != len(candidates)
        or not set(candidates) <= {"baseline", "lightgbm"}
    ):
        raise ValueError("Selecciona baseline o lightgbm, sin candidatos duplicados.")
    available = make_models(
        dataset.feature_names,
        config.seed,
        task=dataset.task,
        categorical_features=dataset.categorical_features,
        allow_missing=dataset.allow_missing,
    )
    models = {name: available[name] for name in candidates}
    contract = ModelContract(
        schema_version=2
        if dataset.allow_missing
        or dataset.categorical_features
        or dataset.task == "regression"
        or len(dataset.target_names) != 2
        else 1,
        feature_names=dataset.feature_names,
        class_names=dataset.target_names,
        target_name=dataset.target_name,
        task=dataset.task,
        categorical_features=dataset.categorical_features,
        feature_type="mixed" if dataset.categorical_features else "number",
        allow_missing=dataset.allow_missing,
    )
    metrics = {}
    # Limitar también BLAS/OpenMP, no solo el paralelismo propio de LightGBM.
    with threadpool_limits(limits=1):
        for name, model in models.items():
            model.fit(split.x_train, split.y_train)
            model.modelops_contract_ = contract.model_dump(mode="json")
            metrics[name] = evaluate_model(
                model, split.x_test, split.y_test, task=dataset.task
            )
    return TrainingResult(config, dataset, fingerprint, split, models, metrics)
