"""Orquestación de ML en memoria, sin persistencia ni tracking externos."""

from dataclasses import dataclass

from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

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


def train_dataset(dataset: TabularDataset, config: TrainingConfig) -> TrainingResult:
    """Ajustar ambos candidatos sobre train y medir una vez sobre el mismo test."""
    fingerprint = dataset_fingerprint(dataset)
    split = split_dataset(dataset, config)
    models = make_models(dataset.feature_names, config.seed)
    metrics = {}
    # Limitar también BLAS/OpenMP, no solo el paralelismo propio de LightGBM.
    with threadpool_limits(limits=1):
        for name, model in models.items():
            model.fit(split.x_train, split.y_train)
            metrics[name] = evaluate_model(model, split.x_test, split.y_test)
    return TrainingResult(config, dataset, fingerprint, split, models, metrics)
