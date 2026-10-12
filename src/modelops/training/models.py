"""Pipelines nuevos, con parámetros fijados antes de observar el test."""

from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_models(feature_names: tuple[str, ...], seed: int) -> dict[str, Pipeline]:
    """Conservar selección y orden de columnas dentro del objeto persistido."""
    estimators = {
        "baseline": LogisticRegression(
            C=1.0, solver="lbfgs", max_iter=2000, random_state=seed
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            num_leaves=15,
            min_child_samples=20,
            random_state=seed,
            n_jobs=1,
            deterministic=True,
            force_col_wise=True,
            verbosity=-1,
        ),
    }
    models = {}
    for name, estimator in estimators.items():
        transformation = StandardScaler() if name == "baseline" else "passthrough"
        preprocess = ColumnTransformer(
            [("numeric", transformation, list(feature_names))],
            remainder="drop",
            verbose_feature_names_out=False,
        ).set_output(transform="pandas")
        models[name] = Pipeline([("preprocess", preprocess), ("model", estimator)])
    return models
