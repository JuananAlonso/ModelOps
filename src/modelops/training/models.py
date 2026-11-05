"""Pipelines con preprocesamiento ajustado exclusivamente en train."""

from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def make_models(
    feature_names: tuple[str, ...],
    seed: int,
    *,
    task: str = "classification",
    categorical_features: tuple[str, ...] = (),
    allow_missing: bool = False,
) -> dict[str, Pipeline]:
    estimators = {
        "baseline": (
            LinearRegression()
            if task == "regression"
            else LogisticRegression(
                C=1.0, solver="lbfgs", max_iter=2000, random_state=seed
            )
        ),
        "lightgbm": (LGBMRegressor if task == "regression" else LGBMClassifier)(
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
    numeric = [name for name in feature_names if name not in categorical_features]
    models = {}
    for name, estimator in estimators.items():
        transformation = StandardScaler() if name == "baseline" else "passthrough"
        if allow_missing:
            steps = [
                ("impute", SimpleImputer(strategy="median", keep_empty_features=True))
            ]
            if name == "baseline":
                steps.append(("scale", StandardScaler()))
            transformation = Pipeline(steps)
        transformers = []
        if numeric:
            transformers.append(("numeric", transformation, numeric))
        if categorical_features:
            transformers.append(
                (
                    "categorical",
                    Pipeline(
                        [
                            (
                                "impute",
                                SimpleImputer(
                                    strategy="constant",
                                    fill_value="__modelops_missing__",
                                    keep_empty_features=True,
                                ),
                            ),
                            (
                                "encode",
                                OneHotEncoder(
                                    handle_unknown="ignore", max_categories=100
                                ),
                            ),
                        ]
                    ),
                    list(categorical_features),
                )
            )
        preprocess = ColumnTransformer(
            transformers,
            remainder="drop",
            verbose_feature_names_out=False,
        ).set_output(
            transform="default"
            if allow_missing or categorical_features or name == "lightgbm"
            else "pandas"
        )
        models[name] = Pipeline([("preprocess", preprocess), ("model", estimator)])
    return models
