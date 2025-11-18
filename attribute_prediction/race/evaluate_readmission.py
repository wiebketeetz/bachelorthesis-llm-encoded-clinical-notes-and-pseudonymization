#!/usr/bin/env python3
"""Evaluate race prediction from precomputed note embeddings."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather
from sklearn import metrics
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    PredefinedSplit,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.utils import resample

try:
    from scipy import sparse  # type: ignore
except ImportError:  # pragma: no cover
    sparse = None  # type: ignore


LR_PARAM_CHOICES = {
    "C": [0.01, 0.1, 1.0, 10.0, 100.0],
    "class_weight": [None, "balanced"],
}


METRIC_FNS = {
    "auroc_per_class": lambda y_true, y_pred, sample_weight=None: metrics.roc_auc_score(
        y_true, y_pred, multi_class="ovr", average=None, sample_weight=sample_weight
    ).tolist(),
    "auroc": lambda y_true, y_pred, sample_weight=None: metrics.roc_auc_score(
        y_true, y_pred, multi_class="ovr", average="macro", sample_weight=sample_weight
    ),
    "brier": lambda y_true, y_pred, sample_weight=None: metrics.brier_score_loss(
        y_true, y_pred, sample_weight=sample_weight
    ),
    "auprc": lambda y_true, y_pred, sample_weight=None: metrics.average_precision_score(
        y_true, y_pred, average="macro", sample_weight=sample_weight
    ),
}


def load_numpy_array(path: Path, mmap_mode: Optional[str] = "r") -> np.ndarray:
    """Load numpy array handling pickled fallbacks with validation."""

    try:
        return np.load(path, mmap_mode=mmap_mode)
    except ValueError as err:
        message = str(err).lower()
        if "pickled data" not in message:
            raise

        logging.warning(
            "Reloading %s with allow_pickle=True; consider regenerating the artifact as numeric array",
            path,
        )
        arr = np.load(path, allow_pickle=True)
        if arr.dtype == object:
            raise ValueError(
                f"File {path} contains object dtype data; expected numeric embeddings. Regenerate the embeddings."
            )
        return np.asarray(arr)


def load_embeddings(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".feather":
        table = feather.read_table(path)
        if "embedding" not in table.column_names:
            raise ValueError(
                f"Feather file {path} must contain an 'embedding' column"
            )
        column = table.column("embedding").combine_chunks()
        if pa.types.is_fixed_size_list(column.type):
            list_size = column.type.list_size
            flat = column.values.to_numpy(zero_copy_only=False)
            embeddings = flat.reshape(len(column), list_size)
            return embeddings.astype(np.float32, copy=False)

        # Fallback: convert variable-length lists (should not happen with our exporter)
        embeddings = np.stack(column.to_pylist())
        return embeddings.astype(np.float32, copy=False)
    return load_numpy_array(path, mmap_mode="r")


def load_labels(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".feather":
        table = feather.read_table(path)
        if "label" in table.column_names:
            column = table.column("label")
        elif table.num_columns == 1:
            column = table.column(0)
        else:
            raise ValueError(
                f"Feather file {path} must have a 'label' column or exactly one column"
            )
        column = column.combine_chunks()
        labels = column.to_numpy()
        return np.asarray(labels, dtype=np.int32)

    return load_numpy_array(path, mmap_mode="r").astype(np.int32, copy=False)


def make_lr_pipeline(solver: str, max_iter: int) -> Pipeline:
    return Pipeline(
        [
            ("scaler", MaxAbsScaler()),
            (
                "model",
                LogisticRegression(
                    n_jobs=1,
                    penalty="l2",
                    tol=0.0001,
                    solver=solver,
                    max_iter=max_iter,
                    random_state=0
                ),
            ),
        ]
    )


def make_param_grid(c_values: List[float]) -> Dict[str, List[Any]]:
    return {
        "model__C": c_values,
        "model__class_weight": LR_PARAM_CHOICES["class_weight"],
    }


def simplify_best_params(best_params: Dict[str, Any], solver: str, max_iter: int) -> Dict[str, Any]:
    simplified: Dict[str, Any] = {"solver": solver, "max_iter": max_iter}
    for key, value in best_params.items():
        if key.startswith("model__"):
            simplified[key.split("__", 1)[1]] = value
        else:
            simplified[key] = value
    return simplified


def bootstrap_summary(
    metric_func,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    test_patient_ids: np.ndarray,
    bootstrap_samples: int,
    base_score: float,
) -> Dict[str, float]:
    unique_ids = np.array(sorted(set(test_patient_ids.tolist())))
    score_list = []
    for seed in range(bootstrap_samples):
        sample_ids = resample(unique_ids, replace=True, random_state=seed)
        counts = {pid: 0 for pid in unique_ids}
        for pid in sample_ids:
            counts[pid] += 1
        weights = np.array([counts[pid] for pid in test_patient_ids], dtype=float)
        score_val = metric_func(y_true, y_pred, sample_weight=weights)
        score_list.append(score_val)

    score_arr = np.array(score_list)
    return {
        "score": base_score,
        "std": float(score_arr.std(ddof=1)),
        "lower": float(np.percentile(score_arr, 2.5)),
        "mean": float(score_arr.mean()),
        "upper": float(np.percentile(score_arr, 97.5)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate readmission classifier on note embeddings")
    parser.add_argument(
        "--embeddings-train-path",
        type=Path,
        required=True,
        help="Path to embeddings file of train (.feather produced by precompute script or legacy .npy)",
    )
    parser.add_argument(
        "--embeddings-test-path",
        type=Path,
        required=True,
        help="Path to embeddings file of test (.feather produced by precompute script or legacy .npy)",
    )
    parser.add_argument(
        "--labels-train-path",
        type=Path,
        required=True,
        help="Path to labels file of train (.feather produced by precompute script or legacy .npy)",
    )
    parser.add_argument(
        "--labels-test-path",
        type=Path,
        required=True,
        help="Path to labels file of test (.feather produced by precompute script or legacy .npy)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.15,
        help="Fraction of rows reserved for the test set",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.15,
        help="Fraction reserved for validation (ignored when --cv-folds > 1)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting and shuffling")
    parser.add_argument(
        "--solver",
        type=str,
        default="lbfgs",
        choices=["lbfgs", "saga", "liblinear", "newton-cg"],
        help="Logistic regression solver",
    )
    parser.add_argument("--max-iter", type=int, default=1000, help="Maximum iterations for logistic regression")
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallelism used during hyper-parameter search")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional JSON file to store metrics and best hyper-parameters",
    )
    parser.add_argument(
        "--no-bootstrap",
        action="store_true",
        help="Skip bootstrap uncertainty estimation on the test set",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=1000,
        help="Number of bootstrap samples for interval estimation",
    )
    parser.add_argument(
        "--c-grid",
        type=float,
        nargs="*",
        default=None,
        help="Override the default C grid for logistic regression",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=0,
        help="Number of stratified folds for training-set cross-validation (0 disables)",
    )
    parser.add_argument(
        "--cv-seed",
        type=int,
        default=42,
        help="Random seed used for cross-validation shuffling",
    )
    return parser.parse_args()


def tune_hyperparams_predefined(
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    pipeline: Pipeline,
    param_grid: Dict[str, List[Any]],
    n_jobs: int,
) -> Tuple[Pipeline, Dict[str, Any]]:
    if sparse is not None and sparse.issparse(X_train):
        X = sparse.vstack([X_train, X_val])  # type: ignore[attr-defined]
    else:
        X = np.concatenate((X_train, X_val), axis=0)
    y = np.concatenate((y_train, y_val), axis=0)

    test_fold = -np.ones(X.shape[0], dtype=int)
    test_fold[X_train.shape[0] :] = 0
    
    clf = GridSearchCV(
        pipeline,
        param_grid,
        scoring="roc_auc_ovr",
        n_jobs=n_jobs,
        verbose=0,
        cv=PredefinedSplit(test_fold),
        refit=False,
    )
    clf.fit(X, y)

    best_params = clf.best_params_ if clf.best_params_ is not None else {}
    tuned_model = clone(pipeline)
    if best_params:
        tuned_model.set_params(**best_params)
    tuned_model.fit(X_train, y_train)
    return tuned_model, best_params


def tune_hyperparams_cv(
    X_train: np.ndarray,
    y_train: np.ndarray,
    pipeline: Pipeline,
    param_grid: Dict[str, List[Any]],
    cv_folds: int,
    seed: int,
    n_jobs: int,
) -> Tuple[Pipeline, Dict[str, Any]]:
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    clf = GridSearchCV(
        pipeline,
        param_grid,
        scoring="roc_auc_ovr",
        n_jobs=n_jobs,
        verbose=0,
        cv=cv,
        refit=False,
    )
    clf.fit(X_train, y_train)

    best_params = clf.best_params_ if clf.best_params_ is not None else {}
    tuned_model = clone(pipeline)
    if best_params:
        tuned_model.set_params(**best_params)
    tuned_model.fit(X_train, y_train)
    return tuned_model, best_params


def compute_cv_metrics(
    X: np.ndarray,
    y: np.ndarray,
    best_params: Dict[str, Any],
    solver: str,
    max_iter: int,
    cv_folds: int,
    seed: int,
) -> Dict[str, Dict[str, Any]]:
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    fold_scores: Dict[str, List[float]] = {metric: [] for metric in METRIC_FNS}

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), start=1):
        logging.info("Cross-validation fold %d/%d", fold, cv_folds)
        model = make_lr_pipeline(solver, max_iter)
        if best_params:
            model.set_params(**best_params)
        model.fit(X[train_idx], y[train_idx])
        proba = model.predict_proba(X[val_idx])
        for metric_name, func in METRIC_FNS.items():
            if metric_name != "auroc_per_class":
                score = func(y[val_idx], proba)
                fold_scores[metric_name].append(float(score))

    summary: Dict[str, Dict[str, Any]] = {}
    for metric_name, values in fold_scores.items():
        if metric_name != "auroc_per_class":
            arr = np.array(values, dtype=float)
            summary[metric_name] = {
                "folds": [float(v) for v in arr.tolist()],
                "mean": float(arr.mean()),
                "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            }

    return summary


def run_evaluation_lr(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    solver: str,
    n_jobs: int,
    max_iter: int,
    c_grid: List[float],
    bootstrap: bool,
    bootstrap_samples: int,
    test_patient_ids: Optional[np.ndarray] = None,
) -> Tuple[Pipeline, Dict[str, Any]]:
    logging.info("Train shape: X=%s, y=%s", X_train.shape, y_train.shape)
    logging.info("Val shape:   X=%s, y=%s", X_val.shape, y_val.shape)
    logging.info("Test shape:  X=%s, y=%s", X_test.shape, y_test.shape)


    #logging.info("Train prevalence: %.4f", np.mean(y_train))
    #logging.info("Val prevalence:   %.4f", np.mean(y_val))
    #logging.info("Test prevalence:  %.4f", np.mean(y_test))

    rng = np.random.default_rng(seed=X_train.shape[0])
    perm = rng.permutation(X_train.shape[0])
    X_train = X_train[perm]
    y_train = y_train[perm]

    pipeline = make_lr_pipeline(solver, max_iter)
    param_grid = make_param_grid(c_grid)
    tuned_model, best_params = tune_hyperparams_predefined(
        X_train,
        X_val,
        y_train,
        y_val,
        pipeline,
        param_grid,
        n_jobs=n_jobs,
    )
    logging.info("Best hyper-parameters: %s", best_params)

    y_train_proba = tuned_model.predict_proba(X_train)
    y_val_proba = tuned_model.predict_proba(X_val)
    y_test_proba = tuned_model.predict_proba(X_test)

    scores: Dict[str, Dict[str, float]] = {}
    for name, fn in METRIC_FNS.items():
        scores[name] = {
            "train": fn(y_train, y_train_proba, sample_weight=None),
            "val": fn(y_val, y_val_proba, sample_weight=None),
            "test": fn(y_test, y_test_proba, sample_weight=None),
        }
    
    ci_results: Dict[str, Dict[str, float]] = {}
    if bootstrap:
        if test_patient_ids is None:
            test_patient_ids = np.arange(len(y_test))
        for metric_name, fn in METRIC_FNS.items():
            base_score = scores[metric_name]["test"]
            ci_results[metric_name] = bootstrap_summary(
                fn,
                y_test,
                y_test_proba,
                test_patient_ids,
                bootstrap_samples,
                base_score,
            )

    return tuned_model, {"scores": scores, "bootstrap": ci_results, "best_params": best_params}


def run_evaluation_lr_cv(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    solver: str,
    n_jobs: int,
    max_iter: int,
    c_grid: List[float],
    bootstrap: bool,
    bootstrap_samples: int,
    cv_folds: int,
    cv_seed: int,
    test_patient_ids: Optional[np.ndarray] = None,
) -> Tuple[Pipeline, Dict[str, Any]]:
    logging.info("Train shape: X=%s, y=%s", X_train.shape, y_train.shape)
    logging.info("Test shape:  X=%s, y=%s", X_test.shape, y_test.shape)
    logging.info("Train prevalence: %.4f", np.mean(y_train))
    logging.info("Test prevalence:  %.4f", np.mean(y_test))

    rng = np.random.default_rng(seed=X_train.shape[0])
    perm = rng.permutation(X_train.shape[0])
    X_train = X_train[perm]
    y_train = y_train[perm]

    pipeline = make_lr_pipeline(solver, max_iter)
    param_grid = make_param_grid(c_grid)
    tuned_model, best_params = tune_hyperparams_cv(
        X_train,
        y_train,
        pipeline,
        param_grid,
        cv_folds=cv_folds,
        seed=cv_seed,
        n_jobs=n_jobs,
    )
    logging.info("Best hyper-parameters (CV): %s", best_params)

    cv_metrics = compute_cv_metrics(
        X_train,
        y_train,
        best_params,
        solver,
        max_iter,
        cv_folds,
        cv_seed,
    )

    y_train_proba = tuned_model.predict_proba(X_train)
    y_test_proba = tuned_model.predict_proba(X_test)

    scores: Dict[str, Dict[str, float]] = {}
    for name, fn in METRIC_FNS.items():
        entry: Dict[str, float] = {
            "train": fn(y_train, y_train_proba, sample_weight=None),
            "test": fn(y_test, y_test_proba, sample_weight=None),
        }
        cv_info = cv_metrics.get(name)
        if cv_info is not None:
            entry["cv_mean"] = cv_info["mean"]
            entry["cv_std"] = cv_info["std"]
        scores[name] = entry
    
    ci_results: Dict[str, Dict[str, float]] = {}
    if bootstrap:
        if test_patient_ids is None:
            test_patient_ids = np.arange(len(y_test))
        for metric_name, fn in METRIC_FNS.items():
            base_score = scores[metric_name]["test"]
            ci_results[metric_name] = bootstrap_summary(
                fn,
                y_test,
                y_test_proba,
                test_patient_ids,
                bootstrap_samples,
                base_score,
            )

    return tuned_model, {
        "scores": scores,
        "bootstrap": ci_results,
        "best_params": best_params,
        "cv": cv_metrics,
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    if not 0 < args.test_size < 1:
        raise ValueError("test_size must be in (0, 1)")
    if args.cv_folds < 0:
        raise ValueError("cv-folds must be non-negative")
    if args.cv_folds == 1:
        raise ValueError("cv-folds must be 0 or >= 2")
    if args.cv_folds <= 1:
        if args.val_size <= 0 or args.test_size + args.val_size >= 1:
            raise ValueError(
                "val_size must be positive and test_size + val_size < 1 when cross-validation is disabled"
            )
    elif args.val_size < 0:
        raise ValueError("val_size cannot be negative")


    if not args.embeddings_train_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {args.embeddings_train_path}")
    if not args.embeddings_test_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {args.embeddings_test_path}")
    if not args.labels_train_path.exists():
        raise FileNotFoundError(f"Labels file not found: {args.labels_train_path}")
    if not args.labels_test_path.exists():
        raise FileNotFoundError(f"Labels file not found: {args.labels_test_path}")

    embeddings_train = load_embeddings(args.embeddings_train_path)
    embeddings_test = load_embeddings(args.embeddings_test_path)
    labels_train = load_labels(args.labels_train_path)
    labels_test = load_labels(args.labels_test_path)
    if embeddings_train.shape[0] != labels_train.shape[0]:
        raise ValueError("Embeddings and labels of train have mismatched row counts")
    if embeddings_test.shape[0] != labels_test.shape[0]:
        raise ValueError("Embeddings and labels of test have mismatched row counts")

    c_grid = args.c_grid if args.c_grid is not None and len(args.c_grid) > 0 else LR_PARAM_CHOICES["C"]
    c_grid = list(c_grid)

    X_train_val = embeddings_train
    X_test = embeddings_test
    y_train_val = labels_train
    y_test = labels_test

    test_ids = np.arange(len(y_test))

    if args.cv_folds > 1:
        if args.val_size > 0:
            logging.info(
                "Ignoring val_size=%.3f because cross-validation with %d folds is enabled",
                args.val_size,
                args.cv_folds,
            )
        X_train, y_train = X_train_val, y_train_val
        best_model, results = run_evaluation_lr_cv(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            solver=args.solver,
            n_jobs=args.n_jobs,
            max_iter=args.max_iter,
            c_grid=c_grid,
            bootstrap=not args.no_bootstrap,
            bootstrap_samples=args.bootstrap_samples,
            cv_folds=args.cv_folds,
            cv_seed=args.cv_seed,
            test_patient_ids=test_ids,
        )
        effective_val_size = 0.0
    else:
        val_fraction = args.val_size / (1.0 - args.test_size)

        X_train_val = embeddings_train
        X_test = embeddings_test
        y_train_val = labels_train
        y_test = labels_test
        
        best_model, results = run_evaluation_lr(
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            solver=args.solver,
            n_jobs=args.n_jobs,
            max_iter=args.max_iter,
            c_grid=c_grid,
            bootstrap=not args.no_bootstrap,
            bootstrap_samples=args.bootstrap_samples,
            test_patient_ids=test_ids,
        )
        effective_val_size = args.val_size

    summary = {
        "embeddings_train_path": str(args.embeddings_train_path.resolve()),
        "labels_train_path": str(args.labels_train_path.resolve()),
        "solver": args.solver,
        "max_iter": args.max_iter,
        "n_jobs": args.n_jobs,
        "val_size": effective_val_size,
        "test_size": args.test_size,
        "seed": args.seed,
        "metrics": results["scores"],
        "bootstrap": results["bootstrap"],
        "c_grid": c_grid,
        "best_params": simplify_best_params(
            results.get("best_params", {}), args.solver, args.max_iter
        ),
        "cv": results.get("cv"),
        "cv_folds": args.cv_folds,
        "cv_seed": args.cv_seed,
    }

    logging.info("Evaluation metrics (test): %s", summary["metrics"])

    if args.output_path is not None:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        args.output_path.write_text(json.dumps(summary, indent=2))
        logging.info("Saved evaluation summary to %s", args.output_path)
    else:
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
