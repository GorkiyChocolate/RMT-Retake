#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import imblearn
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

LOG = logging.getLogger("fraud_benchmark")
SEED = 42
CONDITIONS = ("original", "ros", "rus", "smote")
MODELS = ("logistic_regression", "random_forest")
TARGETS = (
    "is_fraud", "isfraud", "fraud", "class", "label", "target",
    "fraud_label", "fraudulent", "fraud_flag", "is_fraudulent",
    "y", "outcome", "response", "default",
)
DROP_NAMES = {
    "id", "index", "unnamed0", "rowid", "transactionid", "transactionnumber", "transnum",
    "customerid", "userid", "accountid", "cardnumber", "ccnum", "creditcardnumber",
    "merchantid", "personid", "zipcode", "zip", "postalcode", "mcc", "categoryid", "name", "firstname", "lastname", "first", "last",
    "prediction", "predictedfraud", "fraudscore", "fraudprobability", "fraudrisk",
    "isflaggedfraud", "isflaggedfraudulent", "fraudtype", "risklabel",
}


def normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def detect_target(df: pd.DataFrame, override: str | None, label: str) -> str:
    if override:
        exact = next((c for c in df.columns if c == override or normalize(c) == normalize(override)), None)
        if not exact:
            raise ValueError(f"{label}: --target={override!r} not found. Columns: {list(df.columns)}")
        return exact
    for name in TARGETS:
        exact = next((c for c in df.columns if normalize(c) == normalize(name)), None)
        if exact:
            return exact
    raise ValueError(
        f"{label}: target column not detected; use --target {label}=COLUMN "
        f"or a legacy target option. Columns: {list(df.columns)}"
    )


def binary_target(series: pd.Series, label: str) -> pd.Series:
    clean = series.replace({True: 1, False: 0})
    if not pd.api.types.is_numeric_dtype(clean):
        words = clean.astype("string").str.strip().str.lower()
        mapped = words.map({
            "0": 0, "1": 1, "false": 0, "true": 1, "no": 0, "yes": 1,
            "legitimate": 0, "legit": 0, "normal": 0, "non-fraud": 0,
            "not fraud": 0, "fraud": 1, "fraudulent": 1, "fraud transaction": 1,
        })
        clean = mapped
    clean = pd.to_numeric(clean, errors="coerce")
    if clean.isna().any() or not set(clean.unique()).issubset({0, 1}):
        raise ValueError(f"{label}: target must contain only 0 and 1 (or supported text equivalents).")
    if clean.nunique() != 2:
        raise ValueError(f"{label}: target must contain both classes, but got {clean.value_counts().to_dict()}")
    return clean.astype(np.int8)


def read_csv(path: Path, **kw) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"CSV not found: {path}")
    LOG.info("Loading %s", path)
    return pd.read_csv(path, low_memory=False, **kw)


def numeric_features(df: pd.DataFrame, target: str, label: str) -> pd.DataFrame:
    """Use numeric features only: regular SMOTE cannot interpolate categorical labels.

    Known ID and post-outcome variables are excluded by name. Other suspected
    leakage must be reviewed manually using the emitted features.json.
    """
    features: dict[str, pd.Series] = {}
    ignored = []
    target_names = {normalize(v) for v in TARGETS}
    for col in df.columns:
        key = normalize(col)
        if col == target or key in DROP_NAMES or key in target_names or key.startswith("unnamed"):
            continue
        s = df[col]
        if pd.api.types.is_bool_dtype(s):
            values = s.astype(float)
        elif pd.api.types.is_numeric_dtype(s):
            values = pd.to_numeric(s, errors="coerce")
        else:
            converted = pd.to_numeric(s, errors="coerce")
            # Only columns that are *almost entirely numeric* are retained.
            if converted.notna().mean() < 0.98:
                ignored.append(col)
                continue
            values = converted
        values = values.replace([np.inf, -np.inf], np.nan)
        if values.notna().sum() == 0:
            ignored.append(col)
            continue
        features[col] = values.astype(np.float32)
    if not features:
        raise ValueError(f"{label}: no usable numeric features. Columns: {list(df.columns)}")
    if ignored:
        LOG.info("%s: excluding non-numeric or empty columns: %s", label, ignored)
    X = pd.DataFrame(features, index=df.index)
    if X.columns.duplicated().any():
        raise ValueError(f"{label}: duplicate feature names")
    LOG.info("%s: selected %d numeric features: %s", label, X.shape[1], list(X.columns))
    return X


def kartik_csv(path: Path) -> pd.DataFrame:
    # Load only required columns. This cuts memory dramatically compared to
    # reading personal names, addresses, and other high-cardinality strings.
    header = pd.read_csv(path, nrows=0).columns.tolist()
    desired = {"is_fraud", "amt", "city_pop", "lat", "long", "merch_lat", "merch_long",
               "trans_date_trans_time", "dob", "unix_time"}
    cols = [c for c in header if c in desired]
    if "is_fraud" not in cols:
        raise ValueError(f"Kartik CSV {path} has no is_fraud column. Found: {header}")
    return read_csv(path, usecols=cols)


def kartik_features(df: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(index=df.index)
    for col in ("amt", "city_pop", "lat", "long", "merch_lat", "merch_long"):
        if col in df:
            result[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float32)
    if "amt" in result:
        result["log1p_amt"] = np.log1p(result["amt"].clip(lower=0))
    if "city_pop" in result:
        result["log1p_city_pop"] = np.log1p(result["city_pop"].clip(lower=0))

    timestamp = pd.to_datetime(df["trans_date_trans_time"], errors="coerce") if "trans_date_trans_time" in df else None
    if timestamp is None and "unix_time" in df:
        timestamp = pd.to_datetime(df["unix_time"], unit="s", errors="coerce", utc=True)
    if timestamp is not None:
        result["transaction_hour"] = timestamp.dt.hour.astype(np.float32)
        result["transaction_weekday"] = timestamp.dt.dayofweek.astype(np.float32)
        result["transaction_month"] = timestamp.dt.month.astype(np.float32)
        result["is_weekend"] = timestamp.dt.dayofweek.isin([5, 6]).astype(np.float32)
        if "dob" in df:
            dob = pd.to_datetime(df["dob"], errors="coerce")
            # Make both dates timezone-naive for subtraction, if needed.
            ts_naive = timestamp.dt.tz_localize(None) if timestamp.dt.tz is not None else timestamp
            age = (ts_naive - dob).dt.days / 365.25
            result["cardholder_age"] = age.where(age.between(0, 110)).astype(np.float32)

    if all(x in result for x in ("lat", "long", "merch_lat", "merch_long")):
        lat1 = np.radians(result["lat"].to_numpy(dtype=np.float64))
        lon1 = np.radians(result["long"].to_numpy(dtype=np.float64))
        lat2 = np.radians(result["merch_lat"].to_numpy(dtype=np.float64))
        lon2 = np.radians(result["merch_long"].to_numpy(dtype=np.float64))
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        result["customer_merchant_km"] = (6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))).astype(np.float32)
    if result.empty:
        raise ValueError("Kartik: could not build numeric features.")
    result = result.replace([np.inf, -np.inf], np.nan)
    return result


def subsample_training(X: pd.DataFrame, y: pd.Series, max_rows: int, seed: int):
    if max_rows <= 0 or len(y) <= max_rows:
        return X.reset_index(drop=True), y.reset_index(drop=True), "full training set"
    Xsub, _, ysub, _ = train_test_split(
        X, y, train_size=max_rows, random_state=seed, stratify=y
    )
    if ysub.value_counts().min() < 2:
        raise ValueError("Training cap leaves fewer than 2 fraud/legitimate examples. Raise --max-train-rows.")
    return Xsub.reset_index(drop=True), ysub.reset_index(drop=True), f"stratified train sample of {max_rows:,}"


def load_dataset(name: str, args, path: Path | None = None, target_override: str | None = None):
    if name == "kartik" and path is None:
        train = kartik_csv(Path(args.kartik_train))
        test = kartik_csv(Path(args.kartik_test))
        ytr = binary_target(train["is_fraud"], "kartik train")
        yte = binary_target(test["is_fraud"], "kartik test")
        Xtr, Xte = kartik_features(train), kartik_features(test)
        cols = Xtr.columns.tolist()
        if cols != Xte.columns.tolist():
            raise ValueError(f"Kartik: train/test feature mismatch: {set(cols) ^ set(Xte.columns)}")
        train_total, test_total = len(ytr), len(yte)
        mode = "provided chronological train/test files"
        del train, test
    else:
        if path is None:
            path = Path(args.european if name == "european" else args.miadul)
        raw = read_csv(path)
        specified_target = target_override
        if specified_target is None:
            specified_target = args.european_target if name == "european" else args.miadul_target
        target = detect_target(raw, specified_target, name)
        y = binary_target(raw[target], name)
        X = numeric_features(raw, target, name)
        # A single fixed stratified holdout for these single-file datasets.
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=args.test_size, random_state=args.seed, stratify=y
        )
        mode = f"stratified {1-args.test_size:.0%}/{args.test_size:.0%} random split"
        train_total, test_total = len(ytr), len(yte)
        del raw, X, y

    train_counts_full = {str(k): int(v) for k, v in ytr.value_counts().items()}
    Xtr, ytr, cap_note = subsample_training(Xtr, ytr, args.max_train_rows, args.seed)
    Xte, yte = Xte.reset_index(drop=True), yte.reset_index(drop=True)
    if ytr.value_counts().min() < 2:
        raise ValueError(f"{name}: not enough minority samples for SMOTE")
    summary = {
        "dataset": name,
        "train_size_original": int(train_total),
        "test_size": int(test_total),
        "train_class_counts_full": train_counts_full,
        "train_size_used": int(len(ytr)),
        "train_class_counts_used": {str(k): int(v) for k, v in ytr.value_counts().items()},
        "test_class_counts": {str(k): int(v) for k, v in yte.value_counts().items()},
        "split": mode,
        "training_subset": cap_note,
        "feature_count": int(Xtr.shape[1]),
        "features": Xtr.columns.tolist(),
    }
    LOG.info("%s: train used=%s (fraud=%s); test=%s (fraud=%s), features=%s",
             name, len(ytr), int(ytr.sum()), len(yte), int(yte.sum()), Xtr.shape[1])
    return Xtr, Xte, ytr, yte, summary


def make_pipeline(method: str, model: str, minority_count: int, args):
    steps = [
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]
    if method == "ros":
        steps.append(("resample", RandomOverSampler(random_state=args.seed)))
    elif method == "rus":
        steps.append(("resample", RandomUnderSampler(random_state=args.seed)))
    elif method == "smote":
        if minority_count < 2:
            raise ValueError("SMOTE needs at least 2 minority examples")
        steps.append(("resample", SMOTE(random_state=args.seed, k_neighbors=min(5, minority_count-1))))
    if model == "logistic_regression":
        clf = LogisticRegression(max_iter=600, solver="lbfgs", class_weight=None, random_state=args.seed)
    else:
        clf = RandomForestClassifier(
            n_estimators=args.rf_trees, max_depth=args.rf_depth, min_samples_leaf=2,
            random_state=args.seed, n_jobs=args.jobs, class_weight=None,
        )
    steps.append(("classifier", clf))
    return Pipeline(steps)


def score_predictions(y, scores, threshold=0.5):
    predicted = (scores >= threshold).astype(int)
    cm = confusion_matrix(y, predicted, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in cm.ravel())
    return {
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, scores)),
        # Average Precision (AP) is a standard summary of the PR curve.
        "pr_auc_ap": float(average_precision_score(y, scores)),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def save_confusion(y, scores, output: Path, title: str):
    cm = confusion_matrix(y, scores >= 0.5, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5.0, 4.3))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["Legitimate", "Fraud"])
    ax.set_yticks([0, 1], ["Legitimate", "Fraud"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    for (i, j), value in np.ndenumerate(cm):
        ax.text(j, i, f"{value:,}", va="center", ha="center", fontsize=11)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def save_curves(y, scores, output: Path, title: str):
    p, r, _ = precision_recall_curve(y, scores)
    fig, ax = plt.subplots(figsize=(6.2, 4.1))
    ax.plot(r, p, label=f"Average precision: {average_precision_score(y, scores):.4f}")
    ax.set(xlabel="Recall", ylabel="Precision", title=title, xlim=(0, 1), ylim=(0, 1))
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def save_comparison(results: pd.DataFrame, output: Path, dataset: str):
    metrics = ("f1", "recall", "precision", "pr_auc_ap")
    for model in MODELS:
        sub = results[(results.dataset == dataset) & (results.model == model)].copy()
        if sub.empty:
            continue
        sub = sub.set_index("condition").reindex(CONDITIONS)
        fig, ax = plt.subplots(figsize=(9, 4.8))
        xx = np.arange(len(CONDITIONS))
        width = 0.19
        for i, metric in enumerate(metrics):
            ax.bar(xx + (i - 1.5) * width, sub[metric].to_numpy(), width, label=metric)
        ax.set_xticks(xx, CONDITIONS)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Score on untouched test set")
        ax.set_title(f"{dataset}: {model}")
        ax.legend(loc="upper right", ncol=2)
        fig.tight_layout()
        fig.savefig(output / f"comparison_{model}.png", dpi=160)
        plt.close(fig)


def cross_validate_optional(pipe, X, y, method: str, args):
    if args.cv_folds == 0:
        return {}
    n = args.cv_folds
    if y.value_counts().min() < n:
        raise ValueError(f"Cannot run {n}-fold CV with fewer than {n} fraud cases")
    splitter = StratifiedKFold(n_splits=n, shuffle=True, random_state=args.seed)
    fold_scores = []
    for idx, (tr, val) in enumerate(splitter.split(X, y), start=1):
        Xtrain, Xval = X.iloc[tr], X.iloc[val]
        ytrain, yval = y.iloc[tr], y.iloc[val]
        fold_pipe = make_pipeline(method, pipe.named_steps["classifier"].__class__.__name__
                                  .replace("LogisticRegression", "logistic_regression")
                                  .replace("RandomForestClassifier", "random_forest"),
                                  int(ytrain.value_counts().min()), args)
        fold_pipe.fit(Xtrain, ytrain)
        proba = fold_pipe.predict_proba(Xval)[:, 1]
        fold_scores.append(score_predictions(yval, proba))
        LOG.info(" CV fold %s/%s: F1=%.4f PR-AUC(AP)=%.4f", idx, n,
                 fold_scores[-1]["f1"], fold_scores[-1]["pr_auc_ap"])
    return {
        f"cv_{metric}_mean": float(np.mean([f[metric] for f in fold_scores]))
        for metric in ("f1", "precision", "recall", "roc_auc", "pr_auc_ap")
    } | {
        f"cv_{metric}_std": float(np.std([f[metric] for f in fold_scores], ddof=1))
        for metric in ("f1", "precision", "recall", "roc_auc", "pr_auc_ap")
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark Original/ROS/RUS/SMOTE with LR/RF on datasets in a directory."
    )
    parser.add_argument("--data-dir", default="data",
                        help="Directory containing CSV datasets (default: data)")
    parser.add_argument("--kartik-train", default=None, help="Optional path to fraudTrain.csv")
    parser.add_argument("--kartik-test", default=None, help="Optional path to fraudTest.csv")
    parser.add_argument("--european", default=None, help="Optional path to a European CSV")
    parser.add_argument("--miadul", default=None, help="Optional path to a Miadul CSV")
    parser.add_argument("--miadul-target", default=None, help="Manually specify fraud label column for Miadul")
    parser.add_argument("--european-target", default=None, help="Manually specify fraud label column for European")
    parser.add_argument("--out", default="results", help="Output directory")
    parser.add_argument("--max-train-rows", type=int, default=200000,
                        help="Stratified training cap per dataset (default 200000; 0 means full training set)")
    parser.add_argument("--test-size", type=float, default=0.20,
                        help="Holdout fraction for European and Miadul; Kartik uses provided test (default .20)")
    parser.add_argument("--cv-folds", type=int, default=0,
                        help="Optional stratified CV on TRAIN ONLY (0 disabled, 3 or 5 enabled; costly)")
    parser.add_argument("--rf-trees", type=int, default=80, help="Number of trees (default 80)")
    parser.add_argument("--rf-depth", type=int, default=16, help="Maximum RF tree depth (default 16)")
    parser.add_argument("--jobs", type=int, default=4, help="RF worker processes/threads (default 4)")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if bool(args.kartik_train) != bool(args.kartik_test):
        parser.error("--kartik-train and --kartik-test must be supplied together")
    if not (0 < args.test_size < 1):
        parser.error("--test-size must be between 0 and 1")
    if args.cv_folds not in (0, 3, 5):
        parser.error("--cv-folds must be 0, 3 or 5")
    if args.max_train_rows != 0 and args.max_train_rows < 50:
        parser.error("--max-train-rows must be 0 or >= 50")
    return args


def discover_datasets(args):
    """Build dataset inputs from the data directory, keeping Kartik as a pair."""
    explicit = (args.kartik_train, args.kartik_test, args.european, args.miadul)
    if all(explicit):
        return [
            ("kartik", None, None),
            ("european", Path(args.european).expanduser(), args.european_target),
            ("miadul", Path(args.miadul).expanduser(), args.miadul_target),
        ]
    data_dir = Path(args.data_dir).expanduser()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    paths = sorted(data_dir.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No CSV datasets found in {data_dir}")
    train = Path(args.kartik_train).expanduser() if args.kartik_train else data_dir / "fraudTrain.csv"
    test = Path(args.kartik_test).expanduser() if args.kartik_test else data_dir / "fraudTest.csv"
    datasets = []
    if train.is_file() and test.is_file():
        datasets.append(("kartik", None, None))
        paths = [p for p in paths if p.resolve() not in {train.resolve(), test.resolve()}]
        args.kartik_train, args.kartik_test = str(train), str(test)
    for path in paths:
        name = path.stem
        if name == "european" and args.european_target:
            target = args.european_target
        elif name == "miadul" and args.miadul_target:
            target = args.miadul_target
        else:
            target = None
        datasets.append((name, path, target))
    return datasets


def main():
    args = parse_args()
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(out / "experiment.log", encoding="utf-8")])
    config = vars(args).copy()
    metadata = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "command_arguments": config,
        "python": sys.version,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "imbalanced_learn": imblearn.__version__,
        "test_threshold": 0.5,
        "strategy": "Same dataset split, numeric features, fixed LR/RF parameters, balance training only",
        "note": "Default capped training is a resource-conscious exploratory experiment; use --max-train-rows 0 for full training.",
    }
    (out / "experiment_config.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    all_results: list[dict] = []
    all_summaries: list[dict] = []
    failures: list[dict] = []
    datasets = discover_datasets(args)
    for name, path, target_override in datasets:
        dest = out / name
        dest.mkdir(exist_ok=True)
        LOG.info("===== START DATASET: %s =====", name)
        try:
            Xtr, Xte, ytr, yte, info = load_dataset(name, args, path, target_override)
            all_summaries.append(info)
            (dest / "dataset_summary.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
            pd.Series(info["train_class_counts_used"]).to_csv(dest / "class_distribution_train.csv", header=["count"])
        except Exception as exc:
            LOG.exception("Dataset %s failed to load", name)
            failures.append({"dataset": name, "condition": "load", "model": "", "error": repr(exc)})
            continue
        for model in MODELS:
            for condition in CONDITIONS:
                LOG.info("===== %s / %s / %s =====", name, model, condition)
                start = time.perf_counter()
                try:
                    pipe = make_pipeline(condition, model, int(ytr.value_counts().min()), args)
                    cv_metrics = cross_validate_optional(pipe, Xtr, ytr, condition, args)
                    pipe.fit(Xtr, ytr)
                    proba = pipe.predict_proba(Xte)[:, 1]
                    result = {"dataset": name, "model": model, "condition": condition,
                              "train_rows": int(len(ytr)), "test_rows": int(len(yte)),
                              "features": int(Xtr.shape[1]),
                              **score_predictions(yte, proba), **cv_metrics,
                              "seconds": round(time.perf_counter() - start, 2)}
                    all_results.append(result)
                    LOG.info("RESULT F1=%.4f, Recall=%.4f, Precision=%.4f, ROC-AUC=%.4f, PR-AUC(AP)=%.4f, time=%.1fs",
                             result["f1"], result["recall"], result["precision"], result["roc_auc"], result["pr_auc_ap"], result["seconds"])
                    prefix = f"{model}_{condition}"
                    save_confusion(yte, proba, dest / f"cm_{prefix}.png", f"{name} - {prefix}")
                    save_curves(yte, proba, dest / f"pr_curve_{prefix}.png", f"PR Curve: {name}/{prefix}")
                    pd.DataFrame({"actual": yte.to_numpy(), "probability_fraud": proba}).to_csv(
                        dest / f"test_predictions_{prefix}.csv", index=False)
                    pd.DataFrame(all_results).to_csv(out / "all_results.csv", index=False)
                except Exception as exc:
                    LOG.exception("Failed %s / %s / %s", name, model, condition)
                    failures.append({"dataset": name, "model": model, "condition": condition, "error": repr(exc)})
            save_comparison(pd.DataFrame(all_results), dest, name)
        # Release dataset before moving on to next.
        del Xtr, Xte, ytr, yte
    (out / "all_dataset_summaries.json").write_text(json.dumps(all_summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "failures.json").write_text(json.dumps(failures, indent=2, ensure_ascii=False), encoding="utf-8")
    if all_results:
        final = pd.DataFrame(all_results)
        final.to_csv(out / "all_results.csv", index=False)
        LOG.info("Finished: %s successful experiments, %s errors. Output: %s", len(final), len(failures), out)
        display_cols = ["dataset", "model", "condition", "precision", "recall", "f1", "roc_auc", "pr_auc_ap"]
        print("\n" + final[display_cols].to_string(index=False, float_format=lambda f: f"{f:.4f}"))
    else:
        LOG.error("No experiments completed. Inspect experiment.log and failures.json")
    expected_results = len(datasets) * len(MODELS) * len(CONDITIONS)
    if failures or len(all_results) != expected_results:
        sys.exit(1)


if __name__ == "__main__":
    main()
