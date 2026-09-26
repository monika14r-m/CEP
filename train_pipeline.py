
"""
ML-IDS full model-suite training pipeline.

Dataset: NSL-KDD (binary normal vs attack)
Supervised suite:
  Logistic Regression, Gaussian Naive Bayes, KNN, Decision Tree,
  Linear SVM, Random Forest, Extra Trees, Gradient Boosting,
  AdaBoost, HistGradientBoosting, XGBoost, LightGBM, CatBoost, MLP.
Anomaly suite:
  Isolation Forest, One-Class SVM, Local Outlier Factor.

Imbalance handling is represented explicitly in the model metadata:
  class-weighted models, SMOTE variants, sample-weighted boosting, and
  normal-only anomaly detectors.

The pipeline writes every successful model to artifacts/ and is consumed by
app.py. External boosting libraries are imported defensively so the pipeline
still starts if one is unavailable.
"""
import json
import os
import re
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier,
    HistGradientBoostingClassifier, IsolationForest, RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc, confusion_matrix, f1_score, precision_recall_curve, precision_score,
    recall_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.svm import LinearSVC, OneClassSVM
from sklearn.tree import DecisionTreeClassifier

from model_wrappers import AnomalyProbabilityWrapper

np.random.seed(42)
ARTIFACT_DIR = Path(__file__).parent / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)

COLS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
    "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty",
]
TRAIN_URL = "https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/KDDTrain%2B.txt"
TEST_URL = "https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/KDDTest%2B.txt"
RAW_FEATURE_COLS = [c for c in COLS if c not in ("label", "difficulty")]


def load_data():
    train_df = pd.read_csv(TRAIN_URL, names=COLS)
    test_df = pd.read_csv(TEST_URL, names=COLS)
    if train_df.shape[1] != len(COLS) or test_df.shape[1] != len(COLS):
        raise RuntimeError("NSL-KDD column count mismatch")
    return train_df, test_df


def plot_class_distribution(train_df):
    counts = (train_df["label"] != "normal").map({True: "Attack", False: "Normal"}).value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(counts.index, counts.values)
    ax.set_title("NSL-KDD: Normal vs Attack")
    ax.set_ylabel("Connections")
    fig.tight_layout()
    fig.savefig(ARTIFACT_DIR / "class_distribution.png", dpi=120)
    plt.close(fig)


def preprocess(train_df, test_df):
    train_df = train_df.drop(columns=["difficulty"]).copy()
    test_df = test_df.drop(columns=["difficulty"]).copy()
    train_df = train_df.drop_duplicates()

    numeric_cols = train_df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    for c in numeric_cols:
        train_df[c] = train_df[c].replace(-1, np.nan)
        test_df[c] = test_df[c].replace(-1, np.nan)

    medians = {}
    for c in numeric_cols:
        medians[c] = float(train_df[c].median())
        train_df[c] = train_df[c].fillna(medians[c])
        test_df[c] = test_df[c].fillna(medians[c])

    cat_cols = ["protocol_type", "service", "flag"]
    modes = {}
    encoders = {}
    for c in cat_cols:
        modes[c] = str(train_df[c].mode()[0])
        train_df[c] = train_df[c].fillna(modes[c])
        test_df[c] = test_df[c].fillna(modes[c])
        le = LabelEncoder()
        le.fit(train_df[c])
        train_df[c] = le.transform(train_df[c])
        test_df[c] = test_df[c].map(lambda x: x if x in set(le.classes_) else le.classes_[0])
        test_df[c] = le.transform(test_df[c])
        encoders[c] = le

    train_df["binary_label"] = (train_df["label"] != "normal").astype(int)
    test_df["binary_label"] = (test_df["label"] != "normal").astype(int)

    skewed = [c for c in ["duration", "src_bytes", "dst_bytes", "count", "srv_count"] if c in train_df]
    normal_mask = train_df["binary_label"] == 0
    for c in skewed:
        train_df[c] = train_df[c].astype(float)
        test_df[c] = test_df[c].astype(float)
        lo, hi = train_df.loc[normal_mask, c].quantile([0.01, 0.99])
        train_df.loc[normal_mask, c] = train_df.loc[normal_mask, c].clip(lo, hi)
        train_df[c] = np.log1p(train_df[c].clip(lower=0))
        test_df[c] = np.log1p(test_df[c].clip(lower=0))

    feature_cols = [c for c in train_df.columns if c not in ("label", "binary_label")]
    corr = train_df[feature_cols].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    redundant = [c for c in upper.columns if any(upper[c] > 0.95)]
    feature_cols = [c for c in feature_cols if c not in redundant]

    meta = {
        "medians": medians, "modes": modes, "encoders": encoders,
        "skewed_cols": skewed, "feature_cols": feature_cols,
        "redundant_dropped": redundant,
    }
    return train_df, test_df, meta


def metrics(y_true, y_pred, y_score):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    fpr = fp / max(fp + tn, 1)
    fnr = fn / max(fn + tp, 1)
    p, r, _ = precision_recall_curve(y_true, y_score)
    return {
        "cm": cm.tolist(), "precision": float(precision), "recall": float(recall),
        "f1": float(f1), "fpr": float(fpr), "fnr": float(fnr),
        "pr_auc": float(auc(r, p)),
    }


def slugify(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def build_samples(raw_test, per_label=3, max_total=30):
    rows = []
    for label, grp in raw_test.groupby("label"):
        take = grp.sample(min(len(grp), per_label), random_state=42)
        for _, row in take.iterrows():
            rec = {c: (v.item() if hasattr(v, "item") else v) for c, v in row[RAW_FEATURE_COLS].items()}
            rec["attack_type"] = label
            rec["true_label"] = "normal" if label == "normal" else "attack"
            rows.append(rec)
        if len(rows) >= max_total:
            break
    return rows[:max_total]


def fit_score_supervised(name, model, X_train, y_train, X_test, y_test, sample_weight=None):
    if sample_weight is not None:
        try:
            model.fit(X_train, y_train, sample_weight=sample_weight)
        except TypeError:
            model.fit(X_train, y_train)
    else:
        model.fit(X_train, y_train)
    score = model.predict_proba(X_test)[:, 1]
    pred = (score >= 0.5).astype(int)
    return model, metrics(y_test, pred, score)


def main():
    print("Loading NSL-KDD...")
    train_raw, test_raw = load_data()
    plot_class_distribution(train_raw)
    train_df, test_df, meta = preprocess(train_raw, test_raw)

    feature_cols = meta["feature_cols"]
    X_train = train_df[feature_cols]
    y_train = train_df["binary_label"]
    X_test = test_df[feature_cols]
    y_test = test_df["binary_label"]

    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_smote, y_smote = smote.fit_resample(X_train_scaled, y_train)
    X_smote = X_smote.astype(np.float32)
    y_smote = np.asarray(y_smote)

    counts = np.bincount(y_train)
    ratio = float(counts[0] / max(counts[1], 1))
    sample_weight = np.where(y_train.to_numpy() == 1, ratio, 1.0)

    models = {}
    # Baselines
    models["Logistic Regression"] = (LogisticRegression(max_iter=500, class_weight="balanced"), "baseline", "class_weight")
    models["Gaussian Naive Bayes"] = (GaussianNB(), "baseline", "none")
    models["KNN (k=5)"] = (KNeighborsClassifier(n_neighbors=5, n_jobs=-1), "baseline", "none")
    models["SVM (linear)"] = (
        CalibratedClassifierCV(LinearSVC(class_weight="balanced", max_iter=2000, random_state=42), cv=3),
        "baseline", "class_weight",
    )
    # Our / candidate models
    models["Decision Tree"] = (DecisionTreeClassifier(class_weight="balanced", max_depth=10, random_state=42), "ours", "class_weight")
    models["Random Forest"] = (
        RandomForestClassifier(n_estimators=100, class_weight="balanced", max_depth=14, n_jobs=2, random_state=42),
        "ours", "class_weight",
    )
    models["Extra Trees"] = (
        ExtraTreesClassifier(n_estimators=100, class_weight="balanced", max_depth=16, n_jobs=2, random_state=42),
        "ours", "class_weight",
    )
    models["Gradient Boosting"] = (
        GradientBoostingClassifier(n_estimators=100, learning_rate=0.08, max_depth=3, random_state=42),
        "ours", "sample_weight",
    )
    models["AdaBoost"] = (AdaBoostClassifier(n_estimators=100, learning_rate=0.08, random_state=42), "ours", "sample_weight")
    models["HistGradientBoosting"] = (
        HistGradientBoostingClassifier(max_iter=100, learning_rate=0.08, max_leaf_nodes=31, random_state=42),
        "ours", "sample_weight",
    )
    models["MLP Neural Network"] = (
        MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=45, early_stopping=True, random_state=42),
        "ours", "SMOTE",
    )
    models["Random Forest (SMOTE)"] = (
        RandomForestClassifier(n_estimators=100, max_depth=14, n_jobs=2, random_state=42),
        "ours", "SMOTE",
    )
    models["Extra Trees (SMOTE)"] = (
        ExtraTreesClassifier(n_estimators=100, max_depth=16, n_jobs=2, random_state=42),
        "ours", "SMOTE",
    )

    # Optional external gradient boosters. They are required by requirements.txt,
    # but imports remain defensive for local environments.
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = (
            XGBClassifier(n_estimators=120, max_depth=6, learning_rate=0.08,
                          subsample=.8, colsample_bytree=.8, scale_pos_weight=ratio,
                          eval_metric="logloss", n_jobs=2, tree_method="hist", random_state=42),
            "ours", "class_weight",
        )
    except Exception as e:
        print("Skipping XGBoost:", e)

    try:
        from lightgbm import LGBMClassifier
        models["LightGBM"] = (
            LGBMClassifier(n_estimators=120, num_leaves=31, max_depth=8,
                           learning_rate=.08, class_weight="balanced",
                           verbosity=-1, n_jobs=2, random_state=42),
            "ours", "class_weight",
        )
    except Exception as e:
        print("Skipping LightGBM:", e)

    try:
        from catboost import CatBoostClassifier
        models["CatBoost"] = (
            CatBoostClassifier(iterations=120, depth=7, learning_rate=.08,
                               loss_function="Logloss", verbose=False, thread_count=2,
                               class_weights=[1.0, ratio], random_seed=42),
            "ours", "class_weight",
        )
    except Exception as e:
        print("Skipping CatBoost:", e)

    results = {}
    for name, (mdl, group, strategy) in models.items():
        print("Training:", name, " [", strategy, "]")
        if strategy == "SMOTE":
            fitted, met = fit_score_supervised(name, mdl, X_smote, y_smote, X_test_scaled, y_test)
        elif strategy == "sample_weight":
            fitted, met = fit_score_supervised(name, mdl, X_train_scaled, y_train, X_test_scaled, y_test, sample_weight)
        else:
            fitted, met = fit_score_supervised(name, mdl, X_train_scaled, y_train, X_test_scaled, y_test)
        results[name] = {"model": fitted, "metrics": met, "group": group, "strategy": strategy}

    # Anomaly models learn NORMAL traffic only. Risk scores are adapters, not
    # calibrated probabilities.
    normal_train = X_train_scaled[y_train.to_numpy() == 0]
    normal_train = normal_train[:20000]
    anomaly_train = normal_train[:5000]

    anomaly_specs = [
        ("Isolation Forest", IsolationForest(n_estimators=120, contamination="auto", random_state=42, n_jobs=2)),
        ("One-Class SVM", OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)),
        ("Local Outlier Factor", LocalOutlierFactor(n_neighbors=35, novelty=True, contamination="auto", n_jobs=2)),
    ]
    for name, detector in anomaly_specs:
        print("Training:", name, " [normal-only]")
        detector.fit(anomaly_train)
        train_scores = detector.decision_function(anomaly_train)
        low, high = np.percentile(train_scores, [1, 99])
        wrapped = AnomalyProbabilityWrapper(detector, low, high)
        score = wrapped.predict_proba(X_test_scaled)[:, 1]
        pred = (score >= 0.5).astype(int)
        results[name] = {
            "model": wrapped, "metrics": metrics(y_test, pred, score),
            "group": "anomaly", "strategy": "normal-only",
        }

    print("Training: Autoencoder [normal-only]")
    auto = MLPRegressor(hidden_layer_sizes=(32, 16, 32), max_iter=35,
                        early_stopping=True, random_state=42)
    auto.fit(anomaly_train, anomaly_train)
    train_recon = auto.predict(anomaly_train)
    train_err = np.mean((anomaly_train - train_recon) ** 2, axis=1)
    low, high = np.percentile(train_err, [1, 99])
    auto_wrapped = ReconstructionProbabilityWrapper(auto, low, high)
    auto_score = auto_wrapped.predict_proba(X_test_scaled)[:, 1]
    auto_pred = (auto_score >= 0.5).astype(int)
    results["Autoencoder"] = {
        "model": auto_wrapped, "metrics": metrics(y_test, auto_pred, auto_score),
        "group": "anomaly", "strategy": "normal-only",
    }

    # Choose the best supervised model by PR-AUC. Anomaly detectors are
    # retained for comparison but are not automatically promoted to production.
    supervised_names = [n for n, r in results.items() if r["group"] != "anomaly"]
    best_name = max(supervised_names, key=lambda n: results[n]["metrics"]["pr_auc"])

    # Save summary
    summary_rows = []
    for name, r in results.items():
        m = r["metrics"]
        summary_rows.append({
            "Model": name, "Group": r["group"], "Imbalance Strategy": r["strategy"],
            "Precision": round(m["precision"], 4), "Recall": round(m["recall"], 4),
            "F1": round(m["f1"], 4), "FPR": round(m["fpr"], 4),
            "FNR": round(m["fnr"], 4), "PR-AUC": round(m["pr_auc"], 4),
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(ARTIFACT_DIR / "model_comparison.csv", index=False)

    # Confusion matrix + threshold curve for selected production candidate.
    best = results[best_name]
    best_score = best["model"].predict_proba(X_test_scaled)[:, 1]
    best_pred = (best_score >= .5).astype(int)
    cm = confusion_matrix(y_test, best_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    ax.set_xticks([0, 1], ["Normal", "Attack"])
    ax.set_yticks([0, 1], ["Normal", "Attack"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — " + best_name)
    fig.tight_layout()
    fig.savefig(ARTIFACT_DIR / "confusion_matrix.png", dpi=120)
    plt.close(fig)

    p, r, thresholds = precision_recall_curve(y_test, best_score)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(thresholds, p[:-1], label="Precision")
    ax.plot(thresholds, r[:-1], label="Recall")
    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Precision / Recall vs Threshold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ARTIFACT_DIR / "threshold_curve.png", dpi=120)
    plt.close(fig)

    # Feature importance where available.
    if hasattr(best["model"], "feature_importances_"):
        imp = pd.Series(best["model"].feature_importances_, index=feature_cols).nlargest(10)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(imp.index[::-1], imp.values[::-1])
        ax.set_title("Top Feature Importances — " + best_name)
        fig.tight_layout()
        fig.savefig(ARTIFACT_DIR / "feature_importance.png", dpi=120)
        plt.close(fig)

    # Deployment artifacts.
    slug_map = {name: slugify(name) for name in results}
    for name, r in results.items():
        joblib.dump(r["model"], ARTIFACT_DIR / f"model_{slug_map[name]}.joblib")

    joblib.dump(results[best_name]["model"], ARTIFACT_DIR / "model.joblib")
    joblib.dump(scaler, ARTIFACT_DIR / "scaler.joblib")
    joblib.dump(meta["encoders"], ARTIFACT_DIR / "encoders.joblib")

    all_meta = []
    for name, r in results.items():
        m = r["metrics"]
        all_meta.append({
            "name": name, "slug": slug_map[name], "group": r["group"],
            "strategy": r["strategy"], "precision": round(m["precision"], 4),
            "recall": round(m["recall"], 4), "f1": round(m["f1"], 4),
            "fpr": round(m["fpr"], 4), "fnr": round(m["fnr"], 4),
            "pr_auc": round(m["pr_auc"], 4),
        })

    deploy_meta = {
        "feature_cols": feature_cols, "skewed_cols": meta["skewed_cols"],
        "medians": meta["medians"], "modes": meta["modes"],
        "best_model_name": best_name, "default_threshold": 0.5,
        "all_models": all_meta,
        "dataset": "NSL-KDD",
        "supported_imbalance_strategies": [
            "none", "class_weight", "SMOTE", "sample_weight", "normal-only"
        ],
    }
    (ARTIFACT_DIR / "deploy_meta.json").write_text(json.dumps(deploy_meta, indent=2, default=str))

    (ARTIFACT_DIR / "sample_dataset.json").write_text(json.dumps(
        build_samples(test_raw), indent=2, default=lambda v: v.item() if hasattr(v, "item") else v
    ))

    # System A/B engineering scenario from the assignment.
    scenario = pd.DataFrame({
        "System": ["A", "B"], "Alerts/day": [50, 10],
        "Genuine attacks detected": [18, 5], "Total genuine attacks": [20, 20],
    })
    scenario["Recall"] = scenario["Genuine attacks detected"] / scenario["Total genuine attacks"]
    scenario["Missed attacks"] = scenario["Total genuine attacks"] - scenario["Genuine attacks detected"]
    scenario.to_csv(ARTIFACT_DIR / "system_ab_scenario.csv", index=False)

    print("\nSUCCESS")
    print("Best supervised model by PR-AUC:", best_name)
    print("Models trained:", len(results))
    print(summary.sort_values("PR-AUC", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
