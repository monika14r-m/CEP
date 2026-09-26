"""
End-to-end training pipeline for a Machine Learning-based Intrusion Detection
System, built on NSL-KDD.

Combines: data loading -> EDA -> preprocessing -> imbalance handling
(class-weighting + SMOTE comparison) -> model training (Decision Tree,
Random Forest, Random Forest+SMOTE) -> evaluation -> threshold analysis ->
engineering decision scenario -> zero-day/unseen-attack analysis -> artifact
export for deployment.

Run:
    pip install -r requirements.txt
    python train_pipeline.py

Produces an `artifacts/` folder consumed by `app.py` for serving. Every
trained model is saved individually (model_<slug>.joblib) alongside the
overall best one (model.joblib), so the API can expose real side-by-side
model comparison rather than only the single best model's opinion.
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (auc, confusion_matrix, f1_score,
                              precision_recall_curve, precision_score,
                              recall_score)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

np.random.seed(42)
ARTIFACT_DIR = Path(__file__).parent / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
COLS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
    "wrong_fragment", "urgent",
    "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login",
    "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "label", "difficulty",
]

TRAIN_URL = "https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/KDDTrain%2B.txt"
TEST_URL = "https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/KDDTest%2B.txt"

# Raw (pre-encoding, pre-scaling) feature columns — exactly the shape
# app.py's ConnectionRecord expects, so real rows sampled from the raw test
# set can be sent straight to /predict_all with no extra mapping.
RAW_FEATURE_COLS = [c for c in COLS if c not in ("label", "difficulty")]


def load_data():
    train_df = pd.read_csv(TRAIN_URL, names=COLS)
    test_df = pd.read_csv(TEST_URL, names=COLS)

    assert train_df.shape[1] == len(COLS), "Column count mismatch — check the source file"
    assert test_df.shape[1] == len(COLS), "Column count mismatch — check the source file"

    print("Train shape:", train_df.shape, " Test shape:", test_df.shape)
    print("Missing values (train):", train_df.isna().sum().sum())
    print("Missing values (test): ", test_df.isna().sum().sum())

    train_types = set(train_df["label"].unique())
    test_types = set(test_df["label"].unique())
    unseen = test_types - train_types
    print(f"\nAttack types in test but unseen during training ({len(unseen)}):")
    print(sorted(unseen))
    print(f"\nOverall attack rate — train: {(train_df['label'] != 'normal').mean():.3f}"
          f"   test: {(test_df['label'] != 'normal').mean():.3f}")
    return train_df, test_df


# ============================================================================
# 2. EDA
# ============================================================================
def plot_class_distribution(train_df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    binary_counts = (train_df["label"] != "normal").map({True: "Attack", False: "Normal"}).value_counts()
    axes[0].bar(binary_counts.index, binary_counts.values, color=["#4c72b0", "#c44e52"])
    axes[0].set_title("Normal vs Attack — Training Set")
    axes[0].set_ylabel("Number of connections")
    for i, v in enumerate(binary_counts.values):
        axes[0].text(i, v + 500, str(v), ha="center")

    attack_only = train_df[train_df["label"] != "normal"]["label"].value_counts().head(10)
    axes[1].barh(attack_only.index[::-1], attack_only.values[::-1], color="#dd8452")
    axes[1].set_title("Top 10 Attack Types — Training Set")
    axes[1].set_xlabel("Number of connections")

    plt.tight_layout()
    plt.savefig(ARTIFACT_DIR / "class_distribution.png", dpi=120)
    plt.close(fig)
    print("\nAttack type distribution shows this isn't just 'normal vs attack' —")
    print("some types (e.g. neptune, smurf) are common; others are extremely rare,")
    print("a secondary imbalance problem inside the attack class itself.")


# ============================================================================
# 3. PREPROCESSING
# ============================================================================
def preprocess(train_df: pd.DataFrame, test_df: pd.DataFrame):
    train_df = train_df.drop(columns=["difficulty"]).copy()
    test_df = test_df.drop(columns=["difficulty"]).copy()

    print("\nDuplicate rows in train:", train_df.duplicated().sum())
    train_df = train_df.drop_duplicates()

    numeric_cols = [c for c in train_df.select_dtypes(include=["int64", "float64"]).columns]
    for c in numeric_cols:
        train_df[c] = train_df[c].replace(-1, np.nan)
        test_df[c] = test_df[c].replace(-1, np.nan)

    medians = {}
    for c in numeric_cols:
        m = float(train_df[c].median())
        medians[c] = m
        train_df[c] = train_df[c].fillna(m)
        test_df[c] = test_df[c].fillna(m)

    cat_cols = ["protocol_type", "service", "flag"]
    modes = {}
    for c in cat_cols:
        mo = str(train_df[c].mode()[0])
        modes[c] = mo
        train_df[c] = train_df[c].fillna(mo)
        test_df[c] = test_df[c].fillna(mo)

    train_df["binary_label"] = (train_df["label"] != "normal").astype(int)
    test_df["binary_label"] = (test_df["label"] != "normal").astype(int)

    print("\nClass balance (train):")
    print(train_df["binary_label"].value_counts(normalize=True).rename({0: "normal", 1: "attack"}))

    # Outlier capping — normal class only. Attack-side extremes are signal,
    # not noise, so they are left untouched.
    skewed = [c for c in ["duration", "src_bytes", "dst_bytes", "count", "srv_count"]
              if c in train_df.columns]

    # Cast to float first — these are int64 columns, and clipping with float
    # quantile bounds raises a dtype error on newer pandas otherwise.
    for c in skewed:
        train_df[c] = train_df[c].astype(float)
        test_df[c] = test_df[c].astype(float)

    normal_mask = train_df["binary_label"] == 0
    for c in skewed:
        lo, hi = train_df.loc[normal_mask, c].quantile([0.01, 0.99])
        train_df.loc[normal_mask, c] = train_df.loc[normal_mask, c].clip(lo, hi)

    for c in skewed:
        train_df[c] = np.log1p(train_df[c])
        test_df[c] = np.log1p(test_df[c])

    encoders = {}
    for c in cat_cols:
        le = LabelEncoder()
        le.fit(pd.concat([train_df[c], test_df[c]], axis=0))
        train_df[c] = le.transform(train_df[c])
        test_df[c] = le.transform(test_df[c])
        encoders[c] = le

    feature_cols = [c for c in train_df.columns if c not in ("label", "binary_label")]
    corr_matrix = train_df[feature_cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    redundant = [c for c in upper.columns if any(upper[c] > 0.95)]
    print(f"\nDropping {len(redundant)} highly correlated redundant features: {redundant}")
    feature_cols = [c for c in feature_cols if c not in redundant]

    meta = {
        "medians": medians,
        "modes": modes,
        "skewed_cols": skewed,
        "encoders": encoders,
        "feature_cols": feature_cols,
        "redundant_dropped": redundant,
    }
    return train_df, test_df, meta


# ============================================================================
# 4. EVALUATION HELPER
# ============================================================================
def evaluate(name, y_test, y_pred, y_proba):
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    prec_curve, rec_curve, thresholds = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(rec_curve, prec_curve)

    print(f"\n=== {name} ===")
    print(cm)
    print(f"Precision {precision:.3f}  Recall {recall:.3f}  F1 {f1:.3f}  "
          f"FPR {fpr:.4f}  FNR {fnr:.4f}  PR-AUC {pr_auc:.3f}")

    return dict(cm=cm, precision=precision, recall=recall, f1=f1, fpr=fpr, fnr=fnr,
                pr_auc=pr_auc, prec_curve=prec_curve, rec_curve=rec_curve, thresholds=thresholds)


def slugify(name: str) -> str:
    return name.lower().replace(" ", "_").replace("(", "").replace(")", "")


def _to_native(v):
    return v.item() if hasattr(v, "item") else v


def build_sample_dataset(test_df_raw: pd.DataFrame, per_label: int = 3, max_total: int = 30):
    """A small, class-diverse sample of REAL, untransformed test-set
    connections (several normal + several distinct attack types), in the
    exact raw shape the API expects. This is what powers the dashboard's
    'live dataset' option once a backend is connected — real traffic, not
    hand-set demo numbers."""
    rows = []
    for label, grp in test_df_raw.groupby("label"):
        take = grp.sample(min(len(grp), per_label), random_state=42)
        for _, r in take.iterrows():
            rec = {c: _to_native(r[c]) for c in RAW_FEATURE_COLS}
            rec["attack_type"] = label
            rec["true_label"] = "normal" if label == "normal" else "attack"
            rows.append(rec)
        if len(rows) >= max_total:
            break
    return rows[:max_total]


# ============================================================================
# MAIN
# ============================================================================
def main():
    train_df, test_df = load_data()
    test_df_raw = test_df.copy()  # untransformed — used later for real sample records
    plot_class_distribution(train_df)

    train_df, test_df, meta = preprocess(train_df, test_df)
    feature_cols = meta["feature_cols"]

    X_train, y_train = train_df[feature_cols], train_df["binary_label"]
    X_test, y_test = test_df[feature_cols], test_df["binary_label"]

    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"\nFinal feature count: {len(feature_cols)}")
    print(f"Train shape: {X_train_scaled.shape}, Test shape: {X_test_scaled.shape}")

    # ---- Imbalance handling: class-weighting (always) vs SMOTE (if available) ----
    smote_available = False
    try:
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
        smote_available = True
    except ImportError:
        print("\n`imbalanced-learn` not installed — skipping SMOTE comparison "
              "(run `pip install imbalanced-learn` to enable it).")

    # ---- Model training ----
    # "Ours" = the models this project selects/recommends. "Baseline" = the
    # other model families Task 5 asks us to evaluate for comparison, kept
    # deliberately fast approximations (LinearSVC, not kernel SVC; k=5 NN)
    # in line with constraint #5 (limited compute for real-time detection).
    OURS = {
        "Decision Tree": DecisionTreeClassifier(class_weight="balanced", max_depth=10, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", max_depth=15, n_jobs=-1, random_state=42
        ),
    }
    BASELINES = {
        "SVM (linear)": CalibratedClassifierCV(
            LinearSVC(class_weight="balanced", max_iter=2000, random_state=42), cv=3
        ),
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    }
    models = {**OURS, **BASELINES}
    if smote_available:
        OURS["Random Forest (SMOTE)"] = None  # placeholder so the group lookup below finds it
        models["Random Forest (SMOTE)"] = RandomForestClassifier(
            n_estimators=200, max_depth=15, n_jobs=-1, random_state=42
        )
    model_group = {name: "ours" for name in OURS}
    model_group.update({name: "baseline" for name in BASELINES})

    results = {}
    for name, model in models.items():
        if name.endswith("(SMOTE)"):
            model.fit(X_train_smote, y_train_smote)
        else:
            model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        results[name] = {"model": model, "y_pred": y_pred, "y_proba": y_proba}

    eval_results = {
        name: evaluate(name, y_test, r["y_pred"], r["y_proba"]) for name, r in results.items()
    }

    best_name = max(eval_results, key=lambda n: eval_results[n]["pr_auc"])
    print(f"\nBest model by PR-AUC: {best_name}")
    best_model = results[best_name]["model"]
    y_pred_best = results[best_name]["y_pred"]
    y_scores = results[best_name]["y_proba"]

    # ---- Confusion matrix plot for the chosen model ----
    cm = eval_results[best_name]["cm"]
    plt.figure(figsize=(4, 4))
    plt.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.xticks([0, 1], ["Normal", "Attack"])
    plt.yticks([0, 1], ["Normal", "Attack"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix — {best_name}")
    plt.tight_layout()
    plt.savefig(ARTIFACT_DIR / "confusion_matrix.png", dpi=120)
    plt.close()

    # ---- Threshold analysis (Task 7) ----
    prec_curve = eval_results[best_name]["prec_curve"]
    rec_curve = eval_results[best_name]["rec_curve"]
    thresholds = eval_results[best_name]["thresholds"]

    plt.figure(figsize=(6, 4))
    plt.plot(thresholds, prec_curve[:-1], label="Precision")
    plt.plot(thresholds, rec_curve[:-1], label="Recall")
    plt.xlabel("Decision Threshold")
    plt.ylabel("Score")
    plt.title(f"Precision & Recall vs Threshold — {best_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ARTIFACT_DIR / "threshold_curve.png", dpi=120)
    plt.close()

    print("\nThreshold sweep:")
    for t in [0.3, 0.5, 0.7, 0.9]:
        pred_t = (y_scores >= t).astype(int)
        print(f"Threshold={t:.1f}  Precision={precision_score(y_test, pred_t):.3f}  "
              f"Recall={recall_score(y_test, pred_t):.3f}  Alerts={pred_t.sum()}")

    # ---- Engineering decision scenario: System A vs System B (Task 8) ----
    scenario = pd.DataFrame({
        "System": ["A", "B"],
        "Alerts/day": [50, 10],
        "Genuine attacks detected": [18, 5],
        "Total genuine attacks": [20, 20],
    })
    scenario["Recall"] = scenario["Genuine attacks detected"] / scenario["Total genuine attacks"]
    scenario["Missed attacks"] = scenario["Total genuine attacks"] - scenario["Genuine attacks detected"]
    print("\nSystem A vs System B:\n", scenario.to_string(index=False))
    scenario.to_csv(ARTIFACT_DIR / "system_ab_scenario.csv", index=False)

    # ---- Zero-day / unseen attack analysis (Task 9) ----
    train_attack_types = set(train_df["label"].unique()) - {"normal"}
    test_attack_types = set(test_df["label"].unique()) - {"normal"}
    unseen_types = test_attack_types - train_attack_types
    print(f"\nUnseen attack types in test set ({len(unseen_types)}):", sorted(unseen_types))

    unseen_mask = test_df["label"].isin(unseen_types)
    if unseen_mask.sum() > 0:
        unseen_recall = recall_score(y_test[unseen_mask], y_pred_best[unseen_mask])
        known_mask = (~unseen_mask) & (y_test == 1)
        known_recall = (
            recall_score(y_test[known_mask], y_pred_best[known_mask]) if known_mask.sum() else float("nan")
        )
        print(f"Recall on unseen attack types: {unseen_recall:.3f}")
        print(f"Recall on known attack types:  {known_recall:.3f}")

        breakdown = []
        for atk in sorted(unseen_types):
            idx = (test_df["label"] == atk).values
            n = idx.sum()
            if n == 0:
                continue
            detected = y_pred_best[idx].sum()
            breakdown.append({
                "attack_type": atk, "samples": int(n), "detected": int(detected),
                "missed": int(n - detected), "detection_rate": round(float(detected / n), 3),
            })
        breakdown_df = pd.DataFrame(breakdown).sort_values("detection_rate")
        print("\nPer-attack-type breakdown (zero-day types):\n", breakdown_df.to_string(index=False))
        breakdown_df.to_csv(ARTIFACT_DIR / "zero_day_breakdown.csv", index=False)

    # ---- Summary table across all models ----
    summary = pd.DataFrame({
        "Model": list(eval_results.keys()),
        "Precision": [eval_results[n]["precision"] for n in eval_results],
        "Recall": [eval_results[n]["recall"] for n in eval_results],
        "F1": [eval_results[n]["f1"] for n in eval_results],
        "FPR": [eval_results[n]["fpr"] for n in eval_results],
        "FNR": [eval_results[n]["fnr"] for n in eval_results],
        "PR-AUC": [eval_results[n]["pr_auc"] for n in eval_results],
    }).round(3)
    print("\nModel comparison:\n", summary.to_string(index=False))
    summary.to_csv(ARTIFACT_DIR / "model_comparison.csv", index=False)

    # ---- Feature importance plot ----
    if hasattr(best_model, "feature_importances_"):
        importances = pd.Series(best_model.feature_importances_, index=feature_cols) \
            .sort_values(ascending=False).head(10)
        plt.figure(figsize=(6, 4))
        plt.barh(importances.index[::-1], importances.values[::-1])
        plt.title(f"Top 10 Feature Importances — {best_name}")
        plt.tight_layout()
        plt.savefig(ARTIFACT_DIR / "feature_importance.png", dpi=120)
        plt.close()

    # ---- Export artifacts for deployment ----
    # Save every trained model individually (not just the best one) so the
    # API can offer real side-by-side model comparison, not just one opinion.
    slug_map = {name: slugify(name) for name in results}
    for name, r in results.items():
        joblib.dump(r["model"], ARTIFACT_DIR / f"model_{slug_map[name]}.joblib")

    joblib.dump(best_model, ARTIFACT_DIR / "model.joblib")
    joblib.dump(scaler, ARTIFACT_DIR / "scaler.joblib")
    joblib.dump(meta["encoders"], ARTIFACT_DIR / "encoders.joblib")

    all_models_meta = [
        {
            "name": name,
            "slug": slug_map[name],
            "group": model_group.get(name, "baseline"),
            "precision": round(float(eval_results[name]["precision"]), 4),
            "recall": round(float(eval_results[name]["recall"]), 4),
            "f1": round(float(eval_results[name]["f1"]), 4),
            "fpr": round(float(eval_results[name]["fpr"]), 4),
            "fnr": round(float(eval_results[name]["fnr"]), 4),
            "pr_auc": round(float(eval_results[name]["pr_auc"]), 4),
        }
        for name in eval_results
    ]

    deploy_meta = {
        "feature_cols": feature_cols,
        "skewed_cols": meta["skewed_cols"],
        "medians": meta["medians"],
        "modes": meta["modes"],
        "best_model_name": best_name,
        "default_threshold": 0.5,
        "all_models": all_models_meta,
    }
    with open(ARTIFACT_DIR / "deploy_meta.json", "w") as f:
        json.dump(deploy_meta, f, indent=2)

    # ---- Real sample dataset for the dashboard's live mode ----
    sample_dataset = build_sample_dataset(test_df_raw)
    with open(ARTIFACT_DIR / "sample_dataset.json", "w") as f:
        json.dump(sample_dataset, f, indent=2, default=_to_native)
    print(f"\nSaved {len(sample_dataset)} real sample connections to sample_dataset.json")

    print(f"\nArtifacts saved to {ARTIFACT_DIR.resolve()}/")
    print("Files:", sorted(p.name for p in ARTIFACT_DIR.iterdir()))


if __name__ == "__main__":
    main()
