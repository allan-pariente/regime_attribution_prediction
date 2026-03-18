"""
Etape 9 - Modele ML + SHAP
Pipeline : News-Driven Regime Attribution

Objectif : predire les changements de regime ET expliquer quelles features
           les causent via SHAP.

Input  : data/processed/dataset_final.csv   (toutes les features incl. embeddings)

Output : data/processed/model_results.csv   -> predictions + probabilites
         data/processed/shap_values.npy     -> valeurs SHAP brutes
         data/processed/shap_summary.csv    -> importance moyenne par feature
         data/models/random_forest.pkl      -> modele serialise
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.metrics import (classification_report, roc_auc_score,
                              precision_recall_curve, average_precision_score)
from sklearn.preprocessing import StandardScaler
import shap
import joblib
import logging
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Chemins ────────────────────────────────────────────────────────────────────
INPUT_PATH   = Path("data\\processed\\dataset_final.csv")
OUT_RESULTS  = Path("data\\processed\\model_results.csv")
OUT_SHAP     = Path("data\\processed\\shap_values.npy")
OUT_SHAP_SUM = Path("data\\processed\\shap_summary.csv")
OUT_MODEL    = Path("data\\models\\random_forest.pkl")
OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
OUT_RESULTS.parent.mkdir(parents=True, exist_ok=True)

# ── Parametres ─────────────────────────────────────────────────────────────────
N_ESTIMATORS  = 200
N_SPLITS      = 5
RANDOM_STATE  = 42
SHAP_SAMPLE   = 500


# ══════════════════════════════════════════════════════════════════════════════
# 1. Chargement et preparation des features
# ══════════════════════════════════════════════════════════════════════════════

MARKET_FEATURES = [
    "returns", "returns_pct", "volatility_20d", "vix",
    "rsi_14", "macd", "macd_signal", "macd_hist",
    "drawdown", "drawdown_pct",
]

SENTIMENT_FEATURES = [
    "sentiment_mean", "sentiment_std", "sentiment_min", "sentiment_max",
    "sentiment_anomaly", "news_count", "pct_positive", "pct_negative",
]

SHIFT_FEATURES = ["information_shift"]


def load_and_prepare(path: Path) -> tuple:
    log.info(f"Chargement de {path} ...")
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    log.info(f"  {len(emb_cols)} dimensions d'embeddings detectees")

    feature_cols = []
    for group, cols in [
        ("marche",     MARKET_FEATURES),
        ("sentiment",  SENTIMENT_FEATURES),
        ("shift",      SHIFT_FEATURES),
        ("embeddings", emb_cols),
    ]:
        present = [c for c in cols if c in df.columns]
        feature_cols.extend(present)
        log.info(f"  {group:12s} : {len(present)} features")

    log.info(f"  Total features : {len(feature_cols)}")

    df = df.dropna(subset=["y", "returns", "volatility_20d"])
    df[feature_cols] = df[feature_cols].fillna(0)

    log.info(f"  {len(df)} lignes | {df['y'].sum()} changements de regime (y=1)")
    return df, feature_cols


# ══════════════════════════════════════════════════════════════════════════════
# 2. Validation croisee temporelle
# ══════════════════════════════════════════════════════════════════════════════

def cross_validate_model(X: np.ndarray, y: np.ndarray) -> dict:
    log.info(f"Validation croisee temporelle ({N_SPLITS} folds) ...")

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    cv_results = cross_validate(
        model, X, y,
        cv=tscv,
        scoring=["roc_auc", "average_precision", "f1"],
        return_train_score=True,
    )

    metrics = {
        "roc_auc_mean":  cv_results["test_roc_auc"].mean(),
        "roc_auc_std":   cv_results["test_roc_auc"].std(),
        "ap_mean":       cv_results["test_average_precision"].mean(),
        "ap_std":        cv_results["test_average_precision"].std(),
        "f1_mean":       cv_results["test_f1"].mean(),
        "f1_std":        cv_results["test_f1"].std(),
    }

    log.info(f"  ROC-AUC : {metrics['roc_auc_mean']:.3f} +/- {metrics['roc_auc_std']:.3f}")
    log.info(f"  Avg Precision : {metrics['ap_mean']:.3f} +/- {metrics['ap_std']:.3f}")
    log.info(f"  F1  : {metrics['f1_mean']:.3f} +/- {metrics['f1_std']:.3f}")

    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# 3. Entrainement final
# ══════════════════════════════════════════════════════════════════════════════

def train_final_model(X: np.ndarray, y: np.ndarray) -> RandomForestClassifier:
    log.info("Entrainement du modele final sur tout le dataset ...")
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X, y)
    log.info("  Entrainement termine.")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# 4. Evaluation out-of-sample
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_last_fold(X: np.ndarray, y: np.ndarray, feature_cols: list) -> tuple:
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    log.info(f"Evaluation out-of-sample :")
    log.info(f"  Train : {len(X_train)} lignes | Test : {len(X_test)} lignes")
    log.info(f"  Changements dans le test : {y_test.sum()}")

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    log.info("\n" + classification_report(y_test, y_pred, target_names=["stable", "changement"]))

    if y_test.sum() > 0:
        auc = roc_auc_score(y_test, y_proba)
        ap  = average_precision_score(y_test, y_proba)
        log.info(f"  ROC-AUC out-of-sample : {auc:.3f}")
        log.info(f"  Avg Precision         : {ap:.3f}")

    return model, np.arange(split, len(X)), y_proba


# ══════════════════════════════════════════════════════════════════════════════
# 5. Analyse SHAP
# ══════════════════════════════════════════════════════════════════════════════

def compute_shap(model: RandomForestClassifier, X: np.ndarray, feature_cols: list) -> tuple:
    log.info(f"Calcul SHAP sur {min(SHAP_SAMPLE, len(X))} echantillons ...")

    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(X), size=min(SHAP_SAMPLE, len(X)), replace=False)
    idx.sort()
    X_sample = X[idx]

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Gestion des differents formats possibles selon la version de SHAP :
    # - liste [class0, class1]          -> prendre l'index 1
    # - array 3D (samples, feats, classes) -> prendre [:, :, 1]
    # - array 2D (samples, feats)       -> utiliser tel quel
    if isinstance(shap_values, list):
        sv = shap_values[1]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        sv = shap_values[:, :, 1]
    else:
        sv = shap_values

    # Garantir que mean_abs est 1D
    mean_abs = np.abs(sv).mean(axis=0).flatten()

    shap_df = pd.DataFrame({
        "feature":       feature_cols,
        "shap_mean_abs": mean_abs,
    }).sort_values("shap_mean_abs", ascending=False).reset_index(drop=True)

    log.info("Top 20 features par importance SHAP :")
    for _, row in shap_df.head(20).iterrows():
        bar = "#" * int(row["shap_mean_abs"] / shap_df["shap_mean_abs"].max() * 30)
        log.info(f"  {row['feature']:35s} {bar} {row['shap_mean_abs']:.4f}")

    return sv, shap_df


# ══════════════════════════════════════════════════════════════════════════════
# 6. Attribution par groupe de features
# ══════════════════════════════════════════════════════════════════════════════

def group_shap_importance(shap_df: pd.DataFrame, emb_cols: list) -> pd.DataFrame:
    def get_group(feat):
        if feat in MARKET_FEATURES:    return "marche"
        if feat in SENTIMENT_FEATURES: return "sentiment"
        if feat in SHIFT_FEATURES:     return "shift"
        if feat in emb_cols:           return "embeddings"
        return "autre"

    shap_df = shap_df.copy()
    shap_df["group"] = shap_df["feature"].apply(get_group)

    group_df = (
        shap_df.groupby("group")["shap_mean_abs"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )
    group_df.columns = ["group", "shap_total"]
    total = group_df["shap_total"].sum()
    group_df["shap_pct"] = (group_df["shap_total"] / total * 100).round(1)

    log.info("\nImportance SHAP par groupe :")
    for _, row in group_df.iterrows():
        log.info(f"  {row['group']:15s} : {row['shap_pct']:.1f}%")

    return group_df


# ══════════════════════════════════════════════════════════════════════════════
# 7. Sauvegardes
# ══════════════════════════════════════════════════════════════════════════════

def save_results(df: pd.DataFrame, test_indices: np.ndarray,
                 y_proba: np.ndarray, cv_metrics: dict) -> None:
    results_df = df[["date", "y", "regime"]].copy()
    results_df["y_proba"] = np.nan
    results_df.loc[test_indices, "y_proba"] = y_proba
    results_df["regime_change_predicted"] = (
        results_df["y_proba"].fillna(0) > 0.5
    ).astype(int)
    results_df.to_csv(OUT_RESULTS, index=False)
    log.info(f"OK Resultats sauvegardes -> {OUT_RESULTS}")


# ══════════════════════════════════════════════════════════════════════════════
# 8. Point d'entree
# ══════════════════════════════════════════════════════════════════════════════

def main():
    df, feature_cols = load_and_prepare(INPUT_PATH)
    emb_cols = [c for c in feature_cols if c.startswith("emb_")]

    X = df[feature_cols].values
    y = df["y"].values

    log.info(f"\nShape X : {X.shape} | positifs : {y.sum()} ({y.mean():.2%})")

    log.info("")
    cv_metrics = cross_validate_model(X, y)

    log.info("")
    oof_model, test_indices, y_proba = evaluate_last_fold(X, y, feature_cols)

    log.info("")
    final_model = train_final_model(X, y)

    log.info("")
    shap_values, shap_df = compute_shap(final_model, X, feature_cols)

    log.info("")
    group_df = group_shap_importance(shap_df, emb_cols)

    log.info("")
    save_results(df, test_indices, y_proba, cv_metrics)

    np.save(OUT_SHAP, shap_values)
    log.info(f"OK SHAP values sauvegardes -> {OUT_SHAP}")

    shap_df.to_csv(OUT_SHAP_SUM, index=False)
    log.info(f"OK SHAP summary sauvegarde -> {OUT_SHAP_SUM}")

    joblib.dump(final_model, OUT_MODEL)
    log.info(f"OK Modele sauvegarde -> {OUT_MODEL}")

    log.info("")
    log.info("=" * 55)
    log.info("RESUME FINAL")
    log.info("=" * 55)
    log.info(f"ROC-AUC CV    : {cv_metrics['roc_auc_mean']:.3f} +/- {cv_metrics['roc_auc_std']:.3f}")
    log.info(f"Avg Prec CV   : {cv_metrics['ap_mean']:.3f} +/- {cv_metrics['ap_std']:.3f}")
    log.info(f"F1 CV         : {cv_metrics['f1_mean']:.3f} +/- {cv_metrics['f1_std']:.3f}")
    log.info(f"Top feature   : {shap_df.iloc[0]['feature']} (SHAP={shap_df.iloc[0]['shap_mean_abs']:.4f})")
    log.info("=" * 55)


if __name__ == "__main__":
    main()