"""
Etape 10 - Module d'attribution finale (SHAP locaux corriges)
Pipeline : News-Driven Regime Attribution

Pour chaque changement de regime detecte, produit un rapport expliquant :
- Les news les plus significatives dans la fenetre
- Les topics dominants
- Le score d'information shift
- Les features SHAP LOCAUX pour ce jour specifique

Inputs :
  data/processed/news_with_embeddings.csv  -> news + sentiment + topic
  data/processed/dataset_final.csv         -> dataset fusionne
  data/processed/shap_values.npy           -> SHAP locaux (n_samples, n_features)
  data/processed/regime_shift.csv          -> information shift par regime
  data/models/random_forest.pkl            -> modele entraine

Output :
  data/processed/attribution_report.json  -> rapports complets (JSON)
  data/processed/attribution_report.txt   -> rapports lisibles (texte)
"""

import pandas as pd
import numpy as np
import json
import joblib
import shap
from pathlib import Path
from datetime import timedelta
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
NEWS_PATH    = Path("data\\processed\\news_with_embeddings.csv")
DATASET_PATH = Path("data\\processed\\dataset_final.csv")
SHAP_PATH    = Path("data\\processed\\shap_values.npy")
SHIFT_PATH   = Path("data\\processed\\regime_shift.csv")
MODEL_PATH   = Path("data\\models\\random_forest.pkl")
OUT_JSON     = Path("data\\processed\\attribution_report.json")
OUT_TXT      = Path("data\\processed\\attribution_report.txt")
OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

# ── Parametres ─────────────────────────────────────────────────────────────────
WINDOW_BEFORE = 5
WINDOW_AFTER  = 1
TOP_NEWS      = 5
TOP_FEATURES  = 10

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


# ══════════════════════════════════════════════════════════════════════════════
# 1. Chargement
# ══════════════════════════════════════════════════════════════════════════════

def load_all():
    log.info("Chargement des donnees ...")

    news = pd.read_csv(NEWS_PATH, parse_dates=["datetime", "date", "regime_change_date"])
    log.info(f"  {len(news)} articles charges")

    dataset = pd.read_csv(DATASET_PATH, parse_dates=["date"])
    dataset = dataset.sort_values("date").reset_index(drop=True)
    dataset = dataset.dropna(subset=["returns", "volatility_20d"])
    dataset[dataset.select_dtypes(include=[np.number]).columns] = \
        dataset.select_dtypes(include=[np.number]).fillna(0)

    model = joblib.load(MODEL_PATH)
    log.info(f"  Modele charge : {MODEL_PATH}")

    # Reconstruire la liste de features dans le meme ordre que lors de l'entrainement
    emb_cols = [c for c in dataset.columns if c.startswith("emb_")]
    feature_cols = []
    for cols in [MARKET_FEATURES, SENTIMENT_FEATURES, SHIFT_FEATURES, emb_cols]:
        feature_cols.extend([c for c in cols if c in dataset.columns])

    shift_df = pd.read_csv(SHIFT_PATH, parse_dates=["regime_change_date"])

    # ── Calcul des SHAP locaux sur tout le dataset ────────────────────────────
    # On recalcule plutot que de charger shap_values.npy car ce dernier
    # a ete calcule sur un echantillon de 500 lignes (indices non alignes).
    log.info("Calcul des SHAP locaux sur tout le dataset (peut prendre 1-2 min) ...")
    X = dataset[feature_cols].values
    explainer = shap.TreeExplainer(model)
    shap_raw = explainer.shap_values(X)

    # Normalisation du format selon la version de SHAP
    if isinstance(shap_raw, list):
        sv = shap_raw[1]          # classe 1 = changement de regime
    elif isinstance(shap_raw, np.ndarray) and shap_raw.ndim == 3:
        sv = shap_raw[:, :, 1]
    else:
        sv = shap_raw

    # sv shape : (n_lignes_dataset, n_features)
    log.info(f"  SHAP locaux calcules : shape={sv.shape}")

    # Construire un DataFrame de SHAP locaux aligne sur dataset
    shap_df = pd.DataFrame(sv, columns=feature_cols, index=dataset.index)

    return news, dataset, shap_df, shift_df, model, feature_cols, emb_cols


# ══════════════════════════════════════════════════════════════════════════════
# 2. SHAP locaux pour un jour donne
# ══════════════════════════════════════════════════════════════════════════════

def get_top_shap_features_local(
    change_date: pd.Timestamp,
    dataset: pd.DataFrame,
    shap_df: pd.DataFrame,
    feature_cols: list,
    emb_cols: list,
    top_n: int = TOP_FEATURES,
) -> list:
    """
    Retourne les TOP_FEATURES features (hors embeddings) avec leurs valeurs
    SHAP LOCALES pour le jour du changement de regime.
    Les valeurs SHAP locales mesurent la contribution de chaque feature
    a la prediction pour CE jour specifique (pas une moyenne globale).
    """
    row_idx = dataset[dataset["date"] == change_date].index
    if len(row_idx) == 0:
        # Fallback : jour le plus proche
        diffs = (dataset["date"] - change_date).abs()
        row_idx = [diffs.idxmin()]

    idx = row_idx[0]

    # Filtrer les features interpretables (pas les embeddings)
    interpretable_cols = [c for c in feature_cols if c not in emb_cols]

    # SHAP locaux pour cette ligne, sur les features interpretables seulement
    local_shap = shap_df.loc[idx, interpretable_cols].abs()
    top_feats = local_shap.nlargest(top_n).index.tolist()

    result = []
    for feat in top_feats:
        feat_val = dataset.loc[idx, feat] if feat in dataset.columns else None
        shap_val = float(shap_df.loc[idx, feat])
        result.append({
            "feature":        feat,
            "value":          round(float(feat_val), 4) if feat_val is not None and pd.notna(feat_val) else None,
            "shap_local":     round(shap_val, 6),
            "shap_local_abs": round(abs(shap_val), 6),
            "direction":      "hausse_regime" if shap_val > 0 else "baisse_regime",
        })

    return result


# ══════════════════════════════════════════════════════════════════════════════
# 3. Selection des news les plus significatives
# ══════════════════════════════════════════════════════════════════════════════

def get_top_news(change_date, news, window_before=WINDOW_BEFORE,
                 window_after=WINDOW_AFTER, top_n=TOP_NEWS):
    date_min = (change_date - timedelta(days=window_before)).date()
    date_max = (change_date + timedelta(days=window_after)).date()

    dates = news["date"].dt.date if hasattr(news["date"].dt, "date") else news["date"]
    mask = (dates >= date_min) & (dates <= date_max)
    window_news = news[mask].copy()

    if window_news.empty:
        return []

    window_news = window_news.sort_values("sentiment_abs", ascending=False)

    result = []
    for _, row in window_news.head(top_n).iterrows():
        result.append({
            "date":            str(row["date"].date()) if hasattr(row["date"], "date") else str(row["date"]),
            "headline":        str(row["headline"]),
            "sentiment_label": str(row.get("sentiment_label", "unknown")),
            "sentiment_score": round(float(row["sentiment"]), 4),
            "topic":           int(row["topic"]) if pd.notna(row.get("topic")) else None,
            "source":          str(row.get("source", "")),
        })
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 4. Distribution des topics
# ══════════════════════════════════════════════════════════════════════════════

def get_topic_distribution(change_date, news, window_before=WINDOW_BEFORE,
                           window_after=WINDOW_AFTER):
    date_min = (change_date - timedelta(days=window_before)).date()
    date_max = (change_date + timedelta(days=window_after)).date()

    dates = news["date"].dt.date if hasattr(news["date"].dt, "date") else news["date"]
    mask = (dates >= date_min) & (dates <= date_max)
    window_news = news[mask]

    if window_news.empty or "topic" not in window_news.columns:
        return {}

    topic_counts = window_news["topic"].value_counts().head(3)
    total = len(window_news)
    return {
        f"topic_{int(k)}": {"count": int(v), "pct": round(v / total * 100, 1)}
        for k, v in topic_counts.items()
    }


# ══════════════════════════════════════════════════════════════════════════════
# 5. Contexte de marche
# ══════════════════════════════════════════════════════════════════════════════

def get_market_context(change_date, dataset):
    row = dataset[dataset["date"] == change_date]
    if row.empty:
        diffs = (dataset["date"] - change_date).abs()
        row = dataset.loc[[diffs.idxmin()]]

    r = row.iloc[0]

    def safe(col):
        v = r.get(col)
        return round(float(v), 4) if v is not None and pd.notna(v) else None

    return {
        "regime":         str(r.get("regime", "")),
        "returns":        safe("returns"),
        "returns_pct":    safe("returns_pct"),
        "volatility_20d": safe("volatility_20d"),
        "vix":            safe("vix"),
        "rsi_14":         safe("rsi_14"),
        "drawdown_pct":   safe("drawdown_pct"),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 6. Construction du rapport pour un changement de regime
# ══════════════════════════════════════════════════════════════════════════════

def build_attribution(change_date, news, dataset, shap_df, shift_df,
                      model, feature_cols, emb_cols):
    shift_row = shift_df[shift_df["regime_change_date"] == change_date]
    shift_score = (
        round(float(shift_row["information_shift"].values[0]), 4)
        if not shift_row.empty and pd.notna(shift_row["information_shift"].values[0])
        else None
    )

    date_min = (change_date - timedelta(days=WINDOW_BEFORE)).date()
    date_max = (change_date + timedelta(days=WINDOW_AFTER)).date()
    dates = news["date"].dt.date if hasattr(news["date"].dt, "date") else news["date"]
    window_news = news[(dates >= date_min) & (dates <= date_max)]
    sentiment_mean = round(float(window_news["sentiment"].mean()), 4) if not window_news.empty else None
    news_count = len(window_news)

    return {
        "regime_change_date": str(change_date.date()),
        "market_context":     get_market_context(change_date, dataset),
        "news_window": {
            "days_before":      WINDOW_BEFORE,
            "days_after":       WINDOW_AFTER,
            "total_articles":   news_count,
            "sentiment_mean":   sentiment_mean,
            "information_shift": shift_score,
        },
        "top_news":           get_top_news(change_date, news),
        "topic_distribution": get_topic_distribution(change_date, news),
        "top_shap_features":  get_top_shap_features_local(
            change_date, dataset, shap_df, feature_cols, emb_cols
        ),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 7. Rapport texte lisible
# ══════════════════════════════════════════════════════════════════════════════

def format_report_txt(attribution):
    lines = []
    sep = "=" * 65

    cd = attribution["regime_change_date"]
    mc = attribution["market_context"]
    nw = attribution["news_window"]

    lines += [
        sep,
        f"CHANGEMENT DE REGIME -- {cd}",
        sep,
        f"Regime detecte     : {mc.get('regime', 'N/A')}",
        f"Rendement jour J   : {mc.get('returns_pct', 'N/A')}%",
        f"Volatilite 20j     : {mc.get('volatility_20d', 'N/A')}",
        f"VIX                : {mc.get('vix', 'N/A')}",
        f"RSI 14j            : {mc.get('rsi_14', 'N/A')}",
        f"Drawdown           : {mc.get('drawdown_pct', 'N/A')}%",
        "",
        f"Articles analyses  : {nw['total_articles']} "
        f"(J-{nw['days_before']} a J+{nw['days_after']})",
        f"Sentiment moyen    : {nw['sentiment_mean']}",
        f"Information shift  : {nw['information_shift']}",
    ]

    lines += ["", "-" * 65, "TOP NEWS LES PLUS SIGNIFICATIVES", "-" * 65]
    for i, n in enumerate(attribution["top_news"], 1):
        label = n["sentiment_label"].upper()
        score = n["sentiment_score"]
        lines += [
            f"{i}. [{n['date']}] [{label} {score:+.2f}] Topic {n['topic']}",
            f"   {n['headline']}",
            f"   Source : {n['source']}",
            "",
        ]

    lines += ["-" * 65, "TOPICS DOMINANTS", "-" * 65]
    for topic, info in attribution["topic_distribution"].items():
        bar = "#" * int(info["pct"] / 100 * 30)
        lines.append(f"  {topic:10s} : {bar} {info['pct']}% ({info['count']} articles)")

    lines += ["", "-" * 65,
              "FEATURES LES PLUS IMPORTANTES (SHAP LOCAUX -- specifiques a ce jour)",
              "-" * 65]
    for f in attribution["top_shap_features"]:
        direction = "+" if f["shap_local"] > 0 else "-"
        lines.append(
            f"  {f['feature']:30s} valeur={f['value']:>10}  "
            f"SHAP={f['shap_local']:+.6f}  [{direction}]"
        )

    lines += ["", ""]
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# 8. Point d'entree
# ══════════════════════════════════════════════════════════════════════════════

def main():
    news, dataset, shap_df, shift_df, model, feature_cols, emb_cols = load_all()

    change_dates = sorted(dataset[dataset["y"] == 1]["date"].dropna().unique())
    log.info(f"\n{len(change_dates)} changements de regime a analyser")

    all_attributions = []
    txt_sections = []

    for cd in change_dates:
        cd_ts = pd.Timestamp(cd)
        log.info(f"  Attribution : {cd_ts.date()} ...")

        attr = build_attribution(
            cd_ts, news, dataset, shap_df, shift_df,
            model, feature_cols, emb_cols
        )
        all_attributions.append(attr)
        txt_sections.append(format_report_txt(attr))

    # Sauvegarde JSON
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_attributions, f, ensure_ascii=False, indent=2)
    log.info(f"\nOK Rapport JSON sauvegarde -> {OUT_JSON}")

    # Sauvegarde TXT
    header = "\n".join([
        "=" * 65,
        "RAPPORT D'ATTRIBUTION -- NEWS-DRIVEN REGIME ATTRIBUTION",
        "Pipeline IA / Finance Quantitative",
        f"Changements analyses : {len(all_attributions)}",
        "=" * 65,
        "",
    ])
    OUT_TXT.write_text(header + "\n".join(txt_sections), encoding="utf-8")
    log.info(f"OK Rapport TXT sauvegarde -> {OUT_TXT}")

    # Afficher le premier rapport comme exemple
    if txt_sections:
        print("\n--- EXEMPLE : PREMIER CHANGEMENT DE REGIME ---")
        print(txt_sections[0])


if __name__ == "__main__":
    main()