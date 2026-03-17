"""
Étape 4 — Analyse de sentiment avec FinBERT
Pipeline : News-Driven Regime Attribution

Input  : data/processed/news_def.csv
         colonnes : headline, summary, source, url, datetime, category,
                    regime_change_date, _source_api, _category_query,
                    _ticker_query, date

Output : data/processed/news_with_sentiment.csv   (news enrichies)
         data/processed/daily_sentiment.csv        (agrégat journalier)
"""

import pandas as pd
import numpy as np
from transformers import pipeline
from pathlib import Path
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
INPUT_PATH  = Path("data\\processed\\news_df.csv")
OUT_NEWS    = Path("data\\processed\\news_with_sentiment.csv")
OUT_DAILY   = Path("data\\processed\\daily_sentiment.csv")
OUT_NEWS.parent.mkdir(parents=True, exist_ok=True)

# ── Paramètres ─────────────────────────────────────────────────────────────────
BATCH_SIZE  = 32          # réduire à 16 si OOM sur CPU
MAX_LENGTH  = 512
TEXT_COLUMN = "headline"  # colonne principale analysée ; on enrichit aussi avec summary


# ══════════════════════════════════════════════════════════════════════════════
# 1. Chargement des données
# ══════════════════════════════════════════════════════════════════════════════

def load_news(path: Path) -> pd.DataFrame:
    log.info(f"Chargement de {path} …")
    df = pd.read_csv(path, parse_dates=["datetime", "regime_change_date", "date"])

    # Texte tronqué à MAX_LENGTH caractères (sécurité avant tokenisation)
    df["text_for_finbert"] = df[TEXT_COLUMN].fillna("").str.strip().str[:MAX_LENGTH]

    before = len(df)
    df = df[df["text_for_finbert"].str.len() > 0].reset_index(drop=True)
    log.info(f"{len(df)} articles chargés ({before - len(df)} lignes vides supprimées).")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. Chargement de FinBERT
# ══════════════════════════════════════════════════════════════════════════════

def load_finbert():
    log.info("Chargement du modèle FinBERT (ProsusAI/finbert) …")
    finbert = pipeline(
        "text-classification",
        model="ProsusAI/finbert",
        tokenizer="ProsusAI/finbert",
        truncation=True,
        max_length=MAX_LENGTH,
        device=-1,          # CPU ; mettre 0 pour GPU CUDA si disponible
    )
    log.info("Modèle chargé.")
    return finbert


# ══════════════════════════════════════════════════════════════════════════════
# 3. Inférence FinBERT en batch
# ══════════════════════════════════════════════════════════════════════════════

def run_finbert_batch(finbert, texts: list[str]) -> tuple[list[float], list[str]]:
    """
    Renvoie (scores_numériques, labels_bruts).
    score numérique :  +confidence si positive
                       -confidence si negative
                        0.0        si neutral
    """
    results = finbert(texts, batch_size=BATCH_SIZE)

    scores = []
    labels = []
    for r in results:
        lbl   = r["label"]   # 'positive' | 'negative' | 'neutral'
        conf  = r["score"]   # probabilité de la classe prédite [0, 1]
        labels.append(lbl)
        if lbl == "positive":
            scores.append(conf)
        elif lbl == "negative":
            scores.append(-conf)
        else:
            scores.append(0.0)

    return scores, labels


def analyze_news(df: pd.DataFrame, finbert) -> pd.DataFrame:
    log.info(f"Inférence FinBERT sur {len(df)} articles (batch={BATCH_SIZE}) …")
    texts = df["text_for_finbert"].tolist()

    scores, labels = run_finbert_batch(finbert, texts)

    df = df.copy()
    df["sentiment"]       = scores           # valeur numérique [-1, +1]
    df["sentiment_label"] = labels           # 'positive' | 'negative' | 'neutral'
    df["sentiment_abs"]   = df["sentiment"].abs()  # utilisé par le module d'attribution

    log.info("Inférence terminée.")
    log.info(
        f"  positive : {(df['sentiment_label']=='positive').sum()}"
        f"  negative : {(df['sentiment_label']=='negative').sum()}"
        f"  neutral  : {(df['sentiment_label']=='neutral').sum()}"
    )
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 4. Agrégation journalière
# ══════════════════════════════════════════════════════════════════════════════

def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Produit un agrégat par (date) avec :
      - sentiment_mean  : moyenne des scores → ton général du marché
      - sentiment_std   : dispersion → désaccord / incertitude
      - sentiment_min   : pic de panique (score le plus négatif)
      - sentiment_max   : pic d'euphorie
      - news_count      : volume médiatique
      - pct_positive    : fraction d'articles positifs
      - pct_negative    : fraction d'articles négatifs
    """
    log.info("Agrégation journalière …")

    daily = (
        df.groupby("date")
        .agg(
            sentiment_mean  =("sentiment",       "mean"),
            sentiment_std   =("sentiment",       "std"),
            sentiment_min   =("sentiment",       "min"),
            sentiment_max   =("sentiment",       "max"),
            news_count      =("headline",        "count"),
            pct_positive    =("sentiment_label", lambda x: (x == "positive").mean()),
            pct_negative    =("sentiment_label", lambda x: (x == "negative").mean()),
        )
        .reset_index()
    )

    daily["sentiment_std"] = daily["sentiment_std"].fillna(0.0)

    # Anomalie de sentiment : écart par rapport à la moyenne glissante sur 20 jours
    daily = daily.sort_values("date").reset_index(drop=True)
    rolling_mean = daily["sentiment_mean"].rolling(20, min_periods=5).mean()
    rolling_std  = daily["sentiment_mean"].rolling(20, min_periods=5).std().replace(0, np.nan)
    daily["sentiment_anomaly"] = (daily["sentiment_mean"] - rolling_mean) / rolling_std
    daily["sentiment_anomaly"] = daily["sentiment_anomaly"].fillna(0.0)

    log.info(f"Agrégat journalier : {len(daily)} jours.")
    return daily


# ══════════════════════════════════════════════════════════════════════════════
# 5. Contrôle qualité rapide
# ══════════════════════════════════════════════════════════════════════════════

def quality_check(df_news: pd.DataFrame, df_daily: pd.DataFrame) -> None:
    log.info("── Contrôle qualité ──────────────────────────────────────")
    log.info(f"Articles total          : {len(df_news)}")
    log.info(f"Couverture dates        : {df_news['date'].min()} → {df_news['date'].max()}")
    log.info(f"Jours avec news         : {df_daily['date'].nunique()}")
    log.info(f"Sentiment moyen global  : {df_news['sentiment'].mean():.4f}")
    log.info(f"Sentiment std global    : {df_news['sentiment'].std():.4f}")

    # Alerte si trop de neutres (signe que le texte est tronqué ou mal formé)
    pct_neutral = (df_news["sentiment_label"] == "neutral").mean()
    if pct_neutral > 0.6:
        log.warning(
            f"⚠  {pct_neutral:.0%} d'articles neutres — vérifier la qualité du champ headline."
        )

    # Alerte si des jours n'ont aucune news autour d'un changement de régime
    change_dates = df_news["regime_change_date"].dropna().unique()
    covered = set(str(d.date()) for d in pd.to_datetime(change_dates))
    daily_dates = set(str(d) for d in df_daily["date"])
    missing = covered - daily_dates
    if missing:
        log.warning(
            f"⚠  {len(missing)} date(s) de changement de régime sans news : {sorted(missing)}"
        )
    else:
        log.info("✅ Tous les changements de régime ont des news associées.")
    log.info("──────────────────────────────────────────────────────────")


# ══════════════════════════════════════════════════════════════════════════════
# 6. Point d'entrée
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # Chargement
    df = load_news(INPUT_PATH)

    # Modèle
    finbert = load_finbert()

    # Inférence
    df = analyze_news(df, finbert)

    # Agrégat
    daily = aggregate_daily(df)

    # Contrôle qualité
    quality_check(df, daily)

    # Sauvegarde
    df.to_csv(OUT_NEWS, index=False)
    log.info(f"✅ News enrichies sauvegardées → {OUT_NEWS}")

    daily.to_csv(OUT_DAILY, index=False)
    log.info(f"✅ Agrégat journalier sauvegardé → {OUT_DAILY}")


if __name__ == "__main__":
    main()