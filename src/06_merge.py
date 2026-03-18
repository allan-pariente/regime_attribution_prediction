"""
Etape 7 - Fusion du dataset final
Pipeline : News-Driven Regime Attribution

Inputs :
  data/market/nasdaq_10y.csv          -> donnees de marche NASDAQ + VIX
  data/processed/regime_output.csv    -> regimes detectes + dates de changement
  data/processed/daily_sentiment.csv  -> agregat sentiment journalier
  data/processed/daily_embeddings.csv -> agregat embeddings journalier
  data/processed/regime_shift.csv     -> information shift par changement de regime

Output :
  data/processed/dataset_final.csv    -> dataset complet pret pour Granger + ML
  data/processed/dataset_granger.csv  -> subset leger sans embeddings pour Granger
"""

import pandas as pd
import numpy as np
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
MARKET_PATH     = Path("data\\market\\nasdaq_10y.csv")
REGIME_PATH     = Path("data\\processed\\regime_output.csv")
SENTIMENT_PATH  = Path("data\\processed\\daily_sentiment.csv")
EMBEDDINGS_PATH = Path("data\\processed\\daily_embeddings.csv")
SHIFT_PATH      = Path("data\\processed\\regime_shift.csv")
OUT_PATH        = Path("data\\processed\\dataset_final.csv")
OUT_GRANGER     = Path("data\\processed\\dataset_granger.csv")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Chargement
# ══════════════════════════════════════════════════════════════════════════════

def load_market(path: Path) -> pd.DataFrame:
    log.info(f"Chargement marche : {path} ...")
    # Lire tout en string pour controler la conversion numerique
    df = pd.read_csv(path, sep=";", dtype=str)

    # Renommer par position (insensible aux accents / espaces)
    df.columns = [
        "date", "close", "volume", "returns", "returns_pct",
        "volatility_20d", "rsi_14", "macd", "macd_signal",
        "macd_hist", "drawdown", "drawdown_pct", "vix"
    ]

    df["date"] = pd.to_datetime(df["date"].str.strip())
    df = df.drop(columns=["volume"])

    # Conversion numerique : supprimer espaces insecables, remplacer virgule par point
    num_cols = [
        "close", "returns", "returns_pct", "volatility_20d",
        "rsi_14", "macd", "macd_signal", "macd_hist",
        "drawdown", "drawdown_pct", "vix"
    ]
    for col in num_cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.replace("\xa0", "", regex=False)
            .str.replace("\u202f", "", regex=False)
            .str.replace(" ", "", regex=False)
            .str.replace(",", ".", regex=False)
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("date").reset_index(drop=True)
    log.info(f"  {len(df)} jours ({df['date'].min().date()} -> {df['date'].max().date()})")
    log.info(f"  volatility_20d non-NaN : {df['volatility_20d'].notna().sum()}")
    log.info(f"  rsi_14 non-NaN         : {df['rsi_14'].notna().sum()}")
    return df


def load_regime(path: Path) -> pd.DataFrame:
    log.info(f"Chargement regimes : {path} ...")
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    log.info(f"  {len(df)} jours | {df['regime_change'].sum()} changements detectes")
    return df


def load_sentiment(path: Path) -> pd.DataFrame:
    log.info(f"Chargement sentiment : {path} ...")
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def load_embeddings(path: Path) -> pd.DataFrame:
    log.info(f"Chargement embeddings journaliers : {path} ...")
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def load_shift(path: Path) -> pd.DataFrame:
    log.info(f"Chargement information shift : {path} ...")
    df = pd.read_csv(path, parse_dates=["regime_change_date"])
    df = df.rename(columns={"regime_change_date": "date"})
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. Fusion
# ══════════════════════════════════════════════════════════════════════════════

def merge_all(market, regime, sentiment, embeddings, shift) -> pd.DataFrame:
    log.info("Fusion des datasets ...")

    df = market.copy()

    # Regime
    regime_cols = ["date", "regime", "regime_id", "regime_change"]
    df = df.merge(regime[regime_cols], on="date", how="left")

    # Sentiment journalier
    df = df.merge(sentiment, on="date", how="left")

    # Embeddings journaliers
    df = df.merge(embeddings, on="date", how="left")

    # Information shift : propagation forward jusqu'au prochain changement
    df = df.merge(shift[["date", "information_shift"]], on="date", how="left")
    df["information_shift"] = df["information_shift"].ffill()

    # Variable cible binaire
    df["y"] = df["regime_change"].fillna(False).astype(int)

    # Remplir NaN des colonnes news par 0
    news_cols = [c for c in df.columns
                 if c.startswith(("sentiment_", "pct_", "news_count", "topic_", "emb_"))]
    df[news_cols] = df[news_cols].fillna(0)

    df = df.sort_values("date").reset_index(drop=True)
    log.info(f"Dataset final : {len(df)} lignes x {len(df.columns)} colonnes")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 3. Controle qualite
# ══════════════════════════════════════════════════════════════════════════════

def quality_check(df: pd.DataFrame) -> None:
    log.info("-- Controle qualite ------------------------------------------")
    log.info(f"Periode couverte    : {df['date'].min().date()} -> {df['date'].max().date()}")
    log.info(f"Jours de trading    : {len(df)}")
    log.info(f"Changements regime  : {df['y'].sum()}")
    log.info(f"Desequilibre classes: {df['y'].mean():.2%} de jours avec changement")

    nan_rates = df.isnull().mean()
    problematic = nan_rates[nan_rates > 0.3]
    if len(problematic):
        for col, rate in problematic.items():
            log.warning(f"  {col} : {rate:.0%} de valeurs manquantes")
    else:
        log.info("OK Aucune colonne avec plus de 30% de NaN")

    required = ["returns", "volatility_20d", "vix", "sentiment_mean",
                "news_count", "information_shift", "regime", "y"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        log.warning(f"  Colonnes manquantes : {missing}")
    else:
        log.info("OK Toutes les colonnes cles sont presentes")

    # Spot check
    sample = df.dropna(subset=["regime"]).iloc[5]
    log.info(f"Spot check (ligne 5 avec regime) :")
    log.info(f"  date={sample['date'].date()} returns={sample['returns']:.4f} "
             f"volatility={sample['volatility_20d']} rsi={sample['rsi_14']} "
             f"regime={sample['regime']}")
    log.info("--------------------------------------------------------------")


# ══════════════════════════════════════════════════════════════════════════════
# 4. Export subset Granger
# ══════════════════════════════════════════════════════════════════════════════

def export_granger_subset(df: pd.DataFrame) -> None:
    """
    Subset leger pour le test de Granger :
    series temporelles continues uniquement, sans les 384 dims d'embeddings.
    """
    granger_cols = [
        "date", "returns", "volatility_20d", "vix", "rsi_14",
        "drawdown", "sentiment_mean", "sentiment_std", "sentiment_min",
        "news_count", "pct_negative", "information_shift",
        "regime", "regime_id", "y"
    ]
    granger_cols = [c for c in granger_cols if c in df.columns]
    granger_df = df[granger_cols].dropna(subset=["returns", "sentiment_mean"])
    granger_df.to_csv(OUT_GRANGER, index=False)
    log.info(f"OK Subset Granger sauvegarde -> {OUT_GRANGER} ({len(granger_df)} lignes)")


# ══════════════════════════════════════════════════════════════════════════════
# 5. Point d'entree
# ══════════════════════════════════════════════════════════════════════════════

def main():
    market     = load_market(MARKET_PATH)
    regime     = load_regime(REGIME_PATH)
    sentiment  = load_sentiment(SENTIMENT_PATH)
    embeddings = load_embeddings(EMBEDDINGS_PATH)
    shift      = load_shift(SHIFT_PATH)

    df = merge_all(market, regime, sentiment, embeddings, shift)
    quality_check(df)

    df.to_csv(OUT_PATH, index=False)
    log.info(f"OK Dataset final sauvegarde -> {OUT_PATH}")

    export_granger_subset(df)


if __name__ == "__main__":
    main()