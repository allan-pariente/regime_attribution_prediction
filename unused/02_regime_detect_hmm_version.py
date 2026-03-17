"""
02_regime_detect.py
Détection de régimes avec HMM (hmmlearn) et ajout de la tendance
Compatible avec CSV NASDAQ (format FR, virgules, colonnes custom)
"""

import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
from pathlib import Path

# =========================
# CONFIG
# =========================
INPUT_PATH = Path("data\\market\\nasdaq_10y.csv")
OUTPUT_PATH = Path("data\\market\\regime_output.csv")

N_REGIMES = 3
RANDOM_STATE = 42
TENDANCE_WINDOW = 20  # fenêtre glissante pour calculer la tendance


# =========================
# LOAD + CLEAN DATA
# =========================

def clean_numeric(df):
    for col in df.columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", ".", regex=False)
            .replace(["", "None", "nan"], np.nan)
        )

    # Convertir en float quand possible
    for col in df.columns:
        try:
            df[col] = df[col].astype(float)
        except:
            pass

    return df

def load_data(path):
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    df.columns = df.columns.str.strip().str.replace("\ufeff", "", regex=False)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    df = clean_numeric(df)
    return df


# =========================
# FEATURE ENGINEERING
# =========================

def build_features(df):
    from sklearn.preprocessing import StandardScaler

    feature_cols = [
        "Rendement journalier",
        "Volatilité glissante 20j",
        "RSI 14j",
        "MACD"
    ]

    features = df[feature_cols].copy()
    features = features.dropna()

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    features_df = pd.DataFrame(
        features_scaled,
        columns=feature_cols,
        index=features.index
    )
    
    return df.loc[features.index].copy(), features_df


# =========================
# FIT HMM
# =========================
def fit_hmm(features):
    X = features.values

    model = GaussianHMM(
        n_components=N_REGIMES,
        covariance_type="full",
        n_iter=300,
        random_state=RANDOM_STATE
    )

    model.fit(X)
    states = model.predict(X)

    return states, model


# =========================
# LABEL REGIMES
# =========================
def label_regimes(df):
    stats = df.groupby("regime_id")["Rendement journalier"].mean().sort_values()
    labels = ["bear", "sideways", "bull"]
    mapping = {rid: labels[i] for i, rid in enumerate(stats.index)}
    df["regime"] = df["regime_id"].map(mapping)
    return df


# =========================
# DETECT CHANGES
# =========================
def detect_changes(df):
    df["regime_change"] = df["regime_id"].diff().ne(0)
    df.loc[df.index[0], "regime_change"] = False
    return df

def filter_regimes(df, min_duration=5):
    df["group"] = (df["regime_id"] != df["regime_id"].shift()).cumsum()
    counts = df.groupby("group").size()
    valid_groups = counts[counts >= min_duration].index
    return df[df["group"].isin(valid_groups)]


# =========================
# AJOUT DE LA TENDANCE
# =========================
def add_trend(df, window=TENDANCE_WINDOW, threshold=0.002):
    """
    Calcule la tendance sur une fenêtre glissante.
    - up: hausse (> threshold)
    - down: baisse (< -threshold)
    - flat: neutre
    """
    df = df.copy()
    df["rolling_return"] = df["Rendement journalier"].rolling(window=window).mean()
    
    def trend_label(r):
        if pd.isna(r):
            return np.nan
        elif r > threshold:
            return "up"
        elif r < -threshold:
            return "down"
        else:
            return "flat"
    
    df["tendance"] = df["rolling_return"].apply(trend_label)
    df.drop(columns=["rolling_return"], inplace=True)
    return df


# =========================
# MAIN
# =========================
def main():
    print("📥 Loading data...")
    df = load_data(INPUT_PATH)

    print("🧠 Building features...")
    df, features = build_features(df)

    print("🤖 Training HMM...")
    states, model = fit_hmm(features)
    df["regime_id"] = states

    print("🏷️ Labeling regimes...")
    df = label_regimes(df)

    print("🔁 Detecting regime changes...")
    df = detect_changes(df)

    print("🧹 Filtering short regimes...")
    df = filter_regimes(df, min_duration=5)
    df = detect_changes(df)

    print("📈 Adding trend information...")
    df = add_trend(df, window=TENDANCE_WINDOW)

    # Format final
    output = df[["Date", "regime", "regime_id", "regime_change", "tendance"]].copy()
    output.columns = ["date", "regime", "regime_id", "regime_change", "tendance"]

    print("💾 Saving...")
    output.to_csv(OUTPUT_PATH, index=False)

    print("✅ Done !")

if __name__ == "__main__":
    main()