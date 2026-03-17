"""
02_regime_detect_adjusted_5regimes_v3.py
Détection de 5 régimes sans HMM
Basé sur tendance + volatilité + clustering KMeans
Labelling ajusté pour mieux répartir Bull/Bull Strong
Smoothing pour réduire sensibilité au bruit
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# =========================
# CONFIG
# =========================
INPUT_PATH = Path("data\\market\\nasdaq_10y.csv")
OUTPUT_PATH = Path("data\\processed\\regime_output.csv")

N_REGIMES = 5
RANDOM_STATE = 42
ROLLING_DAYS = 10  # pour rendement cumulatif
SMOOTH_DAYS = 20    # nombre de jours pour lisser les régimes

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
    df = df.copy()
    df['rendement_cumul'] = df['Rendement journalier'].rolling(ROLLING_DAYS).sum()
    df['vola_scaled'] = df['Volatilité glissante 20j'].fillna(0)

    feature_cols = ['rendement_cumul', 'vola_scaled']
    features = df[feature_cols].dropna()

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    features_df = pd.DataFrame(
        features_scaled,
        columns=feature_cols,
        index=features.index
    )
    return df.loc[features.index].copy(), features_df

# =========================
# CLUSTERING
# =========================

def cluster_regimes(features_df):
    kmeans = KMeans(n_clusters=N_REGIMES, random_state=RANDOM_STATE)
    clusters = kmeans.fit_predict(features_df)
    return clusters

# =========================
# LABEL REGIMES (5 régimes)
# =========================

def label_regimes(df):
    # Utilisation des quantiles pour répartir les régimes
    q20 = df['rendement_cumul'].quantile(0.2)
    q40 = df['rendement_cumul'].quantile(0.4)
    q60 = df['rendement_cumul'].quantile(0.6)
    q80 = df['rendement_cumul'].quantile(0.8)

    cluster_stats = df.groupby('cluster')['rendement_cumul'].mean()
    labels = {}
    for cid in cluster_stats.index:
        mean_ret = cluster_stats[cid]
        if mean_ret <= q20:
            labels[cid] = 'bear_strong'
        elif mean_ret <= q40:
            labels[cid] = 'bear'
        elif mean_ret <= q60:
            labels[cid] = 'sideways'
        elif mean_ret <= q80:
            labels[cid] = 'bull'
        else:
            labels[cid] = 'bull_strong'

    df['regime'] = df['cluster'].map(labels)
    return df

# =========================
# SMOOTHING DES REGIMES
# =========================

def smooth_regimes(df, window=SMOOTH_DAYS):
    # On convertit les labels en entiers pour appliquer le mode
    label_map = {r: i for i, r in enumerate(df['regime'].unique())}
    reverse_map = {i: r for r, i in label_map.items()}
    df['regime_num'] = df['regime'].map(label_map)

    # rolling mode pour lisser les variations ponctuelles
    df['regime_num_smooth'] = df['regime_num'].rolling(window, center=True, min_periods=1).apply(
        lambda x: pd.Series(x).mode().iloc[0], raw=False
    ).astype(int)

    df['regime'] = df['regime_num_smooth'].map(reverse_map)
    df.drop(['regime_num', 'regime_num_smooth'], axis=1, inplace=True)
    return df

# =========================
# DETECT CHANGES
# =========================

def detect_changes(df):
    df['regime_change'] = df['regime'].ne(df['regime'].shift())
    df.loc[df.index[0], 'regime_change'] = False
    return df

def filter_short_regimes(df, min_duration=5):
    df['group'] = (df['regime'] != df['regime'].shift()).cumsum()
    counts = df.groupby('group').size()
    valid_groups = counts[counts >= min_duration].index
    return df[df['group'].isin(valid_groups)]

# =========================
# MAIN
# =========================

def main():
    print("📥 Loading data...")
    df = load_data(INPUT_PATH)

    print("🧠 Building features...")
    df, features = build_features(df)

    print("🤖 Clustering regimes...")
    clusters = cluster_regimes(features)
    df['cluster'] = clusters

    print("🏷️ Labeling regimes...")
    df = label_regimes(df)

    print("🌀 Smoothing regimes to reduce noise...")
    df = smooth_regimes(df, window=SMOOTH_DAYS)

    print("🔁 Detecting regime changes...")
    df = detect_changes(df)

    print("🧹 Filtering short regimes...")
    df = filter_short_regimes(df, min_duration=5)

    df = detect_changes(df)

    output = df[['Date', 'regime', 'cluster', 'regime_change']].copy()
    output.columns = ['date', 'regime', 'regime_id', 'regime_change']

    print("💾 Saving...")
    output.to_csv(OUTPUT_PATH, index=False)

    print("✅ Done !")

if __name__ == "__main__":
    main()