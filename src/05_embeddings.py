"""
Étape 6 — Embeddings de texte & Information Shift
Pipeline : News-Driven Regime Attribution

Input  : data/processed/news_with_sentiment.csv
         colonnes : headline, datetime, date, regime_change_date, sentiment, ...

Output : data/processed/news_with_embeddings.csv   (news + topic assigné)
         data/processed/embeddings.npy             (matrice brute, shape N×384)
         data/processed/daily_embeddings.csv       (agrégat journalier vectoriel)
         data/processed/regime_shift.csv           (information shift par changement de régime)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import timedelta
from scipy.spatial.distance import cosine
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
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
INPUT_PATH      = Path("data\\processed\\news_with_sentiment.csv")
OUT_NEWS        = Path("data\\processed\\news_with_embeddings.csv")
OUT_EMBEDDINGS  = Path("data\\processed\\embeddings.npy")
OUT_DAILY       = Path("data\\processed\\daily_embeddings.csv")
OUT_SHIFT       = Path("data\\processed\\regime_shift.csv")
OUT_NEWS.parent.mkdir(parents=True, exist_ok=True)

# ── Paramètres ─────────────────────────────────────────────────────────────────
MODEL_NAME   = "all-MiniLM-L6-v2"   # 384 dimensions, bon rapport vitesse/qualité
BATCH_SIZE   = 64
N_CLUSTERS   = 8                     # nombre de topics sémantiques
PCA_DIMS     = 50                    # réduction avant clustering
SHIFT_WINDOW = 5                     # jours avant/après le changement de régime


# ══════════════════════════════════════════════════════════════════════════════
# 1. Chargement
# ══════════════════════════════════════════════════════════════════════════════

def load_news(path: Path) -> pd.DataFrame:
    log.info(f"Chargement de {path} …")
    df = pd.read_csv(path, parse_dates=["datetime", "regime_change_date", "date"])
    df = df.dropna(subset=["headline"]).reset_index(drop=True)
    log.info(f"{len(df)} articles chargés.")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. Calcul des embeddings
# ══════════════════════════════════════════════════════════════════════════════

def compute_embeddings(df: pd.DataFrame) -> np.ndarray:
    """
    Encode chaque headline en vecteur dense de 384 dimensions.
    Retourne un array numpy de shape (N, 384).
    """
    log.info(f"Chargement du modèle '{MODEL_NAME}' …")
    model = SentenceTransformer(MODEL_NAME)

    log.info(f"Encodage de {len(df)} headlines (batch={BATCH_SIZE}) …")
    embeddings = model.encode(
        df["headline"].tolist(),
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    log.info(f"Embeddings calculés — shape : {embeddings.shape}")
    return embeddings


# ══════════════════════════════════════════════════════════════════════════════
# 3. Clustering en topics sémantiques
# ══════════════════════════════════════════════════════════════════════════════

def cluster_topics(embeddings: np.ndarray, df: pd.DataFrame) -> pd.DataFrame:
    """
    Réduit la dimension avec PCA puis applique K-Means.
    Ajoute la colonne 'topic' (int 0..N_CLUSTERS-1) au DataFrame.
    """
    log.info(f"Réduction PCA vers {PCA_DIMS} dimensions …")
    pca = PCA(n_components=PCA_DIMS, random_state=42)
    emb_reduced = pca.fit_transform(embeddings)
    variance_explained = pca.explained_variance_ratio_.sum()
    log.info(f"Variance expliquée par {PCA_DIMS} composantes : {variance_explained:.1%}")

    log.info(f"Clustering K-Means en {N_CLUSTERS} topics …")
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    df = df.copy()
    df["topic"] = kmeans.fit_predict(emb_reduced)

    # Résumé : taille de chaque cluster
    topic_counts = df["topic"].value_counts().sort_index()
    for topic_id, count in topic_counts.items():
        log.info(f"  Topic {topic_id} : {count} articles")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# 4. Information shift autour des changements de régime
# ══════════════════════════════════════════════════════════════════════════════

def information_shift(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    change_date: pd.Timestamp,
    window: int = SHIFT_WINDOW,
) -> float:
    """
    Mesure la distance cosinus entre l'embedding moyen des news
    dans la fenêtre [date - window - 3 ; date - 1] et [date ; date + window].

    Retourne un float entre 0 (aucun changement sémantique) et 1 (rupture totale).
    """
    dates = df["date"].dt.date if hasattr(df["date"].dt, "date") else df["date"]

    before_mask = (dates >= (change_date - timedelta(days=window + 3)).date()) & \
                  (dates <  change_date.date())
    after_mask  = (dates >= change_date.date()) & \
                  (dates <= (change_date + timedelta(days=window)).date())

    idx_before = df[before_mask].index
    idx_after  = df[after_mask].index

    if len(idx_before) == 0 or len(idx_after) == 0:
        return np.nan

    emb_before = embeddings[idx_before].mean(axis=0)
    emb_after  = embeddings[idx_after].mean(axis=0)

    return float(cosine(emb_before, emb_after))


def compute_all_shifts(df: pd.DataFrame, embeddings: np.ndarray) -> pd.DataFrame:
    """
    Calcule l'information shift pour chaque date de changement de régime unique.
    """
    log.info("Calcul de l'information shift par changement de régime …")
    change_dates = pd.to_datetime(
        df["regime_change_date"].dropna().unique()
    )

    records = []
    for cd in sorted(change_dates):
        shift = information_shift(df, embeddings, cd)
        records.append({"regime_change_date": cd, "information_shift": shift})
        log.info(f"  {cd.date()} → shift = {shift:.4f}" if not np.isnan(shift)
                 else f"  {cd.date()} → pas assez de news")

    shift_df = pd.DataFrame(records)
    log.info(f"Information shift calculé pour {len(shift_df)} changements de régime.")
    return shift_df


# ══════════════════════════════════════════════════════════════════════════════
# 5. Agrégat journalier
# ══════════════════════════════════════════════════════════════════════════════

def aggregate_daily(df: pd.DataFrame, embeddings: np.ndarray) -> pd.DataFrame:
    """
    Pour chaque jour, calcule :
      - embedding_mean_0..383 : centroïde des embeddings du jour (384 features)
      - topic_dominant         : topic le plus fréquent du jour
      - topic_diversity        : nombre de topics distincts présents
      - news_count             : nombre d'articles
    """
    log.info("Agrégation journalière des embeddings …")

    df = df.copy()
    df["_emb_idx"] = df.index  # sauvegarder l'index pour récupérer les embeddings

    records = []
    for date, group in df.groupby("date"):
        idxs = group["_emb_idx"].values
        emb_mean = embeddings[idxs].mean(axis=0)          # vecteur (384,)
        record = {"date": date}
        for i, v in enumerate(emb_mean):
            record[f"emb_{i}"] = v
        record["topic_dominant"]  = group["topic"].mode()[0]
        record["topic_diversity"] = group["topic"].nunique()
        record["news_count"]      = len(group)
        records.append(record)

    daily = pd.DataFrame(records)
    log.info(f"Agrégat journalier : {len(daily)} jours, {len(daily.columns)} colonnes.")
    return daily


# ══════════════════════════════════════════════════════════════════════════════
# 6. Point d'entrée
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # Chargement
    df = load_news(INPUT_PATH)

    # Embeddings
    embeddings = compute_embeddings(df)
    np.save(OUT_EMBEDDINGS, embeddings)
    log.info(f"✅ Embeddings bruts sauvegardés → {OUT_EMBEDDINGS}")

    # Topics
    df = cluster_topics(embeddings, df)
    df.to_csv(OUT_NEWS, index=False)
    log.info(f"✅ News avec topics sauvegardées → {OUT_NEWS}")

    # Information shift
    shift_df = compute_all_shifts(df, embeddings)
    shift_df.to_csv(OUT_SHIFT, index=False)
    log.info(f"✅ Information shift sauvegardé → {OUT_SHIFT}")

    # Agrégat journalier
    daily = aggregate_daily(df, embeddings)
    daily.to_csv(OUT_DAILY, index=False)
    log.info(f"✅ Agrégat journalier sauvegardé → {OUT_DAILY}")


if __name__ == "__main__":
    main()