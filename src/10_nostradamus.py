"""
Etape 11 - Prediction en temps reel
Pipeline : News-Driven Regime Attribution

Collecte les news des 5 derniers jours via Finnhub + Guardian,
calcule les features (sentiment FinBERT + embeddings), et predit
si un changement de regime est imminent et lequel.

Inputs (modeles pre-entraines) :
  data/models/random_forest.pkl         -> classifieur entraine
  data/processed/embeddings.npy         -> embeddings historiques (pour PCA/KMeans)
  data/processed/news_with_embeddings.csv -> pour reutiliser le PCA et KMeans fites

Outputs :
  data/live/prediction_YYYYMMDD.json    -> rapport structure
  data/live/prediction_YYYYMMDD.txt     -> rapport lisible
"""

import pandas as pd
import numpy as np
import requests
import joblib
import json
import time
import logging
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from transformers import pipeline as hf_pipeline
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Cles API ───────────────────────────────────────────────────────────────────
FINNHUB_API_KEY  = "our_finnhub_key"   # https://finnhub.io
GUARDIAN_API_KEY = "your_guardian_key"  # https://open-platform.theguardian.com

# ── Chemins ────────────────────────────────────────────────────────────────────
MODEL_PATH     = Path("data\\models\\random_forest.pkl")
DATASET_PATH   = Path("data\\processed\\dataset_final.csv")
NEWS_HIST_PATH = Path("data\\processed\\news_with_embeddings.csv")
OUT_DIR        = Path("data\\live")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Parametres ─────────────────────────────────────────────────────────────────
WINDOW_DAYS    = 5
N_CLUSTERS     = 8
PCA_DIMS       = 50
BATCH_SIZE     = 32
PROBA_THRESHOLD_HIGH   = 0.35   # seuil alerte forte
PROBA_THRESHOLD_MEDIUM = 0.20   # seuil alerte moderee

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
# 1. Collecte des news
# ══════════════════════════════════════════════════════════════════════════════

def fetch_finnhub(from_date: str, to_date: str) -> list:
    """Recupere les news generales via Finnhub."""
    log.info(f"  Finnhub : {from_date} -> {to_date} ...")
    url = "https://finnhub.io/api/v1/news"
    params = {
        "category": "general",
        "from": from_date,
        "to": to_date,
        "token": FINNHUB_API_KEY,
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        articles = []
        for item in data:
            if not item.get("headline"):
                continue
            articles.append({
                "headline":  item["headline"],
                "source":    item.get("source", "finnhub"),
                "datetime":  pd.to_datetime(item["datetime"], unit="s"),
                "url":       item.get("url", ""),
            })
        log.info(f"    {len(articles)} articles Finnhub")
        return articles
    except Exception as e:
        log.warning(f"    Finnhub erreur : {e}")
        return []


def fetch_guardian(from_date: str, to_date: str) -> list:
    """Recupere les news business/economics via The Guardian."""
    log.info(f"  Guardian : {from_date} -> {to_date} ...")
    url = "https://content.guardianapis.com/search"
    params = {
        "from-date":  from_date,
        "to-date":    to_date,
        "section":    "business|money|technology",
        "page-size":  200,
        "api-key":    GUARDIAN_API_KEY,
        "show-fields":"headline",
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        results = r.json().get("response", {}).get("results", [])
        articles = []
        for item in results:
            headline = item.get("webTitle") or (item.get("fields") or {}).get("headline", "")
            if not headline:
                continue
            articles.append({
                "headline": headline,
                "source":   "guardian",
                "datetime": pd.to_datetime(item["webPublicationDate"]),
                "url":      item.get("webUrl", ""),
            })
        log.info(f"    {len(articles)} articles Guardian")
        return articles
    except Exception as e:
        log.warning(f"    Guardian erreur : {e}")
        return []


def collect_news(today: datetime) -> pd.DataFrame:
    log.info("Collecte des news ...")
    from_date = (today - timedelta(days=WINDOW_DAYS)).strftime("%Y-%m-%d")
    to_date   = today.strftime("%Y-%m-%d")

    articles = fetch_finnhub(from_date, to_date)
    time.sleep(1)
    articles += fetch_guardian(from_date, to_date)

    if not articles:
        log.warning("Aucun article collecte — verification des cles API requise.")
        return pd.DataFrame()

    df = pd.DataFrame(articles).drop_duplicates(subset=["headline"])
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_localize(None)
    df["date"] = df["datetime"].dt.date
    df = df.sort_values("datetime").reset_index(drop=True)
    log.info(f"Total : {len(df)} articles uniques sur {df['date'].nunique()} jours")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. Sentiment FinBERT
# ══════════════════════════════════════════════════════════════════════════════

def compute_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Calcul sentiment FinBERT ...")
    finbert = hf_pipeline(
        "text-classification",
        model="ProsusAI/finbert",
        tokenizer="ProsusAI/finbert",
        truncation=True,
        max_length=512,
        device=-1,
    )
    texts = df["headline"].str[:512].tolist()
    results = finbert(texts, batch_size=BATCH_SIZE)

    scores, labels = [], []
    for r in results:
        lbl, conf = r["label"], r["score"]
        labels.append(lbl)
        if lbl == "positive":   scores.append(conf)
        elif lbl == "negative": scores.append(-conf)
        else:                   scores.append(0.0)

    df = df.copy()
    df["sentiment"]       = scores
    df["sentiment_label"] = labels
    df["sentiment_abs"]   = df["sentiment"].abs()
    log.info(f"  pos={sum(1 for l in labels if l=='positive')} "
             f"neg={sum(1 for l in labels if l=='negative')} "
             f"neu={sum(1 for l in labels if l=='neutral')}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 3. Embeddings + topics (reutilise les modeles fits sur historique)
# ══════════════════════════════════════════════════════════════════════════════

def compute_embeddings_and_topics(df_live: pd.DataFrame,
                                   df_hist: pd.DataFrame) -> tuple:
    """
    Encode les headlines live, puis applique le PCA et K-Means
    fites sur les donnees historiques pour garantir la coherence
    avec le modele entraine.
    """
    log.info("Calcul embeddings ...")
    encoder = SentenceTransformer("all-MiniLM-L6-v2")

    # Encoder les headlines live
    emb_live = encoder.encode(
        df_live["headline"].tolist(),
        batch_size=64,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    log.info(f"  Embeddings live shape : {emb_live.shape}")

    # Refit PCA et KMeans sur historique pour coherence
    log.info("  Refit PCA + KMeans sur historique ...")
    emb_hist_cols = [c for c in df_hist.columns if c.startswith("emb_")]

    if len(emb_hist_cols) > 0:
        # Utiliser les embeddings historiques pre-calcules
        emb_hist = df_hist[emb_hist_cols].values
    else:
        log.warning("  Pas d'embeddings historiques — refit impossible sur historique")
        emb_hist = emb_live

    # Fit PCA sur historique
    pca = PCA(n_components=min(PCA_DIMS, emb_hist.shape[0]-1), random_state=42)
    emb_hist_pca = pca.fit_transform(emb_hist)

    # Fit KMeans sur historique
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    kmeans.fit(emb_hist_pca)

    # Transformer les embeddings live avec le meme PCA/KMeans
    emb_live_pca = pca.transform(emb_live)
    df_live = df_live.copy()
    df_live["topic"] = kmeans.predict(emb_live_pca)

    log.info(f"  Topics assignes : {dict(pd.Series(df_live['topic']).value_counts().sort_index())}")
    return df_live, emb_live, pca, kmeans


# ══════════════════════════════════════════════════════════════════════════════
# 3b. Noms descriptifs pour les topics
# ══════════════════════════════════════════════════════════════════════════════

def generate_topic_names(df_hist: pd.DataFrame) -> dict:
    """
    Genere des noms descriptifs pour chaque topic basés sur les headlines historiques.
    Tire les mots-clés dominants et crée des noms intuitifs.
    """
    from collections import Counter
    import re
    
    log.info("Generation des noms de topics ...")
    
    # Mots vides (stop words) à exclure
    stopwords = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
        'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
        'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
    }
    
    topic_names = {}
    
    if 'topic' not in df_hist.columns or 'headline' not in df_hist.columns:
        log.warning("  Colonnes topic/headline manquantes — noms par défaut")
        return {i: f"Topic {i}" for i in range(N_CLUSTERS)}
    
    for topic_id in range(N_CLUSTERS):
        headlines = df_hist[df_hist['topic'] == topic_id]['headline'].tolist()
        
        if not headlines:
            topic_names[topic_id] = f"Topic {topic_id}"
            continue
        
        # Extraire les mots-clés
        all_words = []
        for headline in headlines[:100]:  # Limiter à 100 headlines par topic
            # Nettoyer et tokeniser
            words = re.findall(r'\b[a-z]+\b', headline.lower())
            all_words.extend([w for w in words if w not in stopwords and len(w) > 3])
        
        if not all_words:
            topic_names[topic_id] = f"Topic {topic_id}"
            continue
        
        # Top 5 mots
        counter = Counter(all_words)
        top_words = [w for w, count in counter.most_common(5)]
        
        # Créer un nom descriptif
        name = " / ".join(top_words[:3]).title()
        topic_names[topic_id] = name
        log.info(f"  Topic {topic_id}: {name}")
    
    return topic_names


# ══════════════════════════════════════════════════════════════════════════════
# 4. Features marche actuelles
# ══════════════════════════════════════════════════════════════════════════════

def fetch_fred_series(series_id: str) -> pd.Series:
    """Recupere une serie FRED via l'API CSV publique."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    from io import StringIO
    r = requests.get(url, verify=False, timeout=15)
    df = pd.read_csv(StringIO(r.text))
    df.columns = ["Date", series_id]
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")
    df[series_id] = pd.to_numeric(df[series_id], errors="coerce")
    return df[series_id]


def fetch_yahoo_direct(today: datetime) -> pd.Series:
    """Recupere le volume NASDAQ via Yahoo Finance direct (sans yfinance)."""
    from io import StringIO
    url = (
        "https://query1.finance.yahoo.com/v7/finance/download/%5EIXIC"
        "?period1=1388534400&period2=9999999999&interval=1d&events=history"
    )
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers, verify=False, timeout=15)
    df = pd.read_csv(StringIO(r.text))
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")
    return df["Volume"].apply(pd.to_numeric, errors="coerce")


def get_market_features(today: datetime) -> dict:
    """Telecharge les donnees de marche recentes via FRED + Yahoo direct."""
    log.info("Telechargement donnees marche recentes (FRED + Yahoo) ...")

    try:
        # NASDAQ Composite via FRED
        nasdaq_close = fetch_fred_series("NASDAQCOM")

        # VIX via FRED
        vix_series = fetch_fred_series("VIXCLS")

        if nasdaq_close.empty:
            log.warning("Donnees NASDAQ FRED indisponibles")
            return {}

        # Assembler le dataframe sur les 60 derniers jours
        df = pd.DataFrame({"Close": nasdaq_close, "VIX": vix_series})
        df = df[df.index >= today - timedelta(days=60)]
        df = df.dropna(subset=["Close"]).sort_index()

        # Rendements
        df["returns"]     = df["Close"].pct_change()
        df["returns_pct"] = df["returns"] * 100

        # Volatilite 20j annualisee (coherente avec 01_market_data.py)
        df["volatility_20d"] = df["returns"].rolling(20).std() * np.sqrt(252)

        # VIX forward-fill
        df["vix"] = df["VIX"].ffill()

        # RSI 14j (methode EWM comme dans 01_market_data.py)
        delta    = df["Close"].diff()
        gain     = delta.clip(lower=0)
        loss     = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=13, min_periods=14).mean()
        avg_loss = loss.ewm(com=13, min_periods=14).mean()
        rs       = avg_gain / avg_loss
        df["rsi_14"] = 100 - (100 / (1 + rs))

        # MACD
        ema12          = df["Close"].ewm(span=12, adjust=False).mean()
        ema26          = df["Close"].ewm(span=26, adjust=False).mean()
        df["macd"]        = ema12 - ema26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"]   = df["macd"] - df["macd_signal"]

        # Drawdown
        roll_max          = df["Close"].cummax()
        df["drawdown"]    = (df["Close"] - roll_max) / roll_max
        df["drawdown_pct"]= df["drawdown"] * 100

        last = df.dropna().iloc[-1]
        features = {col: float(last[col]) for col in MARKET_FEATURES if col in last.index}
        log.info(f"  Dernier jour : {last.name.date()} | "
                 f"returns={features.get('returns_pct', 0):.2f}% | "
                 f"VIX={features.get('vix', 0):.1f}")
        return features

    except Exception as e:
        log.warning(f"Erreur telechargement marche : {e}")
        return {}


# ══════════════════════════════════════════════════════════════════════════════
# 5. Aggregation des features textuelles
# ══════════════════════════════════════════════════════════════════════════════

def aggregate_text_features(df: pd.DataFrame,
                              emb_live: np.ndarray) -> dict:
    """Calcule les features agregees sur la fenetre J-5/J."""
    feats = {}

    # Sentiment
    feats["sentiment_mean"] = float(df["sentiment"].mean())
    feats["sentiment_std"]  = float(df["sentiment"].std())
    feats["sentiment_min"]  = float(df["sentiment"].min())
    feats["sentiment_max"]  = float(df["sentiment"].max())
    feats["news_count"]     = len(df)
    feats["pct_positive"]   = float((df["sentiment_label"] == "positive").mean())
    feats["pct_negative"]   = float((df["sentiment_label"] == "negative").mean())

    # Anomalie de sentiment (z-score simple sur la fenetre)
    feats["sentiment_anomaly"] = 0.0  # pas assez de points sur 5j

    # Information shift : distance cosinus entre J-5/J-3 et J-2/J
    mid = len(df) // 2
    if mid > 0 and len(df) - mid > 0:
        from scipy.spatial.distance import cosine
        emb_before = emb_live[:mid].mean(axis=0)
        emb_after  = emb_live[mid:].mean(axis=0)
        feats["information_shift"] = float(cosine(emb_before, emb_after))
    else:
        feats["information_shift"] = 0.0

    # Embedding moyen (384 dims)
    emb_mean = emb_live.mean(axis=0)
    for i, v in enumerate(emb_mean):
        feats[f"emb_{i}"] = float(v)

    return feats


# ══════════════════════════════════════════════════════════════════════════════
# 6. Construction du vecteur de features final
# ══════════════════════════════════════════════════════════════════════════════

def build_feature_vector(market: dict, text: dict, feature_cols: list) -> tuple:
    """
    Assemble le vecteur dans le meme ordre que lors de l'entrainement.
    Utilise les feature_cols extraites du dataset d'entrainement.
    Retourne (X, feature_dict).
    """
    merged = {**market, **text}
    feature_dict = {col: merged.get(col, 0.0) for col in feature_cols}
    X = np.array([[feature_dict[col] for col in feature_cols]])

    return X, feature_dict


# ══════════════════════════════════════════════════════════════════════════════
# 7. Prediction et interpretation
# ══════════════════════════════════════════════════════════════════════════════

def predict(model, X: np.ndarray) -> tuple:
    """Retourne (probabilite changement, regime_probable)."""
    proba = model.predict_proba(X)[0, 1]

    # Determiner le regime probable a partir des features de marche
    # (heuristique basee sur les patterns du dataset d'entrainement)
    return proba


def interpret_regime(market: dict, proba: float) -> str:
    """
    Deduit le regime probable a partir du contexte de marche actuel.
    Bull/bear/strong sont determines par la combinaison rendement + VIX + drawdown.
    """
    ret   = market.get("returns_pct", 0)
    vix   = market.get("vix", 15)
    dd    = market.get("drawdown_pct", 0)
    vol   = market.get("volatility_20d", 0.01)
    rsi   = market.get("rsi_14", 50)

    if proba < PROBA_THRESHOLD_MEDIUM:
        return "stable (pas de changement prevu)"

    # Determiner la direction
    if ret < -2 or (dd < -10 and vix > 25):
        base = "bear_strong"
    elif ret < -0.5 or (dd < -3 and vix > 20):
        base = "bear"
    elif ret > 2 and vix < 18 and rsi > 55:
        base = "bull_strong"
    else:
        base = "bull"

    return base


def alert_level(proba: float) -> str:
    if proba >= PROBA_THRESHOLD_HIGH:
        return "FORTE"
    elif proba >= PROBA_THRESHOLD_MEDIUM:
        return "MODEREE"
    else:
        return "FAIBLE"


# ══════════════════════════════════════════════════════════════════════════════
# 8. Construction du rapport
# ══════════════════════════════════════════════════════════════════════════════

def build_report(today: datetime, df_news: pd.DataFrame,
                 market: dict, text_feats: dict,
                 proba: float, regime_pred: str, topic_names: dict) -> dict:

    top_news = (df_news.sort_values("sentiment_abs", ascending=False)
                .head(5)[["date","headline","sentiment","sentiment_label","source","topic"]]
                .to_dict("records"))
    
    log.info(f"  Top news count: {len(top_news)}")
    for idx, n in enumerate(top_news):
        headline_preview = n.get("headline", "N/A")[:50] if n.get("headline") else "[EMPTY]"
        log.info(f"    {idx+1}. {headline_preview}")
    
    for n in top_news:
        n["date"] = str(n["date"])
        n["sentiment"] = round(float(n["sentiment"]), 4)
        n["topic"] = int(n["topic"]) if pd.notna(n.get("topic")) else None
        n["topic_name"] = topic_names.get(n["topic"], f"Topic {n['topic']}")

    topic_dist = {}
    if "topic" in df_news.columns:
        counts = df_news["topic"].value_counts().head(3)
        total = len(df_news)
        for tid, cnt in counts.items():
            topic_id = int(tid)
            topic_name = topic_names.get(topic_id, f"Topic {topic_id}")
            topics_dict_key = f"{topic_name} (t{topic_id})"
            topic_dist[topics_dict_key] = {
                "count": int(cnt),
                "pct": round(cnt / total * 100, 1),
                "topic_id": topic_id,
                "topic_name": topic_name,
            }

    return {
        "prediction_date":   today.strftime("%Y-%m-%d"),
        "run_at":            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "window":            f"J-{WINDOW_DAYS} a J",
        "articles_analyses": len(df_news),
        "prediction": {
            "probabilite_changement": round(float(proba), 4),
            "alerte":                 alert_level(proba),
            "regime_probable":        regime_pred,
            "seuil_forte":            PROBA_THRESHOLD_HIGH,
            "seuil_moderee":          PROBA_THRESHOLD_MEDIUM,
        },
        "contexte_marche": {k: round(v, 4) for k, v in market.items()},
        "features_textuelles": {
            "sentiment_mean":     round(text_feats.get("sentiment_mean", 0), 4),
            "sentiment_std":      round(text_feats.get("sentiment_std", 0), 4),
            "sentiment_min":      round(text_feats.get("sentiment_min", 0), 4),
            "pct_negative":       round(text_feats.get("pct_negative", 0), 4),
            "news_count":         int(text_feats.get("news_count", 0)),
            "information_shift":  round(text_feats.get("information_shift", 0), 4),
        },
        "top_news":          top_news,
        "topic_distribution": topic_dist,
        "topic_names_mapping": topic_names,
    }

def format_txt(report: dict) -> str:
    sep = "=" * 65
    pred = report["prediction"]
    mkt  = report["contexte_marche"]
    txt  = report["features_textuelles"]

    alerte_marker = {
        "FORTE":   "!!! ALERTE FORTE !!!",
        "MODEREE": ">>  Alerte moderee",
        "FAIBLE":  "    Signal faible",
    }[pred["alerte"]]

    lines = [
        sep,
        f"PREDICTION LIVE — {report['prediction_date']}",
        f"Genere le {report['run_at']}",
        sep,
        "",
        f"  {alerte_marker}",
        f"  Probabilite de changement : {pred['probabilite_changement']:.1%}",
        f"  Regime probable            : {pred['regime_probable']}",
        f"  Seuils : forte={pred['seuil_forte']:.0%}  moderee={pred['seuil_moderee']:.0%}",
        "",
        "-" * 65,
        "CONTEXTE DE MARCHE",
        "-" * 65,
        f"  Rendement (dernier j) : {mkt.get('returns_pct', 'N/A')}%",
        f"  Volatilite 20j        : {mkt.get('volatility_20d', 'N/A')}",
        f"  VIX                   : {mkt.get('vix', 'N/A')}",
        f"  RSI 14j               : {mkt.get('rsi_14', 'N/A')}",
        f"  Drawdown              : {mkt.get('drawdown_pct', 'N/A')}%",
        "",
        "-" * 65,
        f"FEATURES TEXTUELLES ({txt['news_count']} articles sur J-{WINDOW_DAYS}/J)",
        "-" * 65,
        f"  Sentiment moyen    : {txt['sentiment_mean']}",
        f"  Sentiment std      : {txt['sentiment_std']}",
        f"  Sentiment min      : {txt['sentiment_min']}",
        f"  % negatifs         : {txt['pct_negative']:.1%}",
        f"  Information shift  : {txt['information_shift']}",
        "",
        "-" * 65,
        "TOP 5 NEWS LES PLUS SIGNIFICATIVES",
        "-" * 65,
    ]

    for i, n in enumerate(report["top_news"], 1):
        lbl = n["sentiment_label"].upper()
        topic_name = n.get("topic_name", f"Topic {n['topic']}")
        headline = n.get("headline", "[HEADLINE MISSING]").strip()
        
        # Debug: vérifier si headline est vide
        if not headline:
            log.warning(f"  Article {i} a headline vide: {n}")
            headline = "[HEADLINE VIDE]"
        
        lines += [
            f"{i}. [{n['date']}] [{lbl} {n['sentiment']:+.2f}] {topic_name}",
            f"   {headline}",
            f"   Source : {n['source']}",
            "",
        ]

    lines += [
        "-" * 65,
        "TOPICS DOMINANTS",
        "-" * 65,
    ]
    for topic, info in report["topic_distribution"].items():
        bar = "#" * int(info["pct"] / 100 * 30)
        lines.append(f"  {topic:35s} : {bar} {info['pct']:5.1f}% ({info['count']} articles)")

    lines += ["", sep]
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# 9. Point d'entree
# ══════════════════════════════════════════════════════════════════════════════

def main():
    today = datetime.today()
    date_str = today.strftime("%Y%m%d")

    log.info("=" * 55)
    log.info(f"PREDICTION LIVE — {today.strftime('%Y-%m-%d')}")
    log.info("=" * 55)

    # Charger le modele
    log.info("Chargement du modele ...")
    model = joblib.load(MODEL_PATH)

    # Charger les noms des features depuis le dataset d'entrainement
    log.info("Chargement des noms de features ...")
    df_train_sample = pd.read_csv(DATASET_PATH, nrows=1)
    # Exclure les colonnes qui ne sont pas des features (métadonnées, cibles, duplicates de merge)
    exclude_cols = {
        'date', 'close', 'regime', 'regime_id', 'regime_change', 'y',
        'news_count_x', 'news_count_y',  # Duplicates du merge
        'topic_dominant', 'topic_diversity',  # Colonnes de méta-analyse non dans le modèle
    }
    feature_cols = [c for c in df_train_sample.columns if c not in exclude_cols]
    log.info(f"  {len(feature_cols)} features identifiees")

    # Charger les embeddings historiques pour reutiliser PCA/KMeans
    log.info("Chargement historique embeddings ...")
    df_hist = pd.read_csv(NEWS_HIST_PATH)
    log.info(f"  {len(df_hist)} articles historiques charges")

    # Collecte news actuelles
    df_news = collect_news(today)
    if df_news.empty:
        log.error("Aucune news collectee. Verifier les cles API.")
        return

    # Sentiment FinBERT
    df_news = compute_sentiment(df_news)

    # Embeddings + topics
    df_news, emb_live, pca, kmeans = compute_embeddings_and_topics(df_news, df_hist)

    # Noms descriptifs pour les topics
    topic_names = generate_topic_names(df_hist)

    # Features marche actuelles
    market_feats = get_market_features(today)
    if not market_feats:
        log.warning("Features marche indisponibles — zeros utilises")
        market_feats = {col: 0.0 for col in MARKET_FEATURES}

    # Aggregation features textuelles
    text_feats = aggregate_text_features(df_news, emb_live)

    # Construction vecteur de features
    X, feature_dict = build_feature_vector(market_feats, text_feats, feature_cols)
    log.info(f"Vecteur de features construit : {X.shape}")

    # Prediction
    log.info("Prediction ...")
    log.info(f"Features construites : {len(feature_cols)}")
    log.info(f"N features modele : {model.n_features_in_}")
    proba = predict(model, X)
    regime_pred = interpret_regime(market_feats, proba)
    alerte = alert_level(proba)

    log.info(f"\n{'='*55}")
    log.info(f"RESULTAT : probabilite = {proba:.1%}  |  alerte = {alerte}")
    log.info(f"Regime probable : {regime_pred}")
    log.info(f"{'='*55}\n")

    # Rapport
    report = build_report(today, df_news, market_feats, text_feats, proba, regime_pred, topic_names)

    # Sauvegarde JSON
    out_json = OUT_DIR / f"prediction_{date_str}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    log.info(f"OK JSON -> {out_json}")

    # Sauvegarde TXT
    out_txt = OUT_DIR / f"prediction_{date_str}.txt"
    out_txt.write_text(format_txt(report), encoding="utf-8")
    log.info(f"OK TXT  -> {out_txt}")

    # Affichage terminal
    print("\n" + format_txt(report))


if __name__ == "__main__":
    main()