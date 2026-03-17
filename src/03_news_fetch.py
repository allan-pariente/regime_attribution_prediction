"""
03_news_fetch.py — Étape 3 : Récupération des actualités financières
Pipeline : News-Driven Regime Attribution Engine

Entrée  : CSV de données de marché (avec colonnes Date, Rendement journalier, etc.)
          + fichier de régimes (ou détection intégrée par seuil de volatilité/drawdown)
Sortie  : news_raw.json      — toutes les news brutes récupérées
          news_df.csv        — dataset structuré prêt pour FinBERT (étape 4)
          fetch_report.txt   — rapport de collecte (couverture, erreurs, stats)

Sources supportées :
  - Finnhub (principal)
  - Google News RSS  (fallback automatique, sans clé, historique complet)
"""

import os
import json
import time
import logging
import requests
import pandas as pd
from datetime import timedelta
from pathlib import Path

# ─── Configuration ────────────────────────────────────────────────────────────

FINNHUB_API_KEY      = os.getenv("FINNHUB_API_KEY", "d6s4gnhr01qrb5i8jpogd6s4gnhr01qrb5i8jpp0")
GUARDIAN_API_KEY     = os.getenv("GUARDIAN_API_KEY", "30879444-61c6-4d54-8c19-7b12a6acb989")  # https://bonobo.capi.gutools.co.uk/register/developer

DATA_DIR             = Path("data")
MARKET_CSV           = DATA_DIR / "market" / "nasdaq_10y.csv"
REGIME_CSV           = DATA_DIR / "processed" / "regime_output.csv"
NEWS_JSON            = DATA_DIR / "news" / "news_raw.json"
NEWS_CSV             = DATA_DIR / "processed" / "news_df.csv"
REPORT_TXT           = DATA_DIR / "news" / "fetch_report.txt"

WINDOW_BEFORE_DAYS   = 5
WINDOW_AFTER_DAYS    = 0   # IMPORTANT : 0 pour éviter le look-ahead bias
MIN_REGIME_DURATION_DAYS = 5
MIN_REGIME_MAGNITUDE = 0.02
COMPANY_TICKERS      = ["QQQ", "AAPL", "MSFT", "NVDA", "AMZN"]
FINNHUB_CATEGORIES   = ["general", "technology"]
RATE_LIMIT_DELAY     = 1.2

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── Helpers ──────────────────────────────────────────────────────────────────

def safe_get(url, params, label=""):
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        return data if isinstance(data, list) else []
    except requests.exceptions.HTTPError as e:
        log.error(f"[{label}] HTTP {e.response.status_code}")
        return []
    except Exception as e:
        log.error(f"[{label}] Erreur : {e}")
        return []


# ─── Chargement du CSV de marché ─────────────────────────────────────────────

def _detect_encoding_and_sep(csv_path):
    """Détecte l'encodage et le séparateur du CSV."""
    for enc in ("utf-8-sig", "utf-8", "latin-1", "cp1252"):
        try:
            with open(csv_path, encoding=enc, errors="strict") as f:
                sample = f.read(4096)
            sep = ";" if sample.count(";") > sample.count(",") else (
                  "\t" if sample.count("\t") > sample.count(",") else ",")
            return enc, sep
        except Exception:
            continue
    return "latin-1", ","


def load_market_data(csv_path):
    """
    Charge le CSV de marché de façon robuste.
    Gère : séparateur virgule ou point-virgule, encodages FR (latin-1, utf-8-sig),
    décimales en virgule (format européen), espaces insécables, colonne Volume=None.
    """
    log.info(f"Chargement données de marché : {csv_path}")

    enc, sep = _detect_encoding_and_sep(csv_path)
    log.info(f"  Encodage : {enc} | Séparateur : {repr(sep)}")

    if sep == ";":
        # pandas gère decimal="," nativement quand sep != ","
        df = pd.read_csv(csv_path, sep=sep, encoding=enc,
                         on_bad_lines="skip", decimal=",", thousands=" ")
    else:
        # Lecture brute en str pour conversion manuelle des décimales FR
        df = pd.read_csv(csv_path, sep=sep, encoding=enc,
                         on_bad_lines="skip", dtype=str)
        for col in df.columns:
            if col.strip().lower() in ("date", "datetime", "time", "timestamp"):
                continue
            cleaned = (df[col].astype(str)
                               .str.replace("\xa0", "", regex=False)
                               .str.replace("\u202f", "", regex=False)
                               .str.replace(" ", "", regex=False)
                               .str.replace(",", ".", regex=False))
            numeric = pd.to_numeric(cleaned, errors="coerce")
            if numeric.notna().mean() > 0.7:
                df[col] = numeric

    log.info(f"  Colonnes détectées : {df.columns.tolist()}")

    # Détection flexible de la colonne date
    date_col = next(
        (c for c in df.columns if c.strip().lower() in ("date", "datetime", "time", "timestamp")),
        None,
    )
    if date_col is None:
        date_col = df.columns[0]
        log.warning(f"  Colonne date non trouvée par nom — utilisation de '{date_col}'")
    else:
        log.info(f"  Colonne date : '{date_col}'")

    df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
    df = df.rename(columns={date_col: "date"})

    n_bad = df["date"].isna().sum()
    if n_bad:
        log.warning(f"  {n_bad} lignes avec date non parseable — supprimées.")
        df = df.dropna(subset=["date"])

    df = df.sort_values("date").reset_index(drop=True)

    # Supprimer Volume si entièrement None
    for vc in [c for c in df.columns if c.strip().lower() == "volume"]:
        if df[vc].isna().all():
            df = df.drop(columns=[vc])
            log.info(f"  Colonne '{vc}' ignorée (entièrement None).")

    log.info(f"  → {len(df)} jours ({df['date'].min().date()} → {df['date'].max().date()})")
    return df


# ─── Détection des changements de régime ─────────────────────────────────────

def load_or_detect_regime_changes(df):
    """
    Charge depuis regimes.csv ou détecte par seuil vol/drawdown.
    Retourne une liste de pd.Timestamp.
    """
    if REGIME_CSV.exists():
        log.info(f"Chargement des régimes depuis {REGIME_CSV}")
        regime_df = pd.read_csv(REGIME_CSV, parse_dates=["date"])
        regime_df = regime_df.sort_values("date").reset_index(drop=True)
        # Lire la colonne regime_change si elle existe (True/False)
        # sinon la recalculer depuis regime_id
        if "regime_change" in regime_df.columns:
            regime_df["regime_change"] = regime_df["regime_change"].astype(str).str.lower() == "true"
        else:
            regime_df["regime_change"] = regime_df["regime_id"].diff().ne(0)
        change_dates = regime_df.loc[regime_df["regime_change"], "date"].tolist()
        log.info(f"  → {len(change_dates)} changements de régime chargés.")
        return change_dates

    log.info("Fichier régimes absent — détection par seuil (volatilité + drawdown).")

    vol_col = next((c for c in df.columns if "volatil" in c.lower()), None)
    dd_col  = next((c for c in df.columns if "drawdown" in c.lower() and "%" in c), None)
    px_col  = next((c for c in df.columns if any(k in c.lower()
                    for k in ("clôture", "cloture", "close", "prix"))), None)

    if vol_col is None:
        raise ValueError(
            f"Colonne de volatilité introuvable. Colonnes disponibles : {df.columns.tolist()}"
        )

    df = df.copy()
    vol_z = (df[vol_col] - df[vol_col].rolling(60).mean()) / df[vol_col].rolling(60).std()
    df["stress_score"] = vol_z.fillna(0)
    if dd_col:
        dd_z = (df[dd_col].abs() - df[dd_col].abs().rolling(60).mean()) / df[dd_col].abs().rolling(60).std()
        df["stress_score"] += dd_z.fillna(0)

    df["regime_stress"] = (df["stress_score"].abs() > 1.5).astype(int)
    df["regime_change"] = df["regime_stress"].diff().ne(0) & (df["regime_stress"] == 1)

    change_dates = []
    candidates = df.loc[df["regime_change"], "date"].tolist()

    for i, d in enumerate(candidates):
        next_d = candidates[i + 1] if i + 1 < len(candidates) else df["date"].iloc[-1]
        duration = (next_d - d).days
        if px_col:
            window = df[(df["date"] >= d) & (df["date"] <= next_d)]
            magnitude = (abs(window[px_col].iloc[-1] - window[px_col].iloc[0]) / window[px_col].iloc[0]
                         if len(window) > 1 else 0)
        else:
            magnitude = MIN_REGIME_MAGNITUDE

        if duration >= MIN_REGIME_DURATION_DAYS and magnitude >= MIN_REGIME_MAGNITUDE:
            change_dates.append(d)

    log.info(f"  → {len(change_dates)} changements de régime détectés.")
    return change_dates


# ─── Récupération Finnhub ─────────────────────────────────────────────────────

def fetch_finnhub_general(from_date, to_date, category="general"):
    params = {"category": category, "token": FINNHUB_API_KEY, "from": from_date, "to": to_date}
    return safe_get("https://finnhub.io/api/v1/news", params, label=f"Finnhub/{category}")


def fetch_finnhub_company(symbol, from_date, to_date):
    params = {"symbol": symbol, "from": from_date, "to": to_date, "token": FINNHUB_API_KEY}
    return safe_get("https://finnhub.io/api/v1/company-news", params, label=f"Finnhub/{symbol}")


# ─── Récupération Google News RSS ────────────────────────────────────────────
#
# Pas de clé API, pas de limite de taux stricte, couvre tout l'historique.
# Google News RSS accepte les paramètres `before:` et `after:` dans la query
# pour filtrer par date — exemple : "NASDAQ after:2020-03-01 before:2020-03-10"
#
# Dépendance : pip install feedparser

try:
    import feedparser
    _feedparser_ok = True
except ImportError:
    _feedparser_ok = False
    log.warning("feedparser non installé — pip install feedparser")

# Queries Guardian — couvre tout l'historique depuis 2000, gratuitement
GUARDIAN_QUERIES = [
    "NASDAQ stock market",
    "Federal Reserve interest rates",
    "S&P 500 Wall Street",
    "inflation CPI economy",
    "tech stocks earnings",
    "stock market crash correction",
    "Wall Street volatility",
    "Fed monetary policy",
    "China trade war tariffs",
    "oil price market",
]

GNEWS_BASE = "https://news.google.com/rss/search"  # conservé en fallback


def _gnews_rss(query: str, from_date: str, to_date: str) -> list:
    """
    Appelle Google News RSS pour une query + fenêtre temporelle.
    Retourne une liste d'articles normalisés.
    from_date / to_date : format YYYY-MM-DD
    """
    if not _feedparser_ok:
        return []

    # Google News supporte after: et before: dans la query
    q = f"{query} after:{from_date} before:{to_date}"
    params = {"q": q, "hl": "en-US", "gl": "US", "ceid": "US:en"}
    url = GNEWS_BASE + "?" + "&".join(f"{k}={requests.utils.quote(str(v))}" for k, v in params.items())

    try:
        import calendar
        f_dt = pd.Timestamp(from_date)
        t_dt = pd.Timestamp(to_date) + pd.Timedelta(hours=21, minutes=30)

        feed = feedparser.parse(url)
        n_raw = len(feed.entries)
        articles = []
        for entry in feed.entries:
            # Parser le timestamp — si absent ou non parseable : article rejeté
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                ts = calendar.timegm(entry.published_parsed)
            else:
                # Pas de date = on ne peut pas vérifier la fenêtre → rejeté
                log.debug(f"  [RSS] Article sans date ignoré : {entry.get('title','')[:60]}")
                continue

            # Filtre strict : article doit être dans [from_date, to_date 21h30 UTC]
            # ts=0 est aussi rejeté (epoch = date invalide)
            art_date = pd.Timestamp(ts, unit="s")
            if not (ts > 0 and f_dt <= art_date <= t_dt):
                continue

            articles.append({
                "headline": entry.get("title", "").strip(),
                "summary":  entry.get("summary", "").strip(),
                "source":   entry.get("source", {}).get("title", "google_news")
                            if isinstance(entry.get("source"), dict) else "google_news",
                "url":      entry.get("link", ""),
                "datetime": ts,
                "category": "general",
                "_source_api": "google_news_rss",
            })
        log.debug(f"  [RSS] '{query[:40]}' | brut={n_raw} → après filtre date={len(articles)}")
        return articles

    except Exception as e:
        log.warning(f"[GoogleNews RSS] query={query!r} erreur : {e}")
        return []


def fetch_guardian(from_date: str, to_date: str) -> list:
    """
    The Guardian Content API — historique complet depuis 2000, gratuit.
    Clé gratuite sur : https://bonobo.capi.gutools.co.uk/register/developer
    Docs : https://open-platform.theguardian.com/documentation/
    """
    print(f"[DEBUG] GUARDIAN_API_KEY = '{GUARDIAN_API_KEY[:10]}...' | longueur={len(GUARDIAN_API_KEY)}")
    print(f"[DEBUG] Est-ce la clé par défaut ? {GUARDIAN_API_KEY == 'votre_cle_guardian'}")
    if GUARDIAN_API_KEY == "votre_cle_guardian":
        print("[DEBUG] Clé par défaut détectée → Guardian ignoré")
        return []
    print(f"[DEBUG] Clé valide → appel Guardian pour {from_date} → {to_date}")

    all_articles, seen = [], set()
    for query in GUARDIAN_QUERIES:
        try:
            params = {
                "q": query,
                "from-date": from_date,
                "to-date": to_date,
                "order-by": "relevance",
                "show-fields": "headline,trailText",
                "page-size": 50,
                "api-key": GUARDIAN_API_KEY,
                "section": "business|technology|money|us-news",
            }
            r = requests.get(
                "https://content.guardianapis.com/search",
                params=params,
                timeout=10,
            )
            r.raise_for_status()
            results = r.json().get("response", {}).get("results", [])

            for a in results:
                fields   = a.get("fields", {})
                headline = fields.get("headline") or a.get("webTitle", "")
                title_key = "".join(c.lower() for c in headline if c.isalnum())[:80]
                if not title_key or title_key in seen:
                    continue
                seen.add(title_key)

                # Parser la date de publication
                pub = a.get("webPublicationDate", "")  # format ISO : 2020-03-09T14:32:00Z
                try:
                    ts = int(pd.Timestamp(pub).timestamp())
                except Exception:
                    ts = 0

                all_articles.append({
                    "headline":    headline,
                    "summary":     fields.get("trailText", ""),
                    "source":      "the_guardian",
                    "url":         a.get("webUrl", ""),
                    "datetime":    ts,
                    "category":    a.get("sectionId", "business"),
                    "_source_api": "guardian",
                })
            time.sleep(0.3)

        except Exception as e:
            log.warning(f"[Guardian] query={query!r} erreur : {e}")

    log.info(f"  Guardian : {len(all_articles)} articles ({from_date} → {to_date})")
    return all_articles


def fetch_google_news(from_date: str, to_date: str) -> list:
    """
    Google News RSS — fallback si Guardian non configuré.
    Fonctionne bien sur les 12 derniers mois, moins fiable sur l'historique long.
    """
    all_articles, seen = [], set()
    for query in GUARDIAN_QUERIES[:5]:   # 5 queries suffisent en fallback
        arts = _gnews_rss(query, from_date, to_date)
        for a in arts:
            title_key = "".join(c.lower() for c in a.get("headline","") if c.isalnum())[:80]
            if not title_key or title_key in seen:
                continue
            seen.add(title_key)
            all_articles.append(a)
        time.sleep(0.5)
    return all_articles


# ─── Collecte principale ─────────────────────────────────────────────────────

def collect_news_for_changes(change_dates):
    all_news, seen_keys = [], set()
    total = len(change_dates)
    log.info(f"Collecte des news pour {total} changements de régime (J-{WINDOW_BEFORE_DAYS} → J-1, articles AVANT le changement uniquement)")

    for idx, change_date in enumerate(change_dates, 1):
        t       = pd.Timestamp(change_date)
        from_dt = (t - timedelta(days=WINDOW_BEFORE_DAYS)).strftime("%Y-%m-%d")
        to_dt   = t.strftime("%Y-%m-%d")   # borne exclusive : jour J non inclus
        log.info(f"[{idx:>3}/{total}] {t.date()} | fenêtre {from_dt} → {to_dt}")

        batch = []
        f_dt = pd.Timestamp(from_dt)
        t_dt = pd.Timestamp(to_dt) + pd.Timedelta(hours=21, minutes=30)

        def filter_by_date(arts):
            """Rejette les articles hors fenêtre temporelle (Finnhub ignore souvent les dates)."""
            kept = []
            for a in arts:
                ts = a.get("datetime", 0)
                try:
                    ts = int(ts)
                except (TypeError, ValueError):
                    ts = 0
                if ts == 0:
                    continue  # pas de date = on rejette
                art_date = pd.Timestamp(ts, unit="s")
                if f_dt <= art_date <= t_dt:
                    kept.append(a)
            return kept

        for cat in FINNHUB_CATEGORIES:
            arts = fetch_finnhub_general(from_dt, to_dt, cat)
            arts = filter_by_date(arts)
            for a in arts:
                a.setdefault("_source_api", "finnhub"); a["_category_query"] = cat
            batch.extend(arts)
            time.sleep(RATE_LIMIT_DELAY)

        for ticker in COMPANY_TICKERS:
            arts = fetch_finnhub_company(ticker, from_dt, to_dt)
            arts = filter_by_date(arts)
            for a in arts:
                a.setdefault("_source_api", "finnhub"); a["_ticker_query"] = ticker
            batch.extend(arts)
            time.sleep(RATE_LIMIT_DELAY)

        if len(batch) < 10:
            # Guardian en priorité (historique complet 2000-aujourd'hui)
            guardian_arts = fetch_guardian(from_dt, to_dt)
            if guardian_arts:
                log.info(f"  Guardian : +{len(guardian_arts)} articles")
                batch.extend(guardian_arts)
            elif GUARDIAN_API_KEY == "votre_cle_guardian":
                # Pas de clé Guardian configurée — on n'utilise PAS Google News
                # (retourne des articles récents hors fenêtre, inutilisables)
                log.warning("  ⚠️  Aucune clé Guardian configurée — fenêtre ignorée.")
                log.warning("      → Récupère une clé gratuite : https://bonobo.capi.gutools.co.uk/register/developer")
            else:
                # Clé Guardian valide mais 0 résultats → fallback Google News (récent seulement)
                log.info("  Guardian vide → fallback Google News RSS")
                batch.extend(fetch_google_news(from_dt, to_dt))

        n_before = len(all_news)
        for article in batch:
            # Dédoublonnage par titre normalisé (insensible à la casse/ponctuation)
            # Les URLs Google News sont des redirections uniques — non fiables pour dédup
            raw_title = article.get("headline", "") or article.get("title", "")
            title_key = "".join(c.lower() for c in raw_title if c.isalnum())[:80]
            if not title_key:
                continue
            if title_key in seen_keys:
                continue
            seen_keys.add(title_key)
            article["regime_change_date"] = str(t.date())
            all_news.append(article)

        log.info(f"  → +{len(all_news) - n_before} articles (total : {len(all_news)})")

    return all_news


# ─── Construction du DataFrame structuré ─────────────────────────────────────

def build_news_dataframe(all_news):
    if not all_news:
        log.warning("Aucune news collectée.")
        return pd.DataFrame()

    df = pd.DataFrame(all_news)
    keep = ["headline", "summary", "source", "url", "datetime",
            "category", "regime_change_date", "_source_api",
            "_category_query", "_ticker_query"]
    for col in keep:
        if col not in df.columns:
            df[col] = None
    df = df[keep].copy()

    df["datetime"] = pd.to_numeric(df["datetime"], errors="coerce")
    df["datetime"] = pd.to_datetime(df["datetime"], unit="s", errors="coerce", utc=True)
    df["datetime"] = df["datetime"].dt.tz_convert("Europe/Paris").dt.tz_localize(None)
    df["headline"] = df["headline"].fillna("").str.strip()
    df["summary"]  = df["summary"].fillna("").str.strip()
    df = df[df["headline"] != ""]
    df["date"] = df["datetime"].dt.date
    df = df.sort_values("datetime").reset_index(drop=True)

    log.info(f"DataFrame : {len(df)} articles, {df['regime_change_date'].nunique()} régimes couverts.")
    return df


# ─── Rapport de collecte ─────────────────────────────────────────────────────

def write_report(df, change_dates, output_path):
    api_counts = df["_source_api"].value_counts().to_dict() if not df.empty else {}
    lines = [
        "=" * 60,
        "RAPPORT DE COLLECTE — 03_news_fetch.py",
        "=" * 60,
        f"Changements de régime ciblés  : {len(change_dates)}",
        f"Articles collectés (total)     : {len(df)}",
        f"Sources API                    : {api_counts}",
        "", "Couverture par changement de régime :", "-" * 40,
    ]
    for d in change_dates:
        d_str = str(pd.Timestamp(d).date())
        count = len(df[df["regime_change_date"] == d_str]) if not df.empty else 0
        icon  = "✅" if count >= 5 else ("⚠️ " if count > 0 else "❌")
        lines.append(f"  {icon} {d_str} : {count} articles")

    poor = [str(pd.Timestamp(d).date()) for d in change_dates
            if (len(df[df["regime_change_date"] == str(pd.Timestamp(d).date())]) if not df.empty else 0) < 5]
    if poor:
        lines += ["", f"⚠️  {len(poor)} régimes avec couverture insuffisante (<5 articles) :",
                  "   → Élargir WINDOW_BEFORE_DAYS ou ajouter des queries dans GNEWS_QUERIES.",
                  f"   Dates : {', '.join(poor[:10])}{'...' if len(poor) > 10 else ''}"]
    lines += ["", "Conseil : augmenter WINDOW_BEFORE_DAYS si couverture < 80%", "=" * 60]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    for line in lines:
        log.info(line)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    for d in [DATA_DIR / "market", DATA_DIR / "news", DATA_DIR / "processed"]:
        d.mkdir(parents=True, exist_ok=True)

    df_market    = load_market_data(MARKET_CSV)
    change_dates = load_or_detect_regime_changes(df_market)

    if not change_dates:
        log.error("Aucun changement de régime détecté — arrêt.")
        return

    all_news_raw = collect_news_for_changes(change_dates)

    NEWS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(NEWS_JSON, "w", encoding="utf-8") as f:
        json.dump(all_news_raw, f, ensure_ascii=False, indent=2, default=str)
    log.info(f"News brutes → {NEWS_JSON}")

    news_df = build_news_dataframe(all_news_raw)
    if not news_df.empty:
        news_df.to_csv(NEWS_CSV, index=False, encoding="utf-8")
        log.info(f"Dataset structuré → {NEWS_CSV}")

    write_report(news_df, change_dates, REPORT_TXT)
    log.info("Étape 3 terminée. Prochaine étape : 04_sentiment.py (FinBERT)")


if __name__ == "__main__":
    main()