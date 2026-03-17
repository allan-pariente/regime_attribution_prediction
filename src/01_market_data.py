import requests
import pandas as pd
import numpy as np
from io import StringIO
import warnings
warnings.filterwarnings("ignore")

# ─── 1. Nasdaq Composite via FRED ────────────────────────────────────────────
def fetch_fred(series_id):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    r = requests.get(url, verify=False)
    df = pd.read_csv(StringIO(r.text))
    df.columns = ["Date", series_id]
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")
    df[series_id] = pd.to_numeric(df[series_id], errors="coerce")
    return df

nasdaq = fetch_fred("NASDAQCOM")
vix    = fetch_fred("VIXCLS")

# ─── 2. Volume via Yahoo Finance direct ──────────────────────────────────────
def fetch_volume_yahoo():
    url = (
        "https://query1.finance.yahoo.com/v7/finance/download/%5EIXIC"
        "?period1=1388534400&period2=9999999999&interval=1d&events=history"
    )
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers, verify=False)
    df = pd.read_csv(StringIO(r.text))
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")[["Volume"]]
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")
    return df

try:
    volume = fetch_volume_yahoo()
    print(f"✅ Volume récupéré : {len(volume)} jours")
except Exception as e:
    print(f"⚠️ Volume indisponible : {e}")
    volume = None

# ─── 3. Fusion & filtre 10 ans ───────────────────────────────────────────────
df = nasdaq.join(vix, how="left")
df.columns = ["Close", "VIX"]

if volume is not None:
    df = df.join(volume, how="left")
else:
    df["Volume"] = None

df = df.sort_index()
df = df[df.index >= pd.Timestamp.today() - pd.DateOffset(years=10)]
df = df.dropna(subset=["Close"])

# ─── 4. Indicateurs de base ──────────────────────────────────────────────────
df["Daily_Return"]     = df["Close"].pct_change()
df["Daily_Return_Pct"] = df["Daily_Return"] * 100
df["Volatility_20d"]   = df["Daily_Return"].rolling(20).std() * np.sqrt(252)

# ─── 5. RSI (14 jours) ───────────────────────────────────────────────────────
def compute_rsi(series, period=14):
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs       = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

df["RSI_14"] = compute_rsi(df["Close"])

# ─── 6. MACD ─────────────────────────────────────────────────────────────────
ema12             = df["Close"].ewm(span=12, adjust=False).mean()
ema26             = df["Close"].ewm(span=26, adjust=False).mean()
df["MACD"]        = ema12 - ema26
df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
df["MACD_Hist"]   = df["MACD"] - df["MACD_Signal"]

# ─── 7. Drawdown glissant ────────────────────────────────────────────────────
rolling_max        = df["Close"].cummax()
df["Drawdown"]     = (df["Close"] - rolling_max) / rolling_max
df["Drawdown_Pct"] = df["Drawdown"] * 100

# ─── 8. Renommage final ──────────────────────────────────────────────────────
df = df.rename(columns={
    "Close"         : "Prix de clôture",
    "Volume"        : "Volume",
    "Daily_Return"  : "Rendement journalier",
    "Daily_Return_Pct": "Rendement journalier (%)",
    "Volatility_20d": "Volatilité glissante 20j",
    "RSI_14"        : "RSI 14j",
    "MACD"          : "MACD",
    "MACD_Signal"   : "MACD Signal",
    "MACD_Hist"     : "MACD Histogramme",
    "Drawdown"      : "Drawdown",
    "Drawdown_Pct"  : "Drawdown (%)",
    "VIX"           : "VIX",
})

df = df[[
    "Prix de clôture",
    "Volume",
    "Rendement journalier",
    "Rendement journalier (%)",
    "Volatilité glissante 20j",
    "RSI 14j",
    "MACD",
    "MACD Signal",
    "MACD Histogramme",
    "Drawdown",
    "Drawdown (%)",
    "VIX",
]]

# ─── 9. Export CSV ───────────────────────────────────────────────────────────
output_path = "nasdaq_historique_10ans.csv"
df.to_csv(output_path, sep=";", decimal=",", encoding="utf-8-sig")

print(f"✅ Fichier exporté : {output_path}")
print(f"📅 Période         : {df.index.min().date()} → {df.index.max().date()}")
print(f"📊 Nombre de jours : {len(df)}")
print(f"📉 VIX manquant    : {df['VIX'].isna().sum()} jours")
print(f"📦 Volume manquant : {df['Volume'].isna().sum()} jours")
print(df.tail(5).to_string())