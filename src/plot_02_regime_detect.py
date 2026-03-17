import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# --- Charger les données ---

# Regimes
regimes = pd.read_csv("data\\market\\regime_output.csv", sep=",", parse_dates=["date"])
regimes['date'] = pd.to_datetime(regimes['date'])

# Nasdaq
nasdaq = pd.read_csv("data\\market\\nasdaq_10y.csv", sep=";", decimal=",", parse_dates=["Date"])
nasdaq.rename(columns={"Date": "date", "Prix de clôture": "close"}, inplace=True)
nasdaq['date'] = pd.to_datetime(nasdaq['date'], dayfirst=True)

# --- Préparer les couleurs pour les 5 régimes ---
colors = {
    "bear_strong": "red",
    "bear": "lightcoral",
    "sideways": "blue",
    "bull": "lightgreen",
    "bull_strong": "green"
}

# --- Graphique ---
fig, ax = plt.subplots(figsize=(16,6))

# Tracer le prix
ax.plot(nasdaq['date'], nasdaq['close'], color='black', label='Prix NASDAQ')

# Ajouter les régimes en arrière-plan
for idx, row in regimes.iterrows():
    start_date = row['date']
    regime_color = colors.get(row['regime'], 'grey')
    
    # Trouver la date de fin du régime
    next_idx = idx + 1
    if next_idx < len(regimes):
        end_date = regimes.iloc[next_idx]['date']
    else:
        end_date = nasdaq['date'].iloc[-1]  # dernier jour dispo
    
    ax.axvspan(start_date, end_date, color=regime_color, alpha=0.2)

# Formater l'axe x
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_minor_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.xticks(rotation=45)

# Légende
handles = [plt.Rectangle((0,0),1,1, color=colors[r], alpha=0.2) for r in colors]
labels = [r.replace("_", " ").capitalize() for r in colors]
ax.legend(handles=handles + [plt.Line2D([0],[0], color='black')], labels=labels + ['Prix NASDAQ'])

ax.set_title("Régimes de marché vs Prix NASDAQ")
ax.set_ylabel("Prix de clôture")
ax.set_xlabel("Date")
plt.tight_layout()
plt.show()