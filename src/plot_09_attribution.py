"""
Visualisation 09 - Attribution des changements de régime aux news
Plan : Graphe des prix NASDAQ + régimes + numéros de changement
       + Tableau détaillé avec synthèses professionnelles des news

Ce module combine:
  - Graphe 1 : Prix NASDAQ + régimes + points numérotés pour chaque changement
  - Graphe 2 : Tableau détaillé avec synthèses professionnelles des news
  - Une légende structurée et facile à consulter

La visualisation permet de relier visuellement chaque changement aux news significatives
de manière claire et professionnelle.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import json
from pathlib import Path
import numpy as np
from datetime import datetime

# ── Chemins ────────────────────────────────────────────────────────────────────
NASDAQ_PATH      = Path("data\\market\\nasdaq_10y.csv")
REGIME_PATH      = Path("data\\processed\\regime_output.csv")
ATTRIBUTION_PATH = Path("data\\processed\\attribution_report.json")

# ── Parametres visuels ─────────────────────────────────────────────────────────
REGIME_COLORS = {
    "bear_strong": "#8B0000",      # Dark red
    "bear":        "#FF6B6B",      # Light coral
    "sideways":    "#4169E1",      # Royal blue
    "bull":        "#90EE90",      # Light green
    "bull_strong": "#228B22",      # Forest green
}

# Couleurs du sentiment pour les annotations
SENTIMENT_COLORS = {
    "negative":    "#FF6B6B",      # Négatif = rouge
    "neutral":     "#FFD700",      # Neutre = jaune
    "positive":    "#90EE90",      # Positif = vert
}

# Correspondance topics → étiquettes courtes
TOPIC_LABELS = {
    0: "Tech",
    1: "Energy",
    2: "Economy",
    3: "Earnings",
    4: "Oil/Commodities",
    5: "Politics",
    6: "Finance",
    7: "Employment",
    8: "Real Estate",
    9: "Healthcare",
}


# ══════════════════════════════════════════════════════════════════════════════
# 1. Utilitaires de synthèse des news
# ══════════════════════════════════════════════════════════════════════════════

def extract_keywords(headline: str, max_words: int = 4) -> str:
    """
    Extrait les mots-clés principaux du titre pour créer un résumé court.
    Enlève les mots vides (the, a, is, etc) et garde les principaux mots.
    """
    stop_words = {
        "the", "a", "an", "and", "or", "but", "is", "are", "was", "were",
        "be", "been", "being", "have", "has", "had", "do", "does", "did",
        "will", "would", "could", "should", "may", "might", "cannot", "can",
        "as", "to", "for", "in", "on", "at", "by", "of", "from", "with",
        "into", "up", "out", "if", "than", "so", "no", "not", "yet", "which",
        "who", "what", "when", "where", "why", "how", "all", "each", "every",
        "both", "few", "more", "most", "other", "same", "such", "that", "this",
        "about", "through", "during", "before", "after", "above", "below",
        "between", "under", "over", "along", "against", "down", "off", "across",
    }

    # Nettoyer et tokenizer
    clean_headline = headline.lower()
    for char in "\"'()[]{}:;,. - …":
        clean_headline = clean_headline.replace(char, " ")
    
    words = clean_headline.split()
    keywords = [w for w in words if w not in stop_words and len(w) > 2 and not w.isdigit()]
    
    # Limiter et capitaliser
    result = " ".join(keywords[:max_words])
    # Capitaliser le premier mot
    if result:
        result = result[0].upper() + result[1:]
    return result


def create_professional_summary(news_list: list, regime: str = "", max_items: int = 3) -> str:
    """
    Crée une synthèse PROFESSIONNELLE des top news (phrase bien rédigée).
    Combine les themes des articles en une narrative cohérente.
    Le régime est utilisé pour contextualiser le résumé.
    
    Exemple: "Tech sector decline driven by Apple earnings miss and semiconductor weakness"
    """
    if not news_list:
        return "No significant news events"

    # Extraire les themes principaux
    themes = []
    for news_item in news_list[:max_items]:
        headline = news_item.get("headline", "")
        keywords = extract_keywords(headline, max_words=3)
        if keywords:
            themes.append(keywords)

    if not themes:
        return "News-driven market adjustment"

    # Construire une phrase professionnelle
    summary = ", ".join(themes)
    
    # Déterminer le contexte général
    sentiments = [n.get("sentiment_label", "neutral").lower() for n in news_list[:max_items]]
    neg_count = sentiments.count("negative")
    pos_count = sentiments.count("positive")
    
    if neg_count >= pos_count + 1:
        context = "Market concerns"
    elif pos_count > neg_count:
        context = "Positive developments"
    else:
        context = "Mixed market signals"
    
    return f"{context}: {summary}"


def get_sentiment_color(sentiment_label: str) -> str:
    """Retourne la couleur basée sur le sentiment."""
    return SENTIMENT_COLORS.get(sentiment_label, "#FFD700")


# ══════════════════════════════════════════════════════════════════════════════
# 2. Chargement des données
# ══════════════════════════════════════════════════════════════════════════════

def load_data():
    """Charge toutes les données nécessaires."""
    # NASDAQ
    nasdaq = pd.read_csv(
        NASDAQ_PATH,
        sep=";",
        decimal=",",
        parse_dates=["Date"]
    )
    nasdaq.rename(columns={"Date": "date", "Prix de clôture": "close"}, inplace=True)
    nasdaq['date'] = pd.to_datetime(nasdaq['date'], dayfirst=True)
    nasdaq = nasdaq.sort_values("date").reset_index(drop=True)

    # Régimes
    regimes = pd.read_csv(REGIME_PATH, sep=",", parse_dates=["date"])
    regimes['date'] = pd.to_datetime(regimes['date'])
    regimes = regimes.sort_values("date").reset_index(drop=True)

    # Attribution
    with open(ATTRIBUTION_PATH, 'r') as f:
        attribution = json.load(f)

    # Converter en dict par date pour accès facile
    attribution_dict = {}
    for attr in attribution:
        try:
            change_date = pd.to_datetime(attr['regime_change_date'])
            attribution_dict[change_date] = attr
        except:
            pass

    return nasdaq, regimes, attribution_dict


# ══════════════════════════════════════════════════════════════════════════════
# 3. Création du graphe
# ══════════════════════════════════════════════════════════════════════════════

def plot_attribution_graph(nasdaq, regimes, attribution_dict, figsize=(24, 10)):
    """
    Crée le graphe complet avec:
    - Prix NASDAQ en noir/épais
    - Régimes en arrière-plan (couleurs semi-transparentes)
    - Annotations synthétiques des news pour chaque régime
    """

    fig, ax = plt.subplots(figsize=figsize)

    # Tracer le prix NASDAQ (ligne épaisse)
    ax.plot(nasdaq['date'], nasdaq['close'], color='black', label='Prix NASDAQ', linewidth=2.5, zorder=5)

    # Ajouter les régimes en arrière-plan avec plus de visibilité
    for idx, row in regimes.iterrows():
        start_date = row['date']
        regime_color = REGIME_COLORS.get(row['regime'], '#808080')

        # Trouver la date de fin du régime
        next_idx = idx + 1
        if next_idx < len(regimes):
            end_date = regimes.iloc[next_idx]['date']
        else:
            end_date = nasdaq['date'].iloc[-1]

        ax.axvspan(start_date, end_date, color=regime_color, alpha=0.18, zorder=0, linewidth=0)

    # Ajouter les annotations d'attribution
    add_attribution_annotations(ax, nasdaq, attribution_dict)

    # Formatage de l'axe x (par année)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=12))  
    ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(fontsize=9)

    # Axe y formaté
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

    # Légende des régimes
    handles_regime = [
        plt.Rectangle((0, 0), 1, 1, facecolor=REGIME_COLORS[r], alpha=0.5, edgecolor='black', linewidth=0.5)
        for r in ["bear_strong", "bear", "sideways", "bull", "bull_strong"]
    ]
    labels_regime = ["Bear Strong", "Bear", "Sideways", "Bull", "Bull Strong"]

    # Légende des sentiments
    handles_sentiment = [
        plt.Rectangle((0, 0), 1, 1, facecolor=SENTIMENT_COLORS[s], alpha=0.7, edgecolor='black', linewidth=0.5)
        for s in ["negative", "neutral", "positive"]
    ]
    labels_sentiment = ["Negative News", "Neutral News", "Positive News"]

    # Ligne pour le prix
    handles_price = [plt.Line2D([0], [0], color='black', linewidth=2)]
    labels_price = ['NASDAQ Price']

    all_handles = handles_price + handles_regime + handles_sentiment
    all_labels = labels_price + labels_regime + labels_sentiment

    ax.legend(
        handles=all_handles,
        labels=all_labels,
        loc='upper left',
        fontsize=10,
        title='Market Regimes & News Attribution',
        title_fontsize=11,
        ncol=3,
        framealpha=0.95,
        edgecolor='black'
    )

    ax.set_title(
        "NASDAQ Market Regimes & News-Driven Attribution\n"
        "Regime Changes Linked to Financial News Narratives",
        fontsize=16,
        fontweight='bold',
        pad=20
    )
    ax.set_ylabel("Close Price ($)", fontsize=12, fontweight='bold')
    ax.set_xlabel("Date", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.5)

    # Meilleure visibilité
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('gray')
    ax.spines['bottom'].set_color('gray')

    plt.tight_layout()
    return fig, ax


# ══════════════════════════════════════════════════════════════════════════════
# 4. Ajout des annotations d'attribution
# ══════════════════════════════════════════════════════════════════════════════

def add_attribution_annotations(ax, nasdaq, attribution_dict):
    """
    Ajoute des points numérotés pour chaque changement de régime.
    Les numéros se réfèrent au tableau détaillé dans la deuxième fenêtre.
    """

    # Obtenir les limites du graphe
    y_min = nasdaq['close'].min()
    y_max = nasdaq['close'].max()
    y_range = y_max - y_min

    # Parcourir les changements de régime
    change_dates = sorted(attribution_dict.keys())

    # Alterner la position des annotations haut/bas pour éviter les superpositions
    position_toggle = True
    annotations_placed = 0

    for i, change_date in enumerate(change_dates):
        attr = attribution_dict[change_date]

        # Sauter si pas de news - mais TRAITER TOUS les changements
        top_news = attr.get('top_news', [])
        if not top_news:
            continue

        # Trouver le prix à cette date
        price_data = nasdaq[nasdaq['date'] >= change_date]
        if price_data.empty:
            price_data = nasdaq[nasdaq['date'] <= change_date]
        if price_data.empty:
            continue

        price = price_data.iloc[0]['close']

        # Déterminer la couleur basée sur le sentiment
        market_context = attr.get('market_context', {})
        sentiment_mean = attr.get('news_window', {}).get('sentiment_mean', 0.0)

        if sentiment_mean < -0.2:
            point_color = SENTIMENT_COLORS['negative']
        elif sentiment_mean > 0.2:
            point_color = SENTIMENT_COLORS['positive']
        else:
            point_color = SENTIMENT_COLORS['neutral']

        # Ajouter un point avec le numéro
        annotations_placed += 1
        annotation_number = annotations_placed

        # Point sur le graphe
        ax.scatter(change_date, price, color=point_color, s=120, alpha=0.8, zorder=9, 
                   marker='o', edgecolors='black', linewidth=1.5)
        
        # Ajouter le numéro directement sur le point (plus proche)
        ax.text(change_date, price, str(annotation_number), 
                ha='center', va='center', fontsize=7, fontweight='bold', color='white',
                zorder=12)

        position_toggle = not position_toggle

    return annotations_placed


# ══════════════════════════════════════════════════════════════════════════════
# 5. Création du tableau détaillé des attributions
# ══════════════════════════════════════════════════════════════════════════════

def create_html_report(attribution_dict, output_path, png_path=None):
    """
    Crée un rapport HTML interactif avec tableau déroulable et professionnel.
    Contient TOUTES les attributions avec news lisibles + graphe PNG intégré.
    """
    
    # Préparer les données
    change_dates = sorted(attribution_dict.keys())
    rows_html = []
    
    bearish_count = 0
    bullish_count = 0
    neutral_count = 0
    
    # Analyser la distribution des sentiments (DEBUG)
    all_sentiments = []
    
    for idx, change_date in enumerate(change_dates):
        attr = attribution_dict[change_date]
        top_news = attr.get('top_news', [])
        
        if not top_news:
            continue
        
        sentiment_mean = attr.get('news_window', {}).get('sentiment_mean', 0.0)
        all_sentiments.append(sentiment_mean)
    
    if all_sentiments:
        print(f"[DEBUG] Sentiment stats: min={min(all_sentiments):.4f}, max={max(all_sentiments):.4f}, mean={np.mean(all_sentiments):.4f}, median={np.median(all_sentiments):.4f}")
        print(f"[DEBUG] Sentiments > 0.1: {len([s for s in all_sentiments if s > 0.1])}")
        print(f"[DEBUG] Sentiments > 0.0: {len([s for s in all_sentiments if s > 0.0])}")
    
    for idx, change_date in enumerate(change_dates):
        attr = attribution_dict[change_date]
        top_news = attr.get('top_news', [])
        
        if not top_news:
            continue
        
        market_context = attr.get('market_context', {})
        regime = market_context.get('regime', '').replace('_', ' ').title()
        regime_raw = market_context.get('regime', '').lower()
        sentiment_mean = attr.get('news_window', {}).get('sentiment_mean', 0.0)
        news_count = attr.get('news_window', {}).get('total_articles', 0)
        
        # --- CODE CORRIGÉ ---
        # 1. On détermine la couleur de la LIGNE selon le RÉGIME (Prix)
        if "bear" in regime_raw:
            row_color = "#ffebee"  # Rouge très clair pour le fond
            regime_display = "BEARISH MARKET"
        elif "bull" in regime_raw:
            row_color = "#e8f5e9"  # Vert très clair pour le fond
            regime_display = "BULLISH MARKET"
        else:
            row_color = "#ffffff"
            regime_display = regime

        # 2. On détermine l'étiquette de SENTIMENT avec des seuils plus stricts
        # On évite que 0.01 soit considéré comme BULLISH
        if sentiment_mean < -0.15:
            sentiment_text = "NEGATIVE NEWS"
            sentiment_color = "#d32f2f" # Rouge vif
            negative_count_for_stats = 1
            positive_count_for_stats = 0
            neutral_count_for_stats = 0
        elif sentiment_mean > 0.15:
            sentiment_text = "POSITIVE NEWS"
            sentiment_color = "#388e3c" # Vert vif
            negative_count_for_stats = 0
            positive_count_for_stats = 1
            neutral_count_for_stats = 0
        else:
            sentiment_text = "NEUTRAL / MIXED"
            sentiment_color = "#757575" # Gris
            negative_count_for_stats = 0
            positive_count_for_stats = 0
            neutral_count_for_stats = 1
            
        # 3. Actualiser les compteurs
        bearish_count += negative_count_for_stats
        bullish_count += positive_count_for_stats
        neutral_count += neutral_count_for_stats
        
        # 4. On génère le résumé en passant le type de régime pour le contexte
        summary = create_professional_summary(top_news, regime_raw, max_items=3)
        # -----------------------
        
        date_str = change_date.strftime("%Y-%m-%d")
        
        # Créer le détail des news avec tous les articles
        news_details = "<ul style='margin: 5px 0; padding-left: 20px;'>"
        for ni, news_item in enumerate(top_news[:5], 1):
            headline = news_item.get("headline", "")
            sent = news_item.get("sentiment_label", "neutral")
            topic = TOPIC_LABELS.get(news_item.get("topic", "?"), "?")
            news_details += f"<li style='margin: 5px 0;'><small>{headline[:100]}... <br><em>({sent} | {topic})</em></small></li>"
        news_details += "</ul>"
        
        row_html = f"""
        <tr style="background-color: {row_color}; border-bottom: 1px solid #ddd;">
            <td style="padding: 12px; text-align: center; font-weight: bold;">{idx + 1}</td>
            <td style="padding: 12px; text-align: center;">{date_str}</td>
            <td style="padding: 12px; text-align: center; font-weight: bold;">{regime_display}</td>
            <td style="padding: 12px;">{summary}</td>
            <td style="padding: 12px; text-align: center;">
                <span style="background-color: {sentiment_color}; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 11px;">
                    {sentiment_text}
                </span>
            </td>
            <td style="padding: 12px; text-align: center;">
                <details>
                    <summary style="color: #1976d2;">{len(top_news)} articles</summary>
                    {news_details}
                </details>
            </td>
        </tr>
        """
        rows_html.append(row_html)
    
    # Convertir PNG en base64 si disponible
    png_img_html = ""
    if png_path and Path(png_path).exists():
        import base64
        with open(png_path, 'rb') as f:
            png_data = base64.b64encode(f.read()).decode('utf-8')
        png_img_html = f"""
        <div style="margin-bottom: 30px; text-align: center;">
            <h2 style="color: #2c3e50; margin-bottom: 20px;">NASDAQ Price Graph with Regime Changes</h2>
            <img src="data:image/png;base64,{png_data}" style="max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
        </div>
        """
    
    # Créer le HTML complet
    html_content = f"""
    <!DOCTYPE html>
    <html lang="fr">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Regime Attribution Report</title>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 20px;
                min-height: 100vh;
            }}
            
            .container {{
                max-width: 1400px;
                margin: 0 auto;
                background: white;
                border-radius: 12px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.15);
                overflow: hidden;
            }}
            
            .header {{
                background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
                color: white;
                padding: 30px;
                text-align: center;
            }}
            
            .header h1 {{
                font-size: 28px;
                margin-bottom: 10px;
            }}
            
            .header p {{
                font-size: 14px;
                opacity: 0.9;
            }}
            
            .content {{
                padding: 30px;
            }}
            
            .graph-section {{
                margin-bottom: 40px;
            }}
            
            .stats {{
                display: flex;
                justify-content: space-around;
                background: #f5f5f5;
                padding: 20px;
                border-bottom: 1px solid #ddd;
                margin-top: 20px;
            }}
            
            .stat-box {{
                text-align: center;
                flex: 1;
            }}
            
            .stat-number {{
                font-size: 24px;
                font-weight: bold;
                color: #667eea;
            }}
            
            .stat-label {{
                font-size: 12px;
                color: #666;
                margin-top: 5px;
            }}
            
            .table-wrapper {{
                overflow-x: auto;
                padding: 20px;
            }}
            
            table {{
                width: 100%;
                border-collapse: collapse;
                font-size: 14px;
            }}
            
            th {{
                background-color: #2c3e50;
                color: white;
                padding: 15px;
                text-align: left;
                font-weight: 600;
                border: 1px solid #34495e;
                position: sticky;
                top: 0;
            }}
            
            td {{
                border: 1px solid #ecf0f1;
            }}
            
            tr:hover {{
                background-color: rgba(102, 126, 234, 0.05);
            }}
            
            details {{
                cursor: pointer;
            }}
            
            details summary {{
                cursor: pointer;
                user-select: none;
            }}
            
            details[open] summary {{
                margin-bottom: 10px;
            }}
            
            summary::-webkit-details-marker {{
                color: #667eea;
            }}
            
            .footer {{
                background: #f5f5f5;
                padding: 20px;
                text-align: center;
                color: #666;
                font-size: 12px;
                border-top: 1px solid #ddd;
            }}
            
            .legend {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                padding: 20px;
                background: #f9f9f9;
                border-top: 1px solid #ddd;
            }}
            
            .legend-item {{
                display: flex;
                align-items: center;
                gap: 10px;
            }}
            
            .legend-color {{
                width: 30px;
                height: 30px;
                border-radius: 4px;
                border: 1px solid #999;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>NASDAQ Regime Attribution Report</h1>
                <p>Financial News-Driven Market Regime Changes Analysis</p>
            </div>
            
            <div class="content">
                <div class="graph-section">
                    {png_img_html}
                </div>
                
                <h2 style="color: #2c3e50; margin: 30px 0 20px 0; text-align: center;">Attribution Summary</h2>
                
                <div class="stats">
                    <div class="stat-box">
                        <div class="stat-number">{len(rows_html)}</div>
                        <div class="stat-label">Regime Changes</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-number">{bearish_count}</div>
                        <div class="stat-label">Bearish Transitions</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-number">{bullish_count}</div>
                        <div class="stat-label">Bullish Transitions</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-number">{neutral_count}</div>
                        <div class="stat-label">Neutral Transitions</div>
                    </div>
                </div>
                
                <div class="table-wrapper">
                    <table>
                        <thead>
                            <tr>
                                <th style="width: 5%;">#</th>
                                <th style="width: 12%;">Date</th>
                                <th style="width: 12%;">Regime</th>
                                <th style="width: 45%;">News Summary</th>
                                <th style="width: 12%;">Sentiment</th>
                                <th style="width: 14%;">Articles</th>
                            </tr>
                        </thead>
                        <tbody>
                            {''.join(rows_html)}
                        </tbody>
                    </table>
                </div>
                
                <div class="legend">
                    <div class="legend-item">
                        <div class="legend-color" style="background-color: #d32f2f;"></div>
                        <span><strong>NEGATIVE NEWS:</strong> Sentiment Mean &lt; -0.15</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background-color: #388e3c;"></div>
                        <span><strong>POSITIVE NEWS:</strong> Sentiment Mean &gt; +0.15</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background-color: #757575;"></div>
                        <span><strong>NEUTRAL / MIXED:</strong> -0.15 &le; Sentiment Mean &le; +0.15</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background-color: #ffebee;"></div>
                        <span><strong>Row Color:</strong> Background indicates market regime (Bear/Bull/Sideways)</span>
                    </div>
                </div>
            </div>
            
            <div class="footer">
                <p>Interactive report generated from NASDAQ regime detection & news attribution analysis</p>
                <p>Click "articles" rows to expand and view full news details | Numbered points on chart match row numbers</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Sauvegarder le HTML
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\n[DEBUG] Sentiment distribution: BEARISH={bearish_count}, BULLISH={bullish_count}, NEUTRAL={neutral_count}")
    
    return output_path


# ══════════════════════════════════════════════════════════════════════════════
# 6. Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("Chargement des donnees...")
    nasdaq, regimes, attribution_dict = load_data()

    print(f"  NASDAQ: {len(nasdaq)} jours de donnees")
    print(f"  Regimes: {len(regimes)} periodes detectees")
    print(f"  Changements de regime attribues: {len(attribution_dict)}")

    print("\nCreation du graphe principal (Prix + Regimes + Points numerotes)...")
    fig1, ax1 = plot_attribution_graph(nasdaq, regimes, attribution_dict)

    # Sauvegarder le graphe
    output_dir = Path("data\\processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path1 = output_dir / "plot_09_attribution_graph.png"
    fig1.savefig(output_path1, dpi=150, bbox_inches='tight')
    print(f"[OK] Graphe principal sauvegarde: {output_path1}")

    print("Creation du rapport HTML interactif...")
    output_path2 = output_dir / "plot_09_attribution_report.html"
    create_html_report(attribution_dict, output_path2, png_path=output_path1)
    print(f"[OK] Rapport HTML sauvegarde: {output_path2}")


if __name__ == "__main__":
    main()
