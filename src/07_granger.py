"""
Etape 8 - Test de causalite de Granger
Pipeline : News-Driven Regime Attribution

Objectif : tester si le sentiment des news predit les rendements du NASDAQ
           (et inversement), et si les features textuelles precedent les
           changements de regime.

Input  : data/processed/dataset_granger.csv

Output : data/processed/granger_results.csv  -> resultats bruts des tests
         data/processed/granger_report.txt   -> rapport lisible
"""

import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
import warnings
import logging

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Chemins ────────────────────────────────────────────────────────────────────
INPUT_PATH = Path("data\\processed\\dataset_granger.csv")
OUT_CSV    = Path("data\\processed\\granger_results.csv")
OUT_REPORT = Path("data\\processed\\granger_report.txt")
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

# ── Parametres ─────────────────────────────────────────────────────────────────
MAX_LAG = 5      # jours de decalage testes (1 a 5)
ALPHA   = 0.05   # seuil de significativite


# ══════════════════════════════════════════════════════════════════════════════
# 1. Chargement
# ══════════════════════════════════════════════════════════════════════════════

def load_data(path: Path) -> pd.DataFrame:
    log.info(f"Chargement de {path} ...")
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df = df.dropna(subset=["returns", "volatility_20d"])
    log.info(f"  {len(df)} lignes apres suppression NaN sur returns/volatility")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. Test de stationnarite ADF (prerequis de Granger)
# ══════════════════════════════════════════════════════════════════════════════

def check_and_fix_stationarity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Test ADF sur chaque serie. Si non stationnaire, applique une difference d'ordre 1.
    H0 ADF : la serie a une racine unitaire (non stationnaire).
    p < ALPHA => on rejette H0 => serie stationnaire.
    """
    log.info("Tests de stationnarite ADF ...")
    df = df.copy()

    cols = [
        "returns", "sentiment_mean", "sentiment_min", "sentiment_std",
        "pct_negative", "volatility_20d", "vix", "information_shift",
        "news_count", "y"
    ]

    for col in cols:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        if len(series) < 20:
            continue
        adf_result = adfuller(series, autolag="AIC")
        p = adf_result[1]
        stationary = p < ALPHA
        status = "OK stationnaire" if stationary else "-> difference appliquee"
        log.info(f"  {col:30s}  p={p:.4f}  {status}")
        if not stationary:
            df[col] = df[col].diff()

    df = df.dropna().reset_index(drop=True)
    log.info(f"  {len(df)} lignes apres differenciation")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 3. Test de Granger unitaire
# ══════════════════════════════════════════════════════════════════════════════

def granger_test(df: pd.DataFrame, cause: str, effect: str) -> dict:
    """
    Teste si [cause] cause au sens de Granger [effect].
    Convention statsmodels : data = [[effect, cause]].
    Retourne le meilleur lag (p-value minimale) et son p-value.
    """
    data = df[[effect, cause]].dropna()

    if len(data) < MAX_LAG * 10:
        log.warning(f"  Pas assez de donnees : {cause} -> {effect}")
        return {
            "cause": cause, "effect": effect,
            "best_lag": None, "p_value": None, "significant": False
        }

    try:
        results = grangercausalitytests(data[[effect, cause]], maxlag=MAX_LAG, verbose=False)

        best_lag, best_p = 1, 1.0
        lag_details = []
        for lag in range(1, MAX_LAG + 1):
            p = results[lag][0]["ssr_ftest"][1]
            lag_details.append({"lag": lag, "p_value": round(p, 6)})
            if p < best_p:
                best_p, best_lag = p, lag

        significant = best_p < ALPHA
        marker = "  *** SIGNIFICATIF" if significant else ""
        log.info(
            f"  {cause:30s} -> {effect:20s} | "
            f"best_lag={best_lag}  p={best_p:.4f}{marker}"
        )
        return {
            "cause": cause, "effect": effect,
            "best_lag": best_lag, "p_value": round(best_p, 6),
            "significant": significant,
            "lag_details": lag_details,
        }

    except Exception as e:
        log.warning(f"  Erreur {cause} -> {effect} : {e}")
        return {
            "cause": cause, "effect": effect,
            "best_lag": None, "p_value": None, "significant": False
        }


# ══════════════════════════════════════════════════════════════════════════════
# 4. Lancer tous les tests
# ══════════════════════════════════════════════════════════════════════════════

def run_all_tests(df: pd.DataFrame) -> list:
    results = []

    # ── A : News -> Marche ────────────────────────────────────────────────────
    # Question : le sentiment des j derniers jours predit-il les rendements ?
    log.info("")
    log.info("[A] News -> Marche : le sentiment predit-il les rendements ?")
    for cause in ["sentiment_mean", "sentiment_min", "pct_negative", "information_shift"]:
        if cause in df.columns:
            r = granger_test(df, cause=cause, effect="returns")
            r["section"] = "A"
            results.append(r)

    # ── B : Marche -> News ────────────────────────────────────────────────────
    # Question : les rendements predisent-ils le sentiment ?
    # Si B significatif mais pas A => causalite inversee (journalistes reactifs)
    log.info("")
    log.info("[B] Marche -> News : les rendements predisent-ils le sentiment ?")
    for effect in ["sentiment_mean", "pct_negative"]:
        if effect in df.columns:
            r = granger_test(df, cause="returns", effect=effect)
            r["section"] = "B"
            results.append(r)

    # ── C : News -> Changement de regime ─────────────────────────────────────
    # Question : les features textuelles precedent-elles un changement de regime ?
    log.info("")
    log.info("[C] News -> Regime : les features textuelles predisent-elles un changement ?")
    for cause in ["sentiment_mean", "sentiment_min", "pct_negative",
                  "information_shift", "news_count"]:
        if cause in df.columns:
            r = granger_test(df, cause=cause, effect="y")
            r["section"] = "C"
            results.append(r)

    # ── D : Marche -> Regime (baseline) ──────────────────────────────────────
    # Baseline : les features de marche seules predisent-elles deja le regime ?
    # Sert a comparer le pouvoir predictif des news vs le marche lui-meme.
    log.info("")
    log.info("[D] Marche -> Regime (baseline sans news)")
    for cause in ["returns", "volatility_20d", "vix"]:
        if cause in df.columns:
            r = granger_test(df, cause=cause, effect="y")
            r["section"] = "D"
            results.append(r)

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 5. Rapport textuel
# ══════════════════════════════════════════════════════════════════════════════

SECTION_LABELS = {
    "A": "News -> Rendements  (le sentiment predit-il le marche ?)",
    "B": "Rendements -> News  (le marche predit-il le sentiment ?)",
    "C": "News -> Regime      (les features textuelles predisent-elles un changement ?)",
    "D": "Marche -> Regime    (baseline sans features textuelles)",
}

INTERPRETATION = {
    "A": (
        "Si SIGNIFICATIF : les news ont un pouvoir predictif sur les rendements.\n"
        "Comparer avec la section B pour detecter une eventuelle causalite inversee.\n"
        "Si A significatif ET B non : les news precedent reellement le marche.\n"
        "Si A et B tous les deux significatifs : relation bidirectionnelle."
    ),
    "B": (
        "Si SIGNIFICATIF : le marche predit le sentiment -> les journalistes\n"
        "reagissent aux mouvements de prix plutot qu'ils ne les causent.\n"
        "Cela ne disqualifie pas les news, mais indique une causalite partielle."
    ),
    "C": (
        "Si SIGNIFICATIF : les features textuelles des j derniers jours\n"
        "precedent statistiquement un changement de regime.\n"
        "C'est le resultat cle pour valider l'approche du projet."
    ),
    "D": (
        "Baseline : pouvoir predictif des features de marche seules.\n"
        "Si D significatif mais pas C : le marche se predit lui-meme,\n"
        "les news n'apportent pas d'information supplementaire.\n"
        "Si C significatif ET D non : les news apportent un signal unique."
    ),
}


def build_report(results: list, df: pd.DataFrame) -> str:
    lines = []
    sep = "=" * 65

    lines += [
        sep,
        "RAPPORT - CAUSALITE DE GRANGER",
        "News-Driven Regime Attribution Pipeline",
        sep,
        f"Periode    : {df['date'].min().date()} -> {df['date'].max().date()}",
        f"Obs.       : {len(df)}",
        f"Max lag    : {MAX_LAG} jours",
        f"Alpha      : {ALPHA}",
        "",
    ]

    by_section = {}
    for r in results:
        s = r.get("section", "?")
        by_section.setdefault(s, []).append(r)

    for section in ["A", "B", "C", "D"]:
        items = by_section.get(section, [])
        lines += ["", "-" * 65, SECTION_LABELS[section], "-" * 65]

        sig_count = sum(1 for r in items if r.get("significant"))
        lines.append(f"Resultats : {sig_count}/{len(items)} tests significatifs")
        lines.append("")

        for r in items:
            if r["p_value"] is None:
                lines.append(f"  {r['cause']:28s} -> {r['effect']:15s} | donnees insuffisantes")
            else:
                marker = " ***" if r["significant"] else "    "
                lines.append(
                    f"  {r['cause']:28s} -> {r['effect']:15s} |"
                    f" lag={r['best_lag']}  p={r['p_value']:.4f}{marker}"
                )

        lines += ["", "Interpretation :"]
        for il in INTERPRETATION[section].split("\n"):
            lines.append("  " + il)

    # Synthese finale
    lines += ["", sep, "SYNTHESE", sep]
    sig_a = [r for r in results if r.get("section") == "A" and r.get("significant")]
    sig_b = [r for r in results if r.get("section") == "B" and r.get("significant")]
    sig_c = [r for r in results if r.get("section") == "C" and r.get("significant")]
    sig_d = [r for r in results if r.get("section") == "D" and r.get("significant")]

    if sig_c and not sig_d:
        lines.append("=> Les news apportent un signal UNIQUE pour predire les regimes.")
        lines.append("   Les features de marche seules ne suffisent pas.")
    elif sig_c and sig_d:
        lines.append("=> Les news ET le marche predisent les changements de regime.")
        lines.append("   Combiner les deux types de features dans le modele ML.")
    elif not sig_c and sig_d:
        lines.append("=> Le marche se predit lui-meme.")
        lines.append("   Les news n'apportent pas de signal supplementaire.")
        lines.append("   Verifier la qualite/quantite des donnees textuelles.")
    else:
        lines.append("=> Aucun signal de Granger significatif detecte.")
        lines.append("   Envisager d'augmenter MAX_LAG ou de changer la granularite.")

    if sig_a and not sig_b:
        lines.append("=> Causalite news -> marche : les news precedent les rendements.")
    elif sig_b and not sig_a:
        lines.append("=> Causalite inversee : le marche precede les news (journalistes reactifs).")
    elif sig_a and sig_b:
        lines.append("=> Relation bidirectionnelle news <-> marche.")

    lines.append("")
    lines.append("Prochaine etape : 09_causal_ml.py (dowhy / causalnex)")
    lines.append(sep)

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# 6. Point d'entree
# ══════════════════════════════════════════════════════════════════════════════

def main():
    df_raw = load_data(INPUT_PATH)
    df     = check_and_fix_stationarity(df_raw)

    log.info("")
    log.info("=" * 60)
    log.info("LANCEMENT DES TESTS DE GRANGER")
    log.info("=" * 60)
    results = run_all_tests(df)

    # Sauvegarde CSV des resultats bruts
    rows = []
    for r in results:
        rows.append({
            "section":     r.get("section"),
            "cause":       r["cause"],
            "effect":      r["effect"],
            "best_lag":    r["best_lag"],
            "p_value":     r["p_value"],
            "significant": r["significant"],
        })
    df_results = pd.DataFrame(rows)
    df_results.to_csv(OUT_CSV, index=False)
    log.info(f"\nOK Resultats bruts sauvegardes -> {OUT_CSV}")

    # Rapport textuel
    report = build_report(results, df_raw)
    OUT_REPORT.write_text(report, encoding="utf-8")
    log.info(f"OK Rapport sauvegarde -> {OUT_REPORT}")

    # Afficher le rapport dans le terminal
    print("\n" + report)


if __name__ == "__main__":
    main()