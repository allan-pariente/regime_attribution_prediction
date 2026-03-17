"""
debug_guardian.py — Test rapide de la clé Guardian
Lance : python debug_guardian.py
"""
import requests

# ← Colle ta clé ici
GUARDIAN_API_KEY = "30879444-61c6-4d54-8c19-7b12a6acb989"

params = {
    "q": "NASDAQ stock market",
    "from-date": "2020-03-02",
    "to-date": "2020-03-09",
    "order-by": "relevance",
    "show-fields": "headline,trailText",
    "page-size": 10,
    "api-key": GUARDIAN_API_KEY,
}

print(f"Clé utilisée : {GUARDIAN_API_KEY[:8]}...")
r = requests.get("https://content.guardianapis.com/search", params=params, timeout=10)
print(f"Status HTTP  : {r.status_code}")
print(f"Réponse brute: {r.text[:500]}")

if r.status_code == 200:
    data = r.json().get("response", {})
    print(f"\nStatut Guardian : {data.get('status')}")
    print(f"Nb résultats    : {data.get('total', 0)}")
    results = data.get("results", [])
    for a in results[:3]:
        print(f"  - {a.get('webTitle','')[:70]}")
        print(f"    {a.get('webPublicationDate','')}")