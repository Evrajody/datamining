#!/usr/bin/env python3
"""
Scraper de test - posts publics d'une Page Facebook (mbasic).
Usage :
    uv run python scraper_fb.py                # utilise la CONFIG ci-dessous
    uv run python scraper_fb.py lemondefr      # ou passe un slug de Page en argument
Sortie : resultats.json + resultats.csv
"""

import csv
import hashlib
import json
import random
import sys
import time

import requests
from bs4 import BeautifulSoup

# ----------------------- CONFIG (a modifier) -----------------------
PAGES = ["lemondefr", "RFI"]                              # slugs de Pages publiques
THEME_KEYWORDS = ["election", "inflation", "economie"]    # [] = pas de filtre
SALT = "change-moi"                                       # sel d'anonymisation
DELAY = (4, 9)                                            # pause aleatoire (s) entre Pages
OUT_JSON, OUT_CSV = "resultats.json", "resultats.csv"
# Optionnel : si deconnecte ne renvoie rien, colle ici un cookie de session FB
# (copie depuis ton navigateur) pour passer le mur de connexion.
COOKIE = ""
# -------------------------------------------------------------------

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Linux; Android 10) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124 Mobile Safari/537.36"),
    "Accept-Language": "fr-FR,fr;q=0.9",
}
if COOKIE:
    HEADERS["Cookie"] = COOKIE


def anon(name: str) -> str:
    """RGPD : on ne stocke jamais l'auteur en clair."""
    return hashlib.sha256((SALT + name).encode()).hexdigest()[:16]


def matches_theme(text: str) -> bool:
    if not THEME_KEYWORDS:
        return True
    t = text.lower()
    return any(k.lower() in t for k in THEME_KEYWORDS)


def scrape_page(slug: str) -> list[dict]:
    url = f"https://mbasic.facebook.com/{slug}"
    r = requests.get(url, headers=HEADERS, timeout=20)
    html = r.text

    # Facebook bloque desormais les requetes deconnectees (HTTP 400 "Error Facebook"
    # ou redirection /login). On le detecte pour donner un message clair.
    bloque = (
        r.status_code != 200
        or "Error Facebook" in html
        or "/login" in r.url
        or "mbasic.facebook.com/login" in html
        or "Connexion" in html[:2000]
    )
    if bloque:
        print(f"  [!] {slug} : acces bloque par Facebook "
              f"(HTTP {r.status_code}). En deconnecte, FB ne sert plus le contenu.")
        print( "      -> Solutions : (1) cookie de session FB dans COOKIE, "
               "(2) version Playwright + compte dedie, (3) API tierce (Apify).")
        with open(f"debug_{slug}.html", "w", encoding="utf-8") as f:
            f.write(html)
        return []

    soup = BeautifulSoup(html, "lxml")
    posts = []
    # Selecteurs a ajuster : le DOM de FB change souvent. On ratisse large.
    blocs = (soup.select('div[role="article"]')
             or soup.select("article")
             or soup.find_all("div", attrs={"data-ft": True}))

    for art in blocs:
        text = art.get_text(" ", strip=True)
        if len(text) < 15 or not matches_theme(text):
            continue
        lien = art.find("a", href=lambda h: h and ("story.php" in h or "/posts/" in h))
        posts.append({
            "reseau": "facebook",
            "page": slug,
            "type": "post",
            "texte": text[:1000],
            "auteur_hash": anon(slug),     # un post de Page = la Page elle-meme
            "permalink": ("https://mbasic.facebook.com" + lien["href"]) if lien else None,
            "collecte_le": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
    return posts


def main():
    pages = [sys.argv[1]] if len(sys.argv) > 1 else PAGES
    tous = []
    for slug in pages:
        print(f"-> {slug}")
        try:
            res = scrape_page(slug)
            print(f"  {len(res)} post(s) retenu(s)")
            for p in res:
                print("   -", p["texte"][:80])
            tous.extend(res)
        except Exception as e:
            print(f"  [!] erreur : {e}")
        time.sleep(random.uniform(*DELAY))   # debit faible = respectueux

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(tous, f, ensure_ascii=False, indent=2)
    if tous:
        with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=tous[0].keys())
            w.writeheader()
            w.writerows(tous)

    print(f"\n[OK] {len(tous)} post(s) -> {OUT_JSON} / {OUT_CSV}")


if __name__ == "__main__":
    main()
