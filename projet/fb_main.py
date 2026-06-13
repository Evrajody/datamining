"""
fb_main — fichier APPELANT du module scrapper_fb
================================================
Importe le module scrapper_fb et expose ses 3 fonctionnalités en ligne de
commande, avec export CSV (les champs listes/dicts sont sérialisés en JSON).

EXEMPLES :
    uv run python projet/fb_main.py amis junior.mignon.581
    uv run python projet/fb_main.py posts nintendo --scrolls 25
    uv run python projet/fb_main.py recherche "intelligence artificielle"
    uv run python projet/fb_main.py recherche "#datamining" --no-headless

Options : --scrolls N, --out fichier.csv, --no-headless, --cookies chemin
"""

import argparse
import csv
import json
import os

from scrapper_fb import COOKIES_DEFAUT, ScraperFacebook

ICI = os.path.dirname(__file__)
DOSSIER_SORTIE = os.path.join(ICI, "..", "off-dataset")


def exporter_csv(lignes, chemin):
    """Écrit une liste de dicts en CSV (listes/dicts -> JSON)."""
    if not lignes:
        return 0
    os.makedirs(os.path.dirname(chemin), exist_ok=True)
    champs = list(lignes[0].keys())
    with open(chemin, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=champs)
        w.writeheader()
        for ligne in lignes:
            w.writerow({k: (json.dumps(v, ensure_ascii=False)
                            if isinstance(v, (list, dict)) else v)
                        for k, v in ligne.items()})
    return len(lignes)


def exporter_json(lignes, chemin):
    """Écrit la liste de dicts en JSON (UTF-8, indenté, format fourni)."""
    if not lignes:
        return 0
    os.makedirs(os.path.dirname(chemin), exist_ok=True)
    with open(chemin, "w", encoding="utf-8") as f:
        json.dump(lignes, f, ensure_ascii=False, indent=2)
    return len(lignes)


def apercu(ligne):
    return ligne.get("nom") or (ligne.get("text", "") or "")[:80]


def main():
    ap = argparse.ArgumentParser(description="Scraper Facebook (module scrapper_fb).")
    sub = ap.add_subparsers(dest="action", required=True)

    pa = sub.add_parser("amis", help="récupérer les amis d'un profil")
    pa.add_argument("cible"); pa.add_argument("--scrolls", type=int, default=30)
    pp = sub.add_parser("posts", help="récupérer les posts d'un profil/page")
    pp.add_argument("cible"); pp.add_argument("--scrolls", type=int, default=20)
    pr = sub.add_parser("recherche", help="récupérer les posts d'une recherche/#tag")
    pr.add_argument("cible"); pr.add_argument("--scrolls", type=int, default=20)

    for p in (pp, pr):   # options posts / recherche
        p.add_argument("--graphql", action="store_true",
                       help="interception GraphQL : texte+likes+comments+shares+date (RECOMMANDÉ)")
        p.add_argument("--details", action="store_true",
                       help="2e passe par permalien (vrais post_id/url)")
        p.add_argument("--max-details", type=int, default=30)

    for p in (pa, pp, pr):
        p.add_argument("--cookies", default=COOKIES_DEFAUT)
        p.add_argument("--format", choices=["json", "csv"], default="json",
                       help="format de sortie (défaut : json)")
        p.add_argument("--out", default=None)
        p.add_argument("--no-headless", dest="headless", action="store_false")
        p.set_defaults(headless=True)

    args = ap.parse_args()
    slug = "".join(c if c.isalnum() else "_" for c in args.cible).strip("_")[:40]
    out = args.out or os.path.join(
        DOSSIER_SORTIE, f"fb_{args.action}_{slug}.{args.format}")

    try:
        with ScraperFacebook(cookies=args.cookies, headless=args.headless) as fb:
            if args.action == "amis":
                lignes = fb.amis(args.cible, scrolls=args.scrolls)
            elif args.action == "posts":
                lignes = fb.posts(args.cible, scrolls=args.scrolls,
                                  details=args.details, graphql=args.graphql,
                                  max_details=args.max_details)
            else:
                lignes = fb.recherche(args.cible, scrolls=args.scrolls,
                                      details=args.details, graphql=args.graphql,
                                      max_details=args.max_details)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"❌  {e}")
        return

    n = (exporter_json if args.format == "json" else exporter_csv)(lignes, out)
    if n == 0:
        print("Aucun résultat (contenu privé, cookies expirés, ou HTML modifié).")
        return
    print(f"\n✅  {n} résultat(s) -> {out}")
    for ligne in lignes[:8]:
        print(f"   • {apercu(ligne)}")


if __name__ == "__main__":
    main()
