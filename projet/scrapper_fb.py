"""
scrapper_fb — MODULE de scraping Facebook (Playwright)
======================================================
Module réutilisable (sans CLI : voir fb_main.py pour l'exécution).
Pilote un VRAI navigateur Chromium authentifié par cookies pour récupérer :

  1. les AMIS d'un profil        -> ScraperFacebook.amis(compte)      -> [dict]
  2. les POSTS d'un profil/page  -> ScraperFacebook.posts(compte)     -> [dict]
  3. les POSTS d'une RECHERCHE   -> ScraperFacebook.recherche(requete)-> [dict]

FORMAT d'un POST (clés alignées sur facebook_scraper, best-effort) :
  {
    'post_id', 'username', 'user_id',
    'text', 'post_text',          # texte du message (identiques)
    'time',                       # horodatage affiché (relatif) ou None
    'likes', 'comments', 'shares',# entiers (ou None si non lisible)
    'reactions',                  # détail par type non extractible -> None
    'image', 'images',            # 1re image / liste des images
    'link',                       # lien externe inclus dans le post
    'post_url',                   # permalien du post
    'available': True,
  }

PRÉREQUIS :  uv add playwright  &&  uv run playwright install chromium
COOKIES   :  session Facebook (format Netscape) dans projet/cookies.txt

⚠️  Scraper Facebook viole ses CGU ; contenus privés invisibles ; HTML mouvant.
"""

import os
import re
import time
from http.cookiejar import MozillaCookieJar
from urllib.parse import quote

from playwright.sync_api import sync_playwright

ICI = os.path.dirname(__file__)
COOKIES_DEFAUT = os.path.join(ICI, "cookies.txt")
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")

_NAV = ("friends|photos|videos|groups|watch|marketplace|gaming|events|pages|"
        "bookmarks|notifications|messages|settings|policies|help|privacy|home|"
        "me|reel|stories|profile\\.php$")

# JS : liens de profil (amis) -> {url: nom}
_JS_PROFILS = r"""
(navRe) => {
  const re = new RegExp(navRe, "i");
  const out = {};
  for (const a of document.querySelectorAll('a[href*="facebook.com/"], a[href^="/"]')) {
    const name = (a.innerText || "").trim();
    if (!name || name.length < 2 || name.length > 60) continue;
    const m = a.href.match(/facebook\.com\/(profile\.php\?id=\d+|[A-Za-z0-9.\-]+)/);
    if (!m || re.test(m[1])) continue;
    const url = "https://www.facebook.com/" + m[1];
    if (!out[url]) out[url] = name;
  }
  return out;
}
"""

# JS : posts de premier niveau -> [{text, time, post_url, link, images, raw}]
_JS_POSTS = r"""
() => {
  const out = [];
  for (const art of document.querySelectorAll('div[role="article"]')) {
    // écarter les commentaires (articles imbriqués)
    if (art.parentElement && art.parentElement.closest('div[role="article"]')) continue;

    // texte du message
    let text = "";
    const msg = art.querySelector(
      '[data-ad-comet-preview="message"], [data-ad-preview="message"], ' +
      '[data-ad-rendering-role="story_message"]');
    if (msg) text = msg.innerText.trim();

    // permalien + horodatage (l'ancre du permalien porte souvent le texte de date)
    let post_url = "", time_text = "";
    const a = art.querySelector(
      'a[href*="/posts/"], a[href*="story_fbid"], a[href*="/videos/"], ' +
      'a[href*="permalink"], a[href*="/photos/"]');
    if (a) { post_url = a.href.split("?")[0]; time_text = (a.innerText || "").trim(); }

    if (!text) text = (art.innerText || "").trim();   // fallback
    if (!text || text.length < 5) continue;

    // images du contenu
    const images = [...art.querySelectorAll('img')]
      .map(i => i.src).filter(s => s && s.includes("scontent"));

    // lien externe (redirection l.facebook.com)
    let link = "";
    const ext = art.querySelector('a[href*="l.facebook.com/l.php"]');
    if (ext) link = ext.href;

    out.push({ text, time: time_text, post_url, link, images,
               raw: (art.innerText || "") });
  }
  return out;
}
"""


# JS : permaliens propres des posts (sans paramètres de requête) -> [url]
_JS_PERMALIENS = r"""
() => {
  const out = new Set();
  const pats = [
    /https:\/\/www\.facebook\.com\/[^/?#]+\/(?:posts|videos)\/(?:pfbid[\w]+|\d+)/,
    /https:\/\/www\.facebook\.com\/reel\/\d+/,
    /https:\/\/www\.facebook\.com\/(?:story|permalink)\.php\?story_fbid=\d+&id=\d+/
  ];
  for (const a of document.querySelectorAll('a[href]')) {
    for (const p of pats) { const m = a.href.match(p); if (m) { out.add(m[0]); break; } }
  }
  return [...out];
}
"""

# JS : détail du post principal d'une page permalien
_JS_DETAIL = r"""
() => {
  const tops = [...document.querySelectorAll('div[role="article"]')]
    .filter(a => !(a.parentElement && a.parentElement.closest('div[role="article"]')));
  if (!tops.length) return null;
  // sur une page permalien, le post principal est le 1er article de premier niveau
  const art = tops[0];
  let text = "";
  const msg = art.querySelector(
    '[data-ad-comet-preview="message"], [data-ad-preview="message"], ' +
    '[data-ad-rendering-role="story_message"]');
  if (msg) text = msg.innerText.trim();
  if (!text) {                       // sinon : plus long bloc dir="auto"
    let m = 0;
    for (const d of art.querySelectorAll('div[dir="auto"]')) {
      const t = (d.innerText || "").trim();
      if (t.length > m) { m = t.length; text = t; }
    }
  }
  const images = [...art.querySelectorAll('img')]
    .map(i => i.src).filter(s => s && s.includes("scontent"));
  let link = "";
  const ext = art.querySelector('a[href*="l.facebook.com/l.php"]');
  if (ext) link = ext.href;
  // horodatage : ancre dont l'aria-label ressemble à une date
  let temps = "";
  for (const e of art.querySelectorAll('a[aria-label]')) {
    const al = e.getAttribute("aria-label") || "";
    if (/\d{4}|janv|févr|mars|avr|mai|juin|juil|août|sept|oct|nov|déc|hier|\bmin\b|\bh\b/i.test(al)) {
      temps = al; break;
    }
  }
  const labels = [...art.querySelectorAll('[aria-label]')]
    .map(e => e.getAttribute("aria-label")).filter(Boolean).join(" || ");
  return { text, images, link, time: temps, labels, raw: art.innerText || "" };
}
"""


def _charger_cookies(chemin):
    jar = MozillaCookieJar()
    jar.load(chemin, ignore_discard=True, ignore_expires=True)
    return [{
        "name": c.name, "value": c.value,
        "domain": c.domain if c.domain.startswith(".") else "." + c.domain,
        "path": c.path or "/",
        "expires": float(c.expires) if c.expires else -1,
        "httpOnly": False, "secure": True, "sameSite": "None",
    } for c in jar]


def _nombre(txt):
    """« 3,5 K » / « 1.2M » / « 459 » -> entier (ou None)."""
    if not txt:
        return None
    s = txt.upper().replace(" ", "").replace("\xa0", "").replace(" ", "")
    s = s.replace(",", ".")
    m = re.search(r"([\d.]+)\s*([KMB]?)", s)
    if not m:
        return None
    val = float(m.group(1))
    val *= {"K": 1e3, "M": 1e6, "B": 1e9}.get(m.group(2), 1)
    return int(round(val))


# ==========================================================================
# INTERCEPTION GRAPHQL — parsing du JSON interne de Facebook
# ==========================================================================
import json as _json
from datetime import datetime as _dt


def _iter_dicts(node):
    """Parcourt récursivement tous les dicts d'une structure JSON."""
    if isinstance(node, dict):
        yield node
        for v in node.values():
            yield from _iter_dicts(v)
    elif isinstance(node, list):
        for v in node:
            yield from _iter_dicts(v)


def _deep_find(node, cles, max_depth=10):
    """Première valeur trouvée pour l'une des clés (parcours en largeur)."""
    pile = [(node, 0)]
    while pile:
        n, d = pile.pop(0)
        if d > max_depth:
            continue
        if isinstance(n, dict):
            for k, v in n.items():
                if k in cles:
                    return v
                pile.append((v, d + 1))
        elif isinstance(n, list):
            for v in n:
                pile.append((v, d + 1))
    return None


def _compteur(v):
    """Entier depuis un champ compteur : int, « 1,2 K » (str i18n), ou
    {count|total_count|total_comment_count: N}."""
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, str):
        return _nombre(v)
    if isinstance(v, dict):
        for k in ("count", "total_count", "total_comment_count"):
            if isinstance(v.get(k), int):
                return v[k]
    return None


def _parser_bodies(bodies):
    """Transforme les corps de réponse GraphQL (NDJSON) en objets Python."""
    objets = []
    for b in bodies:
        if not b:
            continue
        for ligne in b.split("\n"):
            ligne = ligne.strip()
            if not ligne or not ligne.startswith("{"):
                continue
            try:
                objets.append(_json.loads(ligne))
            except Exception:                       # noqa: BLE001
                continue
    return objets


import base64 as _b64


def _postid_depuis_feedback(fid):
    """id de feedback (base64 « feedback:<POST_ID> ») -> POST_ID numérique."""
    if not isinstance(fid, str):
        return None
    try:
        dec = _b64.b64decode(fid + "===").decode("utf-8", "ignore")
    except Exception:                               # noqa: BLE001
        return None
    m = re.search(r"feedback:(\d+)", dec)
    return m.group(1) if m else None


def _collecter_feedback(objets):
    """Indexe les compteurs (likes/comments/shares) PAR POST_ID.

    Les compteurs arrivent dans des objets « feedback » séparés (réactions,
    partages, commentaires dans des fragments distincts) mais tous portent le
    même id base64 « feedback:<POST_ID> ». On décode cet id pour relier au post.
    """
    fmap = {}

    def parcours(n, courant):
        # « courant » = post_id de feedback hérité de l'ancêtre le plus proche
        if isinstance(n, dict):
            pid = (_postid_depuis_feedback(n.get("id"))
                   if isinstance(n.get("id"), str) else None)
            courant = pid or courant
            if courant:
                rec = fmap.setdefault(courant, {"likes": None, "comments": None,
                                                "shares": None})
                if rec["likes"] is None and ("reaction_count" in n
                                             or "i18n_reaction_count" in n):
                    rec["likes"] = _compteur(n.get("reaction_count")
                                             or n.get("i18n_reaction_count"))
                if rec["shares"] is None and ("share_count" in n
                                              or "i18n_share_count" in n):
                    rec["shares"] = _compteur(n.get("share_count")
                                              or n.get("i18n_share_count"))
                if rec["comments"] is None:
                    for k in ("total_comment_count", "i18n_comment_count",
                              "comment_count"):
                        if k in n:
                            rec["comments"] = _compteur(n[k])
                            break
                    if rec["comments"] is None and isinstance(n.get("comments"), dict):
                        rec["comments"] = _compteur(n["comments"])
            for v in n.values():
                parcours(v, courant)
        elif isinstance(n, list):
            for v in n:
                parcours(v, courant)

    for obj in objets:
        parcours(obj, None)
    return fmap


def _extraire_images(d):
    """URLs d'images du contenu (attachments scontent)."""
    out, vus = [], set()
    for n in _iter_dicts(d):
        img = n.get("image") if isinstance(n, dict) else None
        if isinstance(img, dict) and isinstance(img.get("uri"), str) \
                and "scontent" in img["uri"] and img["uri"] not in vus:
            vus.add(img["uri"])
            out.append(img["uri"])
    return out[:10]


def _extraire_reactions(d):
    """Détail des réactions {nom: nombre} via top_reactions (best-effort)."""
    tr = _deep_find(d, {"top_reactions"})
    edges = tr.get("edges") if isinstance(tr, dict) else None
    if not isinstance(edges, list):
        return None
    res = {}
    for e in edges:
        if not isinstance(e, dict):
            continue
        node = e.get("node") if isinstance(e.get("node"), dict) else {}
        nom = node.get("localized_name") or node.get("reaction_type") or node.get("id")
        cnt = _compteur(e.get("reaction_count"))
        if nom and cnt is not None:
            res[str(nom).lower()] = cnt
    return res or None


def _extraire_posts_graphql(objets, compte):
    """Détecte les « stories » (dicts avec feedback) et en extrait les posts."""
    fmap = _collecter_feedback(objets)
    posts, vus = [], set()
    for obj in objets:
        for d in _iter_dicts(obj):
            fb = d.get("feedback")
            if not isinstance(fb, dict):
                continue

            # compteurs : soit présents ici, soit récupérés via l'id de feedback
            reactions = _deep_find(fb, {"reaction_count", "i18n_reaction_count"})
            comments = _deep_find(fb, {"total_comment_count", "comment_count",
                                       "i18n_comment_count", "comments_count",
                                       "comments"})
            shares = _deep_find(fb, {"share_count", "i18n_share_count",
                                     "reshare_count", "i18n_reshare_count"})
            lk, cm, sh = _compteur(reactions), _compteur(comments), _compteur(shares)

            msg = _deep_find(d, {"message"})
            texte = ""
            if isinstance(msg, dict):
                texte = (msg.get("text") or "").strip()
            elif isinstance(msg, str):
                texte = msg.strip()

            post_id = _deep_find(d, {"post_id"}) or d.get("id")
            url = _deep_find(d, {"wwwURL"}) or _deep_find(d, {"url"})
            ts = _deep_find(d, {"creation_time", "publish_time"})
            acteurs = _deep_find(d, {"actors"})
            user_id = None
            if isinstance(acteurs, list) and acteurs and isinstance(acteurs[0], dict):
                user_id = acteurs[0].get("id")

            # ne garder que de VRAIS posts : texte présent OU id purement numérique
            est_post = bool(texte) or (isinstance(post_id, str) and post_id.isdigit())
            cle = post_id or texte[:60]
            if not cle or cle in vus or not est_post:
                continue
            vus.add(cle)

            # jointure des compteurs via le post_id numérique
            if (lk is None and cm is None and sh is None
                    and isinstance(post_id, str) and post_id.isdigit()):
                ref = fmap.get(post_id, {})
                lk, cm, sh = ref.get("likes"), ref.get("comments"), ref.get("shares")

            temps = None
            if isinstance(ts, int):
                try:
                    temps = _dt.utcfromtimestamp(ts).isoformat(sep=" ")
                except Exception:                   # noqa: BLE001
                    temps = ts

            images = _extraire_images(d)
            base = compte or user_id
            if isinstance(post_id, str) and post_id.isdigit() and base:
                post_url = f"https://www.facebook.com/{base}/posts/{post_id}"
            else:
                post_url = url if isinstance(url, str) else None

            posts.append({
                "post_id": str(post_id) if post_id else None,
                "username": compte,
                "user_id": str(user_id) if user_id else None,
                "text": texte,
                "post_text": texte,
                "time": temps,
                "likes": lk,
                "comments": cm,
                "shares": sh,
                "reactions": _extraire_reactions(d),
                "image": images[0] if images else None,
                "images": images,
                "link": url if isinstance(url, str) else None,
                "post_url": post_url,
                "available": True,
            })
    return posts


def _post_id_depuis_url(url):
    m = re.search(r"/(?:posts|videos|reel)/(pfbid\w+|\d+)", url or "")
    if m:
        return m.group(1)
    m = re.search(r"story_fbid=(\d+)", url or "")
    return m.group(1) if m else None


def _user_depuis_url(url):
    m = re.search(r"facebook\.com/([^/?#]+)/(?:posts|videos)/", url or "")
    if m and m.group(1) not in ("story.php", "permalink.php", "reel"):
        return m.group(1)
    m = re.search(r"[?&]id=(\d+)", url or "")
    return m.group(1) if m else None


def _detail_vers_post(detail, url, compte):
    """dict du JS détail -> dict au FORMAT documenté (avec compteurs)."""
    blob = (detail.get("labels", "") + " || " + detail.get("raw", "")) if detail else ""

    def _cherche(motif):
        m = re.search(motif, blob, re.I)
        return _nombre(m.group(1)) if m else None

    comments = _cherche(r"([\d.,\xa0\s KMB]+?)\s*commentaire")
    shares = _cherche(r"([\d.,\xa0\s KMB]+?)\s*partage")
    likes = (_cherche(r"Toutes les réactions\s*:?\s*([\d.,\xa0\s KMB]+)")
             or _cherche(r"([\d.,\xa0\s KMB]+?)\s*(?:personnes|j’aime|réaction)"))
    images = (detail.get("images") if detail else []) or []
    texte = (detail.get("text", "") if detail else "").strip()

    return {
        "post_id": _post_id_depuis_url(url),
        "username": compte or _user_depuis_url(url),
        "user_id": _user_depuis_url(url),
        "text": texte,
        "post_text": texte,
        "time": (detail.get("time") if detail else None) or None,
        "likes": likes,
        "comments": comments,
        "shares": shares,
        "reactions": None,
        "image": images[0] if images else None,
        "images": images,
        "link": (detail.get("link") if detail else None) or None,
        "post_url": url,
        "available": True,
    }


def _enrichir_post(brut, compte=None):
    """Transforme la sortie JS brute en dict au FORMAT documenté."""
    raw = brut.get("raw", "")
    post_url = brut.get("post_url", "") or ""

    post_id = None
    m = re.search(r"(?:story_fbid=|/posts/|/videos/|/photos/|fbid=)(\d+)", post_url)
    if m:
        post_id = m.group(1)

    user_id = None
    m = re.search(r"[?&]id=(\d+)", post_url)
    if m:
        user_id = m.group(1)

    def _cherche(motif):
        m = re.search(motif, raw, re.I)
        return _nombre(m.group(1)) if m else None

    comments = _cherche(r"([\d., \xa0 KMB]+?)\s*commentaire")
    shares = _cherche(r"([\d., \xa0 KMB]+?)\s*partage")
    likes = _cherche(r"Toutes les réactions\s*:?\s*([\d., \xa0 KMB]+)")

    images = brut.get("images", []) or []
    texte = brut.get("text", "").strip()

    return {
        "post_id": post_id,
        "username": compte,
        "user_id": user_id,
        "text": texte,
        "post_text": texte,
        "time": brut.get("time") or None,
        "likes": likes,
        "comments": comments,
        "shares": shares,
        "reactions": None,            # détail par type non extractible en statique
        "image": images[0] if images else None,
        "images": images,
        "link": brut.get("link") or None,
        "post_url": post_url or None,
        "available": True,
    }


class ScraperFacebook:
    """Contexte navigateur authentifié, réutilisable (context manager)."""

    def __init__(self, cookies=COOKIES_DEFAUT, headless=True):
        if not os.path.exists(cookies):
            raise FileNotFoundError(f"cookies.txt introuvable : {cookies}")
        self._cookies = _charger_cookies(cookies)
        if not any(c["name"] == "c_user" for c in self._cookies):
            raise ValueError("Cookie « c_user » absent : session non connectée.")
        self._headless = headless

    def __enter__(self):
        self._pw = sync_playwright().start()
        self._nav = self._pw.chromium.launch(headless=self._headless)
        self._ctx = self._nav.new_context(user_agent=UA, locale="fr-FR",
                                          viewport={"width": 1280, "height": 900})
        self._ctx.add_cookies(self._cookies)
        return self

    def __exit__(self, *exc):
        self._nav.close()
        self._pw.stop()

    # ------------------------------------------------------------------ AMIS
    def amis(self, compte, scrolls=30, progress=True):
        """Amis d'un PROFIL -> [{'nom', 'profil'}]."""
        url = f"https://www.facebook.com/{compte}/friends"
        self_url = f"https://www.facebook.com/{compte}"
        parasites = re.compile(r"(followers|abonn|j'aime|likes|\bK\b|\bM\b|·)", re.I)
        trouves = {}
        page = self._ouvrir(url)
        for i in range(scrolls):
            page.mouse.wheel(0, 3000)
            time.sleep(1.2)
            trouves.update(page.evaluate(_JS_PROFILS, _NAV))
            if progress:
                print(f"   amis : {len(trouves)} repérés", end="\r")
        page.close()
        if progress:
            print()
        return [{"nom": n, "profil": u}
                for u, n in sorted(trouves.items(), key=lambda kv: kv[1])
                if u != self_url and not parasites.search(n)]

    # ----------------------------------------------------------------- POSTS
    def posts(self, compte, scrolls=20, details=False, graphql=False,
              max_details=30, progress=True):
        """Posts d'un PROFIL/PAGE -> [dict au format documenté].

        graphql=True  : intercepte le JSON GraphQL interne -> texte + likes +
                        comments + shares + time fiables (RECOMMANDÉ).
        details=False : lecture rapide du fil (compteurs souvent None).
        details=True  : 2e passe par permalien -> vrais post_id/url/user_id.
        """
        url = f"https://www.facebook.com/{compte}"
        if graphql:
            return self._scraper_graphql(url, compte, scrolls, max_details, progress)
        if not details:
            return self._scraper_posts(url, compte, scrolls, progress)
        return self._scraper_details(url, compte, scrolls, max_details, progress)

    # ------------------------------------------------------------- RECHERCHE
    def recherche(self, requete, scrolls=20, details=False, graphql=False,
                  max_details=30, progress=True):
        """Posts d'une recherche ou d'un #hashtag -> [dict au format documenté]."""
        if requete.startswith("#"):
            url = f"https://www.facebook.com/hashtag/{requete[1:]}"
        else:
            url = f"https://www.facebook.com/search/posts/?q={quote(requete)}"
        if graphql:
            return self._scraper_graphql(url, None, scrolls, max_details, progress)
        if not details:
            return self._scraper_posts(url, None, scrolls, progress)
        return self._scraper_details(url, None, scrolls, max_details, progress)

    # --------------------------------------------------------------- internes
    def _ouvrir(self, url):
        page = self._ctx.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=60000)
        time.sleep(4)
        if any(k in page.title().lower() for k in ("connexion", "log in", "log into")):
            page.close()
            raise RuntimeError("Redirigé vers la connexion : cookies expirés.")
        return page

    def _scraper_posts(self, url, compte, scrolls, progress):
        page = self._ouvrir(url)
        vus, resultat = set(), []
        for i in range(scrolls):
            page.mouse.wheel(0, 3000)
            time.sleep(1.4)
            for brut in page.evaluate(_JS_POSTS):
                cle = brut["text"][:80]
                if cle in vus:
                    continue
                vus.add(cle)
                resultat.append(_enrichir_post(brut, compte))
            if progress:
                print(f"   posts : {len(resultat)} repérés", end="\r")
        page.close()
        if progress:
            print()
        return resultat

    def _scraper_graphql(self, url, compte, scrolls, max_posts, progress):
        """Intercepte les réponses GraphQL pendant le défilement, puis parse."""
        page = self._ctx.new_page()
        bodies = []

        def on_response(resp):
            if "/api/graphql" in resp.url or "/graphql" in resp.url:
                try:
                    bodies.append(resp.text())
                except Exception:                   # noqa: BLE001
                    pass

        page.on("response", on_response)
        page.goto(url, wait_until="domcontentloaded", timeout=60000)
        time.sleep(4)
        if any(k in page.title().lower() for k in ("connexion", "log in", "log into")):
            page.close()
            raise RuntimeError("Redirigé vers la connexion : cookies expirés.")

        for i in range(scrolls):
            page.mouse.wheel(0, 3000)
            time.sleep(1.6)
            if progress:
                print(f"   graphql — scroll {i+1}/{scrolls}, "
                      f"{len(bodies)} réponses captées", end="\r")
            # on s'arrête tôt si on a déjà assez de stories avec message
            if len([1 for b in bodies if "creation_time" in (b or "")]) > max_posts * 3:
                break
        if progress:
            print()
        page.close()
        # parsing global : permet de relier messages et compteurs (réponses séparées)
        posts = {}
        for p in _extraire_posts_graphql(_parser_bodies(bodies), compte):
            cle = p["post_id"] or p["text"][:60]
            if cle:
                anc = posts.get(cle)
                if not anc or (p["likes"] is not None and anc["likes"] is None):
                    posts[cle] = p
        return list(posts.values())[:max_posts]

    def _scraper_details(self, url, compte, scrolls, max_details, progress):
        # --- phase 1 : collecte des permaliens propres ---
        page = self._ouvrir(url)
        permaliens = []
        vus = set()
        for i in range(scrolls):
            page.mouse.wheel(0, 3000)
            time.sleep(1.4)
            for p in page.evaluate(_JS_PERMALIENS):
                if p not in vus:
                    vus.add(p)
                    permaliens.append(p)
            if progress:
                print(f"   phase 1 — {len(permaliens)} permaliens", end="\r")
            if len(permaliens) >= max_details:
                break
        page.close()
        permaliens = permaliens[:max_details]
        if progress:
            print(f"\n   phase 2 — visite de {len(permaliens)} posts…")

        # --- phase 2 : visite de chaque post ---
        resultat = []
        page = self._ctx.new_page()
        for j, lien in enumerate(permaliens, 1):
            try:
                page.goto(lien, wait_until="domcontentloaded", timeout=45000)
                try:
                    page.wait_for_selector('div[role="article"]', timeout=15000)
                except Exception:                   # noqa: BLE001
                    pass
                time.sleep(3.0)
                page.mouse.wheel(0, 800)            # déclenche le rendu de la barre d'actions
                time.sleep(1.2)
                detail = page.evaluate(_JS_DETAIL)
                resultat.append(_detail_vers_post(detail, lien, compte))
            except Exception:                       # noqa: BLE001
                resultat.append(_detail_vers_post(None, lien, compte))
            if progress:
                print(f"   phase 2 — {j}/{len(permaliens)}", end="\r")
        page.close()
        if progress:
            print()
        return resultat
