"""
PROJET DE DATA MINING — Application web (NiceGUI)
=================================================
Alternative moderne à l'app Streamlit / Tkinter : interface web rendue dans un
navigateur (ou fenêtre native), écrite 100% en Python.

Parcours guidé :
    1) GRILLE DES TÂCHES   : Segmentation / Classification / Associations
    2) CHOIX DE L'ALGO     : cartes avec animation d'entrée
    3) DONNÉES             : upload CSV ou jeu de démo + prévisualisation
    4) PARAMÈTRES          : cible, split, profondeur/K/support selon l'algo
    5) LOADER              : spinner pendant le traitement (thread non bloquant)
    6) RÉSULTATS           : métriques + graphiques ECharts interactifs

Calculs : scikit-learn + Apriori "maison" (apriori.py), réutilisés tels quels.

Lancement :  uv run python projet/app_web.py
            (ouvre http://localhost:8080 ; mettre native=True pour une fenêtre)
"""

import io
import json

import numpy as np
import pandas as pd
from nicegui import run, ui
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

from apriori import regles_association

# --------------------------------------------------------------------------
# THÈME SOMBRE + ACCENTS VIFS
# --------------------------------------------------------------------------
BG      = "#0F1117"
SURFACE = "#1A1D29"
TEXT    = "#F2F3F7"
MUTED   = "#9097AB"
PRIMARY = "#7C5CFF"
INK     = "#0F1117"
MINT    = "#00E0B8"
PINK    = "#FF5C8A"
SKY     = "#42C6FF"
AMBER   = "#FFB020"
PALETTE = [MINT, PINK, SKY, AMBER, PRIMARY, "#9B6BFF", "#2ECC71", "#E67E22"]

TACHES = {
    "Segmentation": {
        "icone": "🫧", "accent": MINT,
        "desc": "Regrouper des individus qui se ressemblent,\nsans étiquette. Profils cachés.",
        "algos": ["K-means"],
    },
    "Classification": {
        "icone": "🌳", "accent": PRIMARY,
        "desc": "Prédire une classe connue à partir\nd'exemples étiquetés (supervisé).",
        "algos": ["Arbre de décision", "k plus proches voisins"],
    },
    "Recherche d'associations": {
        "icone": "🛒", "accent": PINK,
        "desc": "Trouver des règles « si A alors B »\ndans des transactions (Apriori).",
        "algos": ["Apriori"],
    },
}
ALGO_DESC = {
    "K-means": "Partitionne en K groupes\nautour de centres mobiles.",
    "Arbre de décision": "Suite de questions oui/non\nsur les attributs.",
    "k plus proches voisins": "Vote des K voisins\nles plus proches.",
    "Apriori": "Itemsets fréquents puis règles\nsupport / confiance.",
}


# --------------------------------------------------------------------------
# JEUX DE DÉMONSTRATION
# --------------------------------------------------------------------------
def demo_segmentation():
    rng = np.random.default_rng(0)
    g1 = rng.normal([110, 1, 8], [10, 0.5, 1.5], (25, 3))
    g2 = rng.normal([150, 5, 2], [12, 1.0, 1.0], (25, 3))
    g3 = rng.normal([90, 3, 14], [8, 0.8, 2.0], (25, 3))
    X = np.vstack([g1, g2, g3])
    return pd.DataFrame(X, columns=["calories", "proteines", "fibres"]).round(1)


def demo_classification():
    rng = np.random.default_rng(1)
    n = 300
    anciennete = rng.integers(1, 72, n)
    minutes = rng.normal(180, 60, n).clip(0)
    appels_service = rng.poisson(1.5, n)
    forfait_intl = rng.integers(0, 2, n)
    proba = 1 / (1 + np.exp(-(0.06 * appels_service * 2 + 0.02 * (12 - anciennete) - 0.5)))
    churn = (rng.random(n) < proba).astype(int)
    return pd.DataFrame({
        "anciennete": anciennete, "minutes_jour": minutes.round(1),
        "appels_service": appels_service, "forfait_intl": forfait_intl,
        "churn": np.where(churn == 1, "oui", "non"),
    })


def demo_association():
    paniers = [
        "pain,lait,vin", "pain,lait", "vin,fromage", "pain,vin,fromage",
        "pain,lait,vin,fromage", "lait,fromage", "pain,vin", "vin,fromage,oeufs",
        "pain,lait,oeufs", "vin,fromage", "pain,vin,fromage,oeufs", "lait,oeufs",
        "pain,lait,vin", "vin,fromage", "pain,lait,fromage",
    ]
    return pd.DataFrame({"panier": paniers})


DEMO = {
    "Segmentation": demo_segmentation,
    "Classification": demo_classification,
    "Recherche d'associations": demo_association,
}


# --------------------------------------------------------------------------
# CALCUL (pur, sans UI — exécuté dans un thread via run.io_bound)
# --------------------------------------------------------------------------
def compute(cfg, df):
    algo = cfg["algo"]
    res = {"algo": algo}

    if algo in ("Arbre de décision", "k plus proches voisins"):
        cible = cfg["cible"]
        feats = [c for c in df.columns if c != cible]
        X = pd.get_dummies(df[feats])
        y = df[cible].astype(str)
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=cfg["split"], random_state=0, stratify=y)
        if algo == "Arbre de décision":
            model = DecisionTreeClassifier(max_depth=cfg["param"], random_state=0)
        else:
            sc = StandardScaler().fit(Xtr)
            Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
            model = KNeighborsClassifier(n_neighbors=cfg["param"])
        model.fit(Xtr, ytr)
        yp = model.predict(Xte)
        classes = sorted(y.unique())
        avg = "binary" if len(classes) == 2 else "macro"
        pos = classes[-1] if len(classes) == 2 else None
        res.update({
            "acc": accuracy_score(yte, yp),
            "prec": precision_score(yte, yp, average=avg, pos_label=pos, zero_division=0),
            "rec": recall_score(yte, yp, average=avg, pos_label=pos, zero_division=0),
            "f1": f1_score(yte, yp, average=avg, pos_label=pos, zero_division=0),
            "cm": confusion_matrix(yte, yp, labels=classes).tolist(),
            "classes": [str(c) for c in classes], "split": cfg["split"],
            "model": model if algo == "Arbre de décision" else None,
            "features": list(X.columns),
        })

    elif algo == "K-means":
        cols = df.select_dtypes("number").columns.tolist()
        X = df[cols].to_numpy()
        if cfg["norm"]:
            X = StandardScaler().fit_transform(X)
        model = KMeans(n_clusters=cfg["k"], n_init=10, random_state=0)
        labels = model.fit_predict(X)
        proj = PCA(2).fit_transform(X) if X.shape[1] > 2 else X[:, :2]
        cen = (PCA(2).fit(X).transform(model.cluster_centers_)
               if X.shape[1] > 2 else model.cluster_centers_[:, :2])
        vals, counts = np.unique(labels, return_counts=True)
        res.update({
            "sil": float(silhouette_score(X, labels)) if cfg["k"] > 1 else float("nan"),
            "inertia": float(model.inertia_), "k": cfg["k"],
            "proj": proj.tolist(), "centers": cen.tolist(),
            "labels": labels.tolist(),
            "sizes": {int(a): int(b) for a, b in zip(vals, counts)},
        })

    elif algo == "Apriori":
        transactions = [set(str(v).split(",")) for v in df[cfg["col"]].dropna()]
        regles = regles_association(transactions, cfg["sup"], cfg["conf"])
        res.update({"regles": regles, "n": len(regles)})

    return res


# --------------------------------------------------------------------------
# OPTIONS ECHARTS (graphes interactifs, stylés sombre)
# --------------------------------------------------------------------------
def _axis(name=""):
    return {"nameTextStyle": {"color": MUTED}, "name": name,
            "axisLabel": {"color": MUTED}, "axisLine": {"lineStyle": {"color": "#3A4054"}},
            "splitLine": {"lineStyle": {"color": "#242938"}}}


def opt_confusion(cm, classes):
    data = [[j, i, cm[i][j]] for i in range(len(classes)) for j in range(len(classes))]
    mx = max(max(r) for r in cm)
    return {
        "backgroundColor": "transparent",
        "tooltip": {"position": "top"},
        "grid": {"height": "66%", "top": "6%", "left": "14%"},
        "xAxis": {"type": "category", "data": classes, "name": "prédit", **_axis("prédit")},
        "yAxis": {"type": "category", "data": classes, "name": "réel", **_axis("réel")},
        "visualMap": {"min": 0, "max": int(mx), "calculable": True,
                      "orient": "horizontal", "left": "center", "bottom": "2%",
                      "inRange": {"color": [SURFACE, PRIMARY]},
                      "textStyle": {"color": MUTED}},
        "series": [{"type": "heatmap", "data": data,
                    "label": {"show": True, "color": "#fff", "fontWeight": "bold"},
                    "emphasis": {"itemStyle": {"shadowBlur": 10}}}],
    }


def opt_scatter(proj, labels, centers, k):
    series = []
    for c in range(k):
        pts = [proj[i] for i in range(len(labels)) if labels[i] == c]
        series.append({"name": f"cluster {c}", "type": "scatter", "symbolSize": 13,
                       "itemStyle": {"color": PALETTE[c % len(PALETTE)]}, "data": pts})
    series.append({"name": "centres", "type": "scatter", "symbol": "diamond",
                   "symbolSize": 26, "itemStyle": {"color": "#fff", "borderColor": "#000",
                   "borderWidth": 2}, "data": centers})
    return {"backgroundColor": "transparent", "tooltip": {},
            "legend": {"textStyle": {"color": MUTED}, "top": "2%"},
            "xAxis": _axis(), "yAxis": _axis(), "series": series}


def opt_bar(cats, vals, color):
    return {"backgroundColor": "transparent", "tooltip": {},
            "xAxis": {"type": "category", "data": cats, **_axis()},
            "yAxis": {"type": "value", **_axis()},
            "series": [{"type": "bar", "data": vals,
                        "itemStyle": {"color": color, "borderRadius": [6, 6, 0, 0]}}]}


def opt_barh(cats, vals, color):
    return {"backgroundColor": "transparent", "tooltip": {},
            "grid": {"left": "32%", "right": "6%"},
            "xAxis": {"type": "value", **_axis()},
            "yAxis": {"type": "category", "data": cats, "axisLabel": {"color": MUTED,
                      "fontSize": 10}, "axisLine": {"lineStyle": {"color": "#3A4054"}}},
            "series": [{"type": "bar", "data": vals,
                        "itemStyle": {"color": color, "borderRadius": [0, 6, 6, 0]}}]}


# --------------------------------------------------------------------------
# HELPERS UI
# --------------------------------------------------------------------------
def table_for(df, n=200, pag=10):
    cols = [{"name": str(c), "label": str(c), "field": str(c), "align": "center"}
            for c in df.columns]
    rows = json.loads(df.head(n).to_json(orient="records"))
    for i, r in enumerate(rows):
        r["id"] = i
    ui.table(columns=cols, rows=rows, row_key="id", pagination=pag).props(
        "flat dark").classes("w-full max-w-5xl").style(
        f"background:{SURFACE};border-radius:14px")


def metric(value, label, color):
    with ui.card().classes("rounded-2xl items-center justify-center").style(
            f"background:{color};width:210px;height:118px;box-shadow:0 8px 26px rgba(0,0,0,.35)"):
        ui.label(value).style(f"color:{INK};font-size:36px;font-weight:800;line-height:1")
        ui.label(label).style(f"color:{INK};font-size:15px;font-weight:700")


def chart_card(title, options, height=360):
    with ui.card().classes("rounded-2xl w-full max-w-3xl").style(
            f"background:{SURFACE};box-shadow:0 8px 26px rgba(0,0,0,.25)"):
        ui.label(title).style(f"color:{TEXT};font-weight:700;font-size:18px")
        ui.echart(options).classes("w-full").style(f"height:{height}px")


def select_card(title, desc, icon, accent, on_click):
    card = ui.card().classes(
        "rounded-2xl cursor-pointer transition-transform hover:scale-105 p-0 "
        "overflow-hidden").style(
        f"background:{SURFACE};width:300px;height:236px;box-shadow:0 10px 30px rgba(0,0,0,.35)")
    with card:
        with ui.element("div").style(
                f"background:{accent};height:72px;width:100%;display:flex;"
                "align-items:center;padding-left:20px"):
            ui.label(icon).style("font-size:32px")
        with ui.column().classes("p-5 items-center w-full gap-2"):
            ui.label(title).style(
                f"color:{TEXT};font-size:22px;font-weight:800;text-align:center")
            ui.label(desc).style(
                f"color:{MUTED};font-size:13px;text-align:center;white-space:pre-line")
    card.on("click", on_click)


# ==========================================================================
# APPLICATION (une instance par client)
# ==========================================================================
class Studio:
    def __init__(self):
        ui.dark_mode().enable()
        ui.colors(primary=PRIMARY)
        ui.query("body").style(f"background:{BG}")
        ui.add_css("""
            @keyframes slidein { from {opacity:0; transform:translateX(46px);}
                                 to {opacity:1; transform:none;} }
            .screen { animation: slidein .34s cubic-bezier(.2,.7,.3,1); }
        """)
        self.state = {}
        self.body = ui.column().classes("w-full items-center").style("min-height:100vh")
        self.show(self.home)

    # ---- gestion des écrans -------------------------------------------
    def show(self, builder):
        self.body.clear()
        with self.body:
            with ui.column().classes("screen w-full items-center gap-3"):
                builder()

    def header(self, title, sub=None, back=None):
        with ui.row().classes("w-full items-center gap-4 px-10 pt-4"):
            if back:
                ui.button("← Retour", on_click=back).props("flat no-caps").style(
                    f"color:{TEXT};font-weight:700")
            with ui.column().classes("gap-0"):
                ui.label(title).style(f"color:{TEXT};font-size:34px;font-weight:800")
                if sub:
                    ui.label(sub).style(f"color:{MUTED};font-size:16px")

    # ---- 1. accueil ----------------------------------------------------
    def home(self):
        ui.label("Data Mining Studio").classes("mt-10").style(
            "font-size:56px;font-weight:800;background:linear-gradient(90deg,"
            f"{PRIMARY},{MINT});-webkit-background-clip:text;background-clip:text;"
            "-webkit-text-fill-color:transparent")
        ui.label("Master 2  ·  choisis une tâche pour commencer").classes("mb-8").style(
            f"color:{MUTED};font-size:18px")
        with ui.row().classes("gap-6 flex-wrap justify-center"):
            for nom, info in TACHES.items():
                select_card(nom, info["desc"], info["icone"], info["accent"],
                            lambda n=nom: self.choose_task(n))

    def choose_task(self, nom):
        self.state = {"task": nom}
        self.show(self.algo)

    # ---- 2. algorithme -------------------------------------------------
    def algo(self):
        task = self.state["task"]
        self.header("Quel algorithme ?", f"Tâche : {task}",
                    back=lambda: self.show(self.home))
        with ui.row().classes("gap-6 flex-wrap justify-center mt-6"):
            for a in TACHES[task]["algos"]:
                select_card(a, ALGO_DESC[a], "✨", TACHES[task]["accent"],
                            lambda x=a: self.choose_algo(x))

    def choose_algo(self, a):
        self.state["algo"] = a
        self.state.pop("df", None)
        self.show(self.data)

    # ---- 3. données ----------------------------------------------------
    def data(self):
        self.header("Jeu de données", f"{self.state['task']}  ·  {self.state['algo']}",
                    back=lambda: self.show(self.algo))
        with ui.column().classes("w-full items-center gap-4 px-10 mt-2"):
            with ui.row().classes("gap-4 items-start"):
                ui.upload(label="Importer un CSV", auto_upload=True,
                          on_upload=self.on_upload).props("accept=.csv").classes(
                    "max-w-xs").style(f"background:{SURFACE};border-radius:12px")
                ui.button("🎲  Jeu de démonstration", on_click=self.load_demo).props(
                    "no-caps").style(
                    f"background:{MINT};color:{INK};font-weight:700;height:56px")
            self.src_label = ui.label("Aucune donnée chargée.").style(f"color:{MUTED}")
            self.prev = ui.column().classes("w-full items-center")
            self.next_btn = ui.button("Continuer  →",
                                      on_click=lambda: self.show(self.params)).props(
                "no-caps").style("font-weight:700;align-self:flex-end")
            self.next_btn.set_enabled(False)
        if "df" in self.state:
            self._set_df(self.state["df"], self.state.get("source", ""), store=False)

    def on_upload(self, e):
        raw = e.content.read()
        df = None
        for sep in (",", ";", "\t"):
            try:
                tmp = pd.read_csv(io.BytesIO(raw), sep=sep)
                df = tmp
                if tmp.shape[1] > 1:
                    break
            except Exception:
                pass
        if df is None or df.shape[1] == 0:
            ui.notify("CSV illisible.", type="negative")
            return
        self._set_df(df, e.name)

    def load_demo(self):
        self._set_df(DEMO[self.state["task"]](), "Jeu de démonstration")

    def _set_df(self, df, source, store=True):
        if store:
            self.state["df"] = df
            self.state["source"] = source
        self.src_label.set_text(
            f"✓  {source}   —   {df.shape[0]} lignes × {df.shape[1]} colonnes")
        self.src_label.style(f"color:{MINT}")
        self.prev.clear()
        with self.prev:
            table_for(df)
        self.next_btn.set_enabled(True)

    # ---- 4. paramètres -------------------------------------------------
    def params(self):
        algo, df = self.state["algo"], self.state["df"]
        self.header("Paramètres", f"{self.state['task']}  ·  {algo}",
                    back=lambda: self.show(self.data))
        self.ctrl = {}
        with ui.card().classes("rounded-2xl w-full max-w-3xl mt-2").style(
                f"background:{SURFACE}"):
            with ui.column().classes("w-full gap-5 p-6"):
                num_cols = df.select_dtypes("number").columns.tolist()

                if algo in ("Arbre de décision", "k plus proches voisins"):
                    self.ctrl["cible"] = self._select(
                        "Variable cible (à prédire)", list(df.columns), df.columns[-1])
                    self.ctrl["split"] = self._slider(
                        "Part du jeu de test (%)", 10, 50, 1, 33)
                    if algo == "Arbre de décision":
                        self.ctrl["depth"] = self._slider("Profondeur maximale", 1, 12, 1, 4)
                    else:
                        self.ctrl["k"] = self._slider("Nombre de voisins K", 1, 25, 1, 5)

                elif algo == "K-means":
                    with ui.row().classes("w-full items-center gap-4"):
                        ui.label("Variables numériques").style(
                            f"color:{TEXT};font-weight:700;width:240px")
                        ui.label(", ".join(num_cols) or "(aucune)").style(
                            f"color:{MUTED}")
                    self.ctrl["k"] = self._slider("Nombre de clusters K", 2, 8, 1, 3)
                    with ui.row().classes("w-full items-center gap-4"):
                        ui.label("Normaliser (z-score)").style(
                            f"color:{TEXT};font-weight:700;width:240px")
                        self.ctrl["norm"] = ui.switch(value=True)

                elif algo == "Apriori":
                    cols = list(df.columns)
                    default = "panier" if "panier" in cols else cols[0]
                    self.ctrl["col"] = self._select("Colonne des paniers", cols, default)
                    ui.label("Articles séparés par des virgules (ex. pain,lait,vin)").style(
                        f"color:{MUTED};font-style:italic;font-size:13px")
                    self.ctrl["sup"] = self._slider("Support minimum", 0.05, 1.0, 0.05,
                                                    0.30, "{:.2f}")
                    self.ctrl["conf"] = self._slider("Confiance minimum", 0.10, 1.0, 0.05,
                                                     0.60, "{:.2f}")

        with ui.row().classes("w-full max-w-3xl justify-end mt-3"):
            ui.button("🚀  Lancer l'analyse", on_click=self.run_analysis).props(
                "no-caps").style("font-weight:800;font-size:16px;height:56px")

    def _slider(self, label, mn, mx, step, val, fmt="{:.0f}"):
        with ui.row().classes("w-full items-center gap-4"):
            ui.label(label).style(f"color:{TEXT};font-weight:700;width:240px")
            s = ui.slider(min=mn, max=mx, step=step, value=val).classes("flex-grow")
            ui.label().bind_text_from(s, "value",
                                      backward=lambda v: fmt.format(v or 0)).style(
                f"color:{PRIMARY};font-weight:800;width:64px;text-align:right")
        return s

    def _select(self, label, options, default):
        with ui.row().classes("w-full items-center gap-4"):
            ui.label(label).style(f"color:{TEXT};font-weight:700;width:240px")
            s = ui.select(options=[str(o) for o in options], value=str(default)).props(
                "outlined dark").classes("flex-grow")
        return s

    # ---- 5. loader + calcul -------------------------------------------
    async def run_analysis(self):
        algo = self.state["algo"]
        c = self.ctrl
        cfg = {"algo": algo}
        if algo in ("Arbre de décision", "k plus proches voisins"):
            cfg["cible"] = c["cible"].value
            cfg["split"] = c["split"].value / 100
            cfg["param"] = int(c["depth"].value) if algo == "Arbre de décision" \
                else int(c["k"].value)
        elif algo == "K-means":
            cfg["k"] = int(c["k"].value)
            cfg["norm"] = bool(c["norm"].value)
        elif algo == "Apriori":
            cfg["col"] = c["col"].value
            cfg["sup"] = round(c["sup"].value, 2)
            cfg["conf"] = round(c["conf"].value, 2)

        self.show(self.loader)
        try:
            res = await run.io_bound(compute, cfg, self.state["df"])
        except Exception as exc:                       # noqa: BLE001
            ui.notify(f"Erreur de traitement : {exc}", type="negative")
            self.show(self.params)
            return
        self.state["result"] = res
        self.show(self.results)

    def loader(self):
        with ui.column().classes("items-center justify-center w-full").style(
                "min-height:58vh"):
            ui.spinner(size="80px").props("color=primary")
            ui.label("Traitement en cours…").classes("mt-6").style(
                f"color:{TEXT};font-size:26px;font-weight:800")
            ui.label(self.state["algo"]).style(f"color:{MUTED};font-size:16px")

    # ---- 6. résultats --------------------------------------------------
    def results(self):
        algo, res = self.state["algo"], self.state["result"]
        self.header("Résultats", f"{self.state['task']}  ·  {algo}",
                    back=lambda: self.show(self.params))
        with ui.column().classes("w-full items-center gap-5 px-10 pb-10"):
            if algo in ("Arbre de décision", "k plus proches voisins"):
                self._res_classif(res)
            elif algo == "K-means":
                self._res_kmeans(res)
            else:
                self._res_apriori(res)
            ui.button("🏠  Nouvelle analyse", on_click=lambda: self.show(self.home)).props(
                "no-caps").classes("mt-2").style("font-weight:700")

    def _res_classif(self, res):
        with ui.row().classes("gap-4 flex-wrap justify-center"):
            metric(f"{res['acc']:.0%}", "Exactitude", MINT)
            metric(f"{res['prec']:.0%}", "Précision", SKY)
            metric(f"{res['rec']:.0%}", "Rappel", AMBER)
            metric(f"{res['f1']:.0%}", "F1-score", PINK)
        ui.label(f"Découpage apprentissage / test : "
                 f"{100*(1-res['split']):.0f}% / {100*res['split']:.0f}%").style(
            f"color:{MUTED}")
        chart_card("Matrice de confusion",
                   opt_confusion(res["cm"], res["classes"]), height=380)
        if res.get("model") is not None:
            with ui.card().classes("rounded-2xl w-full max-w-4xl").style(
                    "background:#F2F3F7"):
                ui.label("Arbre de décision (3 premiers niveaux)").style(
                    "color:#1B1A17;font-weight:700;font-size:18px")
                with ui.pyplot(figsize=(9, 5)):
                    import matplotlib.pyplot as plt
                    plot_tree(res["model"], feature_names=res["features"],
                              class_names=list(res["model"].classes_), filled=True,
                              rounded=True, fontsize=7, max_depth=3, ax=plt.gca())

    def _res_kmeans(self, res):
        with ui.row().classes("gap-4 flex-wrap justify-center"):
            metric(f"{res['sil']:.3f}", "Silhouette", MINT)
            metric(f"{res['inertia']:.1f}", "Inertie", SKY)
            metric(str(res["k"]), "Clusters", PINK)
        chart_card(f"K-means — {res['k']} clusters (projection 2D)",
                   opt_scatter(res["proj"], res["labels"], res["centers"], res["k"]),
                   height=420)
        ks = sorted(res["sizes"])
        chart_card("Taille des clusters",
                   opt_bar([str(k) for k in ks], [res["sizes"][k] for k in ks], PRIMARY),
                   height=300)

    def _res_apriori(self, res):
        with ui.row().classes("justify-center"):
            metric(str(res["n"]), "Règles trouvées", PINK)
        regles = res["regles"]
        if not regles:
            ui.label("Aucune règle : baisse le support ou la confiance.").style(
                f"color:{MUTED};font-size:18px")
            return
        tab = pd.DataFrame([{
            "Si (antécédent)": ", ".join(sorted(r["antecedent"])),
            "Alors (conséquent)": ", ".join(sorted(r["consequent"])),
            "Support": r["support"], "Confiance": r["confidence"], "Lift": r["lift"],
        } for r in regles])
        table_for(tab, pag=12)
        top = tab.head(10)
        cats = [f"{a} → {b}" for a, b in
                zip(top["Si (antécédent)"], top["Alors (conséquent)"])]
        chart_card("Top règles par confiance",
                   opt_barh(cats[::-1], list(top["Confiance"])[::-1], PINK), height=400)


@ui.page("/")
def index():
    Studio()


if __name__ in {"__main__", "__mp_main__"}:
    ui.run(title="Data Mining Studio", dark=True, reload=False, port=8080,
           native=False)
