"""
PROJET DE DATA MINING — Application graphique (CustomTkinter)
============================================================
Alternative "desktop" à l'application Streamlit (projet/app.py).
Parcours guidé (wizard), thème sombre + accents vifs, grosse typographie.

    1) GRILLE DES TÂCHES   : Segmentation / Classification / Associations
    2) CHOIX DE L'ALGO     : écran qui glisse (animation translate)
    3) DONNÉES             : import CSV ou jeu de démo + prévisualisation
    4) PARAMÈTRES          : cible, split, profondeur/K/support selon l'algo
    5) LOADER              : barre de progression animée pendant le traitement
    6) RÉSULTATS           : mesures de performance + graphiques

UI : CustomTkinter.  Calculs : scikit-learn + Apriori "maison" (apriori.py).

Lancement :  uv run python projet/app_tk.py
"""

import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import customtkinter as ctk
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
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
BG        = "#0F1117"
SURFACE   = "#1A1D29"
SURFACE_2 = "#232735"
TEXT      = "#F2F3F7"
MUTED     = "#9097AB"
PRIMARY   = "#7C5CFF"
PRIMARY_H = "#6B4DE6"
LINE      = "#2A2F3D"
INK       = "#0F1117"

MINT  = "#00E0B8"
PINK  = "#FF5C8A"
SKY   = "#42C6FF"
AMBER = "#FFB020"

FAM = "DejaVu Sans"          # police lourde toujours présente sous Linux
EMO = "Noto Color Emoji"

ctk.set_appearance_mode("dark")

# polices créées après l'apparition du root (CTkFont a besoin d'un Tk vivant)
F = {}


def init_fonts():
    F["hero"]      = ctk.CTkFont(FAM, 56, "bold")
    F["title"]     = ctk.CTkFont(FAM, 40, "bold")
    F["sub"]       = ctk.CTkFont(FAM, 17)
    F["h2"]        = ctk.CTkFont(FAM, 26, "bold")
    F["card_t"]    = ctk.CTkFont(FAM, 23, "bold")
    F["card_d"]    = ctk.CTkFont(FAM, 14)
    F["metric_v"]  = ctk.CTkFont(FAM, 38, "bold")
    F["metric_l"]  = ctk.CTkFont(FAM, 15, "bold")
    F["body"]      = ctk.CTkFont(FAM, 16)
    F["body_b"]    = ctk.CTkFont(FAM, 16, "bold")
    F["btn"]       = ctk.CTkFont(FAM, 17, "bold")
    F["emoji"]     = ctk.CTkFont(EMO, 30)
    F["emoji_sm"]  = ctk.CTkFont(EMO, 20)


# --------------------------------------------------------------------------
# DESCRIPTION DES TÂCHES ET DES ALGORITHMES
# --------------------------------------------------------------------------
TACHES = {
    "Segmentation": {
        "icone": "🫧", "accent": MINT,
        "desc": "Regrouper des individus qui se ressemblent,\nsans étiquette. Découvre des profils cachés.",
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
        "anciennete": anciennete,
        "minutes_jour": minutes.round(1),
        "appels_service": appels_service,
        "forfait_intl": forfait_intl,
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
# CARTE CLIQUABLE (tâche ou algorithme)
# --------------------------------------------------------------------------
class Card(ctk.CTkFrame):
    def __init__(self, master, titre, desc, icone, accent, command, w=320, h=250):
        super().__init__(master, width=w, height=h, corner_radius=20,
                         fg_color=SURFACE, border_width=2, border_color=SURFACE)
        self.pack_propagate(False)
        self.command, self.accent = command, accent

        strip = ctk.CTkFrame(self, height=74, corner_radius=14, fg_color=accent)
        strip.pack(fill="x", padx=8, pady=8)
        strip.pack_propagate(False)
        ctk.CTkLabel(strip, text=icone, font=F["emoji"], text_color=INK).pack(
            side="left", padx=18)

        ctk.CTkLabel(self, text=titre, font=F["card_t"], text_color=TEXT,
                     wraplength=w - 40).pack(pady=(16, 8))
        ctk.CTkLabel(self, text=desc, font=F["card_d"], text_color=MUTED,
                     wraplength=w - 44, justify="center").pack(padx=16)

        self._wire(self)
        self.configure(cursor="hand2")

    def _wire(self, widget):
        widget.bind("<Button-1>", lambda e: self.command())
        widget.bind("<Enter>", self._enter)
        widget.bind("<Leave>", self._leave)
        for child in widget.winfo_children():
            self._wire(child)

    def _enter(self, _):
        self.configure(border_color=self.accent)

    def _leave(self, _):
        self.configure(border_color=SURFACE)


def button(master, text, command, color=PRIMARY, hover=PRIMARY_H,
           text_color="white", w=230, h=54):
    return ctk.CTkButton(master, text=text, command=command, fg_color=color,
                         hover_color=hover, text_color=text_color, width=w,
                         height=h, corner_radius=14, font=F["btn"])


# --------------------------------------------------------------------------
# TABLEAU (Treeview ttk stylé sombre)
# --------------------------------------------------------------------------
def make_tree(parent, height=10):
    wrap = ctk.CTkFrame(parent, fg_color=SURFACE, corner_radius=14)
    tv = ttk.Treeview(wrap, show="headings", height=height)
    vs = ttk.Scrollbar(wrap, orient="vertical", command=tv.yview)
    hs = ttk.Scrollbar(wrap, orient="horizontal", command=tv.xview)
    tv.configure(yscrollcommand=vs.set, xscrollcommand=hs.set)
    tv.grid(row=0, column=0, sticky="nsew", padx=(10, 0), pady=10)
    vs.grid(row=0, column=1, sticky="ns", pady=10)
    hs.grid(row=1, column=0, sticky="ew", padx=10)
    wrap.rowconfigure(0, weight=1)
    wrap.columnconfigure(0, weight=1)
    return wrap, tv


def fill_tree(tv, df, max_rows=200):
    tv.delete(*tv.get_children())
    tv.tag_configure("odd", background=SURFACE_2)
    tv.tag_configure("even", background=SURFACE)
    tv["columns"] = list(df.columns)
    for c in df.columns:
        tv.heading(c, text=str(c))
        tv.column(c, width=max(110, min(240, 14 * len(str(c)))), anchor="center")
    for i, (_, row) in enumerate(df.head(max_rows).iterrows()):
        tag = "odd" if i % 2 else "even"
        tv.insert("", "end", values=[row[c] for c in df.columns], tags=(tag,))


# --------------------------------------------------------------------------
# APPLICATION
# --------------------------------------------------------------------------
class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Data Mining Studio")
        self.geometry("1240x840")
        self.minsize(1100, 740)
        self.configure(fg_color=BG)
        init_fonts()
        self._style_tree()

        self.container = ctk.CTkFrame(self, fg_color=BG)
        self.container.pack(fill="both", expand=True)
        self.current = None
        self.state = {}
        self._compute_done = False
        self._compute_error = None

        self.go(self.build_home, animate=False)

    def _style_tree(self):
        s = ttk.Style()
        s.theme_use("clam")
        s.configure("Treeview", background=SURFACE, fieldbackground=SURFACE,
                    foreground=TEXT, rowheight=34, borderwidth=0,
                    font=(FAM, 13))
        s.map("Treeview", background=[("selected", PRIMARY)],
              foreground=[("selected", "white")])
        s.configure("Treeview.Heading", background=PRIMARY, foreground="white",
                    font=(FAM, 13, "bold"), borderwidth=0, relief="flat")
        s.map("Treeview.Heading", background=[("active", PRIMARY_H)])
        s.configure("Treeview", borderwidth=0)
        s.configure("TScrollbar", background=SURFACE_2, troughcolor=SURFACE,
                    borderwidth=0, arrowcolor=MUTED)

    # ---- navigation avec animation translate ----------------------------
    def go(self, builder, direction="left", animate=True):
        new = ctk.CTkFrame(self.container, fg_color=BG)
        builder(new)
        if self.current is None or not animate:
            if self.current is not None:
                self.current.destroy()
            new.place(x=0, y=0, relwidth=1, relheight=1)
            self.current = new
            return
        self._animate(self.current, new, direction)

    def _animate(self, old, new, direction):
        self.update_idletasks()
        w = self.container.winfo_width() or 1240
        start = w if direction == "left" else -w
        new.place(x=start, y=0, relwidth=1, relheight=1)
        old.place(x=0, y=0, relwidth=1, relheight=1)
        steps = 16

        def step(i):
            ease = 1 - (1 - i / steps) ** 3
            new.place_configure(x=int(start * (1 - ease)))
            old.place_configure(x=int(-start * ease))
            if i < steps:
                self.after(12, lambda: step(i + 1))
            else:
                old.destroy()
                new.place_configure(x=0)
                self.current = new

        step(1)

    def _header(self, parent, titre, sous_titre=None, retour=None):
        head = ctk.CTkFrame(parent, fg_color=BG)
        head.pack(fill="x", padx=44, pady=(30, 8))
        if retour:
            button(head, "←  Retour", retour, color=SURFACE, hover=SURFACE_2,
                   text_color=TEXT, w=140, h=46).pack(side="left", padx=(0, 22))
        box = ctk.CTkFrame(head, fg_color=BG)
        box.pack(side="left")
        ctk.CTkLabel(box, text=titre, font=F["title"], text_color=TEXT).pack(anchor="w")
        if sous_titre:
            ctk.CTkLabel(box, text=sous_titre, font=F["sub"],
                         text_color=MUTED).pack(anchor="w")

    # =====================================================================
    # ÉCRAN 1 — GRILLE DES TÂCHES
    # =====================================================================
    def build_home(self, f):
        wrap = ctk.CTkFrame(f, fg_color=BG)
        wrap.pack(expand=True)
        ctk.CTkLabel(wrap, text="Data Mining Studio", font=F["hero"],
                     text_color=TEXT).pack(pady=(30, 2))
        ctk.CTkLabel(wrap, text="Master 2  ·  choisis une tâche pour commencer",
                     font=F["sub"], text_color=MUTED).pack(pady=(0, 40))
        grid = ctk.CTkFrame(wrap, fg_color=BG)
        grid.pack()
        for i, (nom, info) in enumerate(TACHES.items()):
            Card(grid, nom, info["desc"], info["icone"], info["accent"],
                 command=lambda n=nom: self._choose_task(n)).grid(
                row=0, column=i, padx=20, pady=12)

    def _choose_task(self, nom):
        self.state = {"task": nom}
        self.go(self.build_algo)

    # =====================================================================
    # ÉCRAN 2 — CHOIX DE L'ALGORITHME
    # =====================================================================
    def build_algo(self, f):
        task = self.state["task"]
        self._header(f, "Quel algorithme ?", f"Tâche : {task}",
                     retour=lambda: self.go(self.build_home, "right"))
        zone = ctk.CTkFrame(f, fg_color=BG)
        zone.pack(expand=True)
        for i, a in enumerate(TACHES[task]["algos"]):
            Card(zone, a, ALGO_DESC[a], "✨", TACHES[task]["accent"],
                 command=lambda x=a: self._choose_algo(x)).grid(
                row=0, column=i, padx=20, pady=12)

    def _choose_algo(self, algo):
        self.state["algo"] = algo
        self.state.pop("df", None)
        self.go(self.build_data)

    # =====================================================================
    # ÉCRAN 3 — DONNÉES
    # =====================================================================
    def build_data(self, f):
        self._header(f, "Jeu de données",
                     f"{self.state['task']}  ·  {self.state['algo']}",
                     retour=lambda: self.go(self.build_algo, "right"))
        body = ctk.CTkFrame(f, fg_color=BG)
        body.pack(fill="both", expand=True, padx=44, pady=8)

        bar = ctk.CTkFrame(body, fg_color=BG)
        bar.pack(fill="x", pady=(0, 14))
        button(bar, "📁  Importer un CSV", self._import_csv, w=230).pack(
            side="left", padx=(0, 14))
        button(bar, "🎲  Jeu de démonstration", self._load_demo, color=MINT,
               hover="#15C7A6", text_color=INK, w=270).pack(side="left")

        self.lbl_source = ctk.CTkLabel(body, text="Aucune donnée chargée.",
                                       font=F["body"], text_color=MUTED)
        self.lbl_source.pack(anchor="w", pady=(2, 10))

        wrap, self.preview = make_tree(body, height=12)
        wrap.pack(fill="both", expand=True)

        foot = ctk.CTkFrame(body, fg_color=BG)
        foot.pack(fill="x", pady=14)
        self.btn_next = button(foot, "Continuer  →", self._after_data, w=210)
        self.btn_next.pack(side="right")
        self._toggle_next(False)

        if "df" in self.state:
            self._show_df(self.state["df"], self.state.get("source", ""))

    def _toggle_next(self, on):
        self.btn_next.configure(
            state="normal" if on else "disabled",
            fg_color=PRIMARY if on else SURFACE_2,
            text_color="white" if on else MUTED)

    def _show_df(self, df, source):
        self.state["df"] = df
        self.state["source"] = source
        self.lbl_source.configure(
            text=f"✓  {source}   —   {df.shape[0]} lignes × {df.shape[1]} colonnes",
            text_color=MINT)
        fill_tree(self.preview, df)
        self._toggle_next(True)

    def _import_csv(self):
        path = filedialog.askopenfilename(
            title="Choisir un fichier CSV",
            filetypes=[("CSV", "*.csv"), ("Tous", "*.*")])
        if not path:
            return
        df = None
        for sep in (",", ";", "\t"):
            try:
                tmp = pd.read_csv(path, sep=sep)
                if tmp.shape[1] > 1:
                    df = tmp
                    break
                df = tmp
            except Exception:
                pass
        if df is None or df.shape[1] == 0:
            messagebox.showerror("Erreur", "Impossible de lire ce fichier CSV.")
            return
        self._show_df(df, path.split("/")[-1])

    def _load_demo(self):
        self._show_df(DEMO[self.state["task"]](), "Jeu de démonstration")

    def _after_data(self):
        self.go(self.build_params)

    # =====================================================================
    # ÉCRAN 4 — PARAMÈTRES
    # =====================================================================
    def build_params(self, f):
        algo = self.state["algo"]
        df = self.state["df"]
        self._header(f, "Paramètres", f"{self.state['task']}  ·  {algo}",
                     retour=lambda: self.go(self.build_data, "right"))
        body = ctk.CTkFrame(f, fg_color=SURFACE, corner_radius=18)
        body.pack(fill="x", padx=70, pady=10)
        inner = ctk.CTkFrame(body, fg_color=SURFACE)
        inner.pack(fill="x", padx=36, pady=28)
        self.params = {}

        def row(label):
            r = ctk.CTkFrame(inner, fg_color=SURFACE)
            r.pack(fill="x", pady=13)
            ctk.CTkLabel(r, text=label, font=F["body_b"], text_color=TEXT,
                         width=270, anchor="w").pack(side="left")
            return r

        def slider(parent, lo, hi, init, steps, fmt="{:.0f}"):
            val = ctk.CTkLabel(parent, text=fmt.format(init), font=F["body_b"],
                               text_color=PRIMARY, width=70)
            val.pack(side="right")
            var = tk.DoubleVar(value=init)

            def on(v):
                val.configure(text=fmt.format(float(v)))
            sc = ctk.CTkSlider(parent, from_=lo, to=hi, number_of_steps=steps,
                               variable=var, command=on, progress_color=PRIMARY,
                               button_color=PRIMARY, button_hover_color=PRIMARY_H,
                               fg_color=SURFACE_2, height=18)
            sc.pack(side="left", fill="x", expand=True, padx=16)
            return var

        def menu(parent, values, default):
            var = tk.StringVar(value=default)
            ctk.CTkOptionMenu(parent, values=[str(v) for v in values],
                              variable=var, font=F["body"], fg_color=SURFACE_2,
                              button_color=PRIMARY, button_hover_color=PRIMARY_H,
                              dropdown_fg_color=SURFACE_2, dropdown_font=F["body"],
                              width=300, height=40, corner_radius=10).pack(
                side="left")
            return var

        num_cols = df.select_dtypes("number").columns.tolist()

        if algo in ("Arbre de décision", "k plus proches voisins"):
            self.params["cible"] = menu(row("Variable cible (à prédire)"),
                                        list(df.columns), df.columns[-1])
            self.params["split"] = slider(row("Part du jeu de test (%)"),
                                          10, 50, 33, 40)
            if algo == "Arbre de décision":
                self.params["depth"] = slider(row("Profondeur maximale"),
                                              1, 12, 4, 11)
            else:
                self.params["k"] = slider(row("Nombre de voisins K"), 1, 25, 5, 24)

        elif algo == "K-means":
            r = row("Variables numériques")
            ctk.CTkLabel(r, text=", ".join(num_cols) or "(aucune)", font=F["body"],
                         text_color=MUTED, wraplength=440, justify="left",
                         anchor="w").pack(side="left")
            self.params["k"] = slider(row("Nombre de clusters K"), 2, 8, 3, 6)
            self.params["norm"] = tk.BooleanVar(value=True)
            rn = row("Normaliser (z-score)")
            ctk.CTkSwitch(rn, text="", variable=self.params["norm"],
                          progress_color=MINT, button_color=TEXT).pack(side="left")

        elif algo == "Apriori":
            cols = list(df.columns)
            default = "panier" if "panier" in cols else cols[0]
            self.params["col"] = menu(row("Colonne des paniers"), cols, default)
            ctk.CTkLabel(inner, text="Articles séparés par des virgules "
                         "(ex. pain,lait,vin)", font=ctk.CTkFont(FAM, 13, "italic"),
                         text_color=MUTED).pack(anchor="w", pady=(0, 4))
            self.params["sup"] = slider(row("Support minimum"), 0.05, 1.0, 0.30,
                                        19, "{:.2f}")
            self.params["conf"] = slider(row("Confiance minimum"), 0.10, 1.0, 0.60,
                                         18, "{:.2f}")

        foot = ctk.CTkFrame(f, fg_color=BG)
        foot.pack(fill="x", padx=70, pady=18)
        button(foot, "🚀  Lancer l'analyse", self._run, w=280, h=60).pack(
            side="right")

    # =====================================================================
    # ÉCRAN 5 — LOADER + calcul en arrière-plan
    # =====================================================================
    def _run(self):
        p, algo = self.params, self.state["algo"]
        cfg = {"algo": algo}
        if algo in ("Arbre de décision", "k plus proches voisins"):
            cfg["cible"] = p["cible"].get()
            cfg["split"] = p["split"].get() / 100
            cfg["param"] = int(round(p["depth"].get())) if algo == "Arbre de décision" \
                else int(round(p["k"].get()))
        elif algo == "K-means":
            cfg["k"] = int(round(p["k"].get()))
            cfg["norm"] = p["norm"].get()
        elif algo == "Apriori":
            cfg["col"] = p["col"].get()
            cfg["sup"] = round(p["sup"].get(), 2)
            cfg["conf"] = round(p["conf"].get(), 2)
        self.state["cfg"] = cfg

        self._compute_done = False
        self._compute_error = None
        self.go(self.build_loader)
        threading.Thread(target=self._compute, daemon=True).start()
        self.after(140, self._poll)

    def build_loader(self, f):
        wrap = ctk.CTkFrame(f, fg_color=BG)
        wrap.place(relx=0.5, rely=0.5, anchor="center")
        ctk.CTkLabel(wrap, text="⚙️", font=ctk.CTkFont(EMO, 64)).pack(pady=(0, 18))
        ctk.CTkLabel(wrap, text="Traitement en cours…", font=F["h2"],
                     text_color=TEXT).pack()
        ctk.CTkLabel(wrap, text=self.state["algo"], font=F["sub"],
                     text_color=MUTED).pack(pady=(4, 22))
        self.prog = ctk.CTkProgressBar(wrap, width=360, height=14,
                                       progress_color=PRIMARY, fg_color=SURFACE_2,
                                       corner_radius=8, mode="indeterminate")
        self.prog.pack()
        self.prog.start()

    def _poll(self):
        if self._compute_done:
            if hasattr(self, "prog"):
                self.prog.stop()
            if self._compute_error:
                messagebox.showerror("Erreur de traitement", self._compute_error)
                self.go(self.build_params, "right")
            else:
                self.go(self.build_results)
            return
        self.after(140, self._poll)

    def _compute(self):
        try:
            cfg = self.state["cfg"]
            df = self.state["df"]
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
                    model = DecisionTreeClassifier(max_depth=cfg["param"],
                                                   random_state=0)
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
                    "prec": precision_score(yte, yp, average=avg, pos_label=pos,
                                            zero_division=0),
                    "rec": recall_score(yte, yp, average=avg, pos_label=pos,
                                        zero_division=0),
                    "f1": f1_score(yte, yp, average=avg, pos_label=pos,
                                   zero_division=0),
                    "cm": confusion_matrix(yte, yp, labels=classes),
                    "classes": classes, "split": cfg["split"],
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
                    "sil": silhouette_score(X, labels) if cfg["k"] > 1 else float("nan"),
                    "inertia": model.inertia_, "k": cfg["k"],
                    "proj": proj, "centers": cen, "labels": labels,
                    "sizes": dict(zip(vals.tolist(), counts.tolist())),
                })

            elif algo == "Apriori":
                transactions = [set(str(v).split(","))
                                for v in df[cfg["col"]].dropna()]
                regles = regles_association(transactions, cfg["sup"], cfg["conf"])
                res.update({"regles": regles, "n": len(regles)})

            self.state["result"] = res
        except Exception as exc:                       # noqa: BLE001
            self._compute_error = f"{type(exc).__name__}: {exc}"
        finally:
            self._compute_done = True

    # =====================================================================
    # ÉCRAN 6 — RÉSULTATS
    # =====================================================================
    def build_results(self, f):
        algo = self.state["algo"]
        self._header(f, "Résultats", f"{self.state['task']}  ·  {algo}",
                     retour=lambda: self.go(self.build_params, "right"))
        scroll = ctk.CTkScrollableFrame(f, fg_color=BG)
        scroll.pack(fill="both", expand=True, padx=44, pady=(0, 10))

        res = self.state["result"]
        if algo in ("Arbre de décision", "k plus proches voisins"):
            self._results_classif(scroll, res)
        elif algo == "K-means":
            self._results_kmeans(scroll, res)
        else:
            self._results_apriori(scroll, res)

        foot = ctk.CTkFrame(scroll, fg_color=BG)
        foot.pack(fill="x", pady=24)
        button(foot, "🏠  Nouvelle analyse", lambda: self.go(self.build_home),
               w=250).pack()

    def _metric_row(self, parent, items):
        row = ctk.CTkFrame(parent, fg_color=BG)
        row.pack(fill="x", pady=12)
        for label, value, color in items:
            card = ctk.CTkFrame(row, width=226, height=116, corner_radius=18,
                                fg_color=color)
            card.pack(side="left", padx=9)
            card.pack_propagate(False)
            ctk.CTkLabel(card, text=value, font=F["metric_v"],
                         text_color=INK).pack(pady=(22, 0))
            ctk.CTkLabel(card, text=label, font=F["metric_l"],
                         text_color=INK).pack()

    def _embed_fig(self, parent, fig, dark=True):
        if dark:
            fig.patch.set_facecolor(SURFACE)
            for ax in fig.axes:
                ax.set_facecolor(SURFACE)
                ax.title.set_color(TEXT)
                ax.title.set_fontweight("bold")
                ax.xaxis.label.set_color(MUTED)
                ax.yaxis.label.set_color(MUTED)
                ax.tick_params(colors=MUTED)
                for sp in ax.spines.values():
                    sp.set_color(LINE)
        else:
            fig.patch.set_facecolor("#F2F3F7")
        holder = ctk.CTkFrame(parent, fg_color=SURFACE if dark else "#F2F3F7",
                              corner_radius=16)
        holder.pack(pady=12)
        c = FigureCanvasTkAgg(fig, master=holder)
        c.draw()
        c.get_tk_widget().pack(padx=12, pady=12)

    def _results_classif(self, body, res):
        self._metric_row(body, [
            ("Exactitude", f"{res['acc']:.0%}", MINT),
            ("Précision", f"{res['prec']:.0%}", SKY),
            ("Rappel", f"{res['rec']:.0%}", AMBER),
            ("F1-score", f"{res['f1']:.0%}", PINK),
        ])
        ctk.CTkLabel(body, text="Découpage apprentissage / test : "
                     f"{100*(1-res['split']):.0f}% / {100*res['split']:.0f}%",
                     font=F["body"], text_color=MUTED).pack(anchor="w", pady=(2, 4))

        cm, classes = res["cm"], res["classes"]
        fig = Figure(figsize=(4.8, 4.2), dpi=100)
        ax = fig.add_subplot(111)
        ax.imshow(cm, cmap="Purples")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, str(v), ha="center", va="center", fontweight="bold",
                    color="white" if v > cm.max() / 2 else INK)
        ax.set_xticks(range(len(classes)), classes, rotation=45)
        ax.set_yticks(range(len(classes)), classes)
        ax.set_xlabel("prédit"); ax.set_ylabel("réel")
        ax.set_title("Matrice de confusion")
        fig.tight_layout()
        self._embed_fig(body, fig)

        if res.get("model") is not None:
            fig2 = Figure(figsize=(9, 5), dpi=100)
            ax2 = fig2.add_subplot(111)
            plot_tree(res["model"], feature_names=res["features"],
                      class_names=list(res["model"].classes_), filled=True,
                      rounded=True, fontsize=7, max_depth=3, ax=ax2)
            ax2.set_title("Arbre de décision (3 premiers niveaux)")
            fig2.tight_layout()
            self._embed_fig(body, fig2, dark=False)

    def _results_kmeans(self, body, res):
        self._metric_row(body, [
            ("Silhouette", f"{res['sil']:.3f}", MINT),
            ("Inertie", f"{res['inertia']:.1f}", SKY),
            ("Clusters", str(res["k"]), PINK),
        ])
        proj, cen, labels = res["proj"], res["centers"], res["labels"]
        fig = Figure(figsize=(6.6, 4.8), dpi=100)
        ax = fig.add_subplot(111)
        ax.scatter(proj[:, 0], proj[:, 1], c=labels, cmap="cool", s=48)
        ax.scatter(cen[:, 0], cen[:, 1], marker="X", s=280, c=TEXT,
                   edgecolors=INK, linewidths=1.5, label="centres")
        ax.legend(); ax.set_title(f"K-means — {res['k']} clusters (projection 2D)")
        fig.tight_layout()
        self._embed_fig(body, fig)

        fig2 = Figure(figsize=(6.6, 3), dpi=100)
        ax2 = fig2.add_subplot(111)
        ks = list(res["sizes"].keys())
        ax2.bar([str(k) for k in ks], [res["sizes"][k] for k in ks], color=PRIMARY)
        ax2.set_title("Taille des clusters"); ax2.set_xlabel("cluster")
        fig2.tight_layout()
        self._embed_fig(body, fig2)

    def _results_apriori(self, body, res):
        self._metric_row(body, [("Règles trouvées", str(res["n"]), PINK)])
        regles = res["regles"]
        if not regles:
            ctk.CTkLabel(body, text="Aucune règle : baisse le support ou la "
                         "confiance.", font=F["sub"], text_color=MUTED).pack(pady=24)
            return
        tab = pd.DataFrame([{
            "Si (antécédent)": ", ".join(sorted(r["antecedent"])),
            "Alors (conséquent)": ", ".join(sorted(r["consequent"])),
            "Support": r["support"], "Confiance": r["confidence"], "Lift": r["lift"],
        } for r in regles])
        wrap, tv = make_tree(body, height=min(12, len(tab)))
        wrap.pack(fill="x", pady=12)
        fill_tree(tv, tab)

        top = tab.head(10)
        fig = Figure(figsize=(8, 4.2), dpi=100)
        ax = fig.add_subplot(111)
        labels = [f"{a} → {b}" for a, b in
                  zip(top["Si (antécédent)"], top["Alors (conséquent)"])]
        ax.barh(range(len(top))[::-1], top["Confiance"], color=PINK)
        ax.set_yticks(range(len(top))[::-1], labels, fontsize=8)
        ax.set_xlabel("Confiance"); ax.set_title("Top règles par confiance")
        fig.tight_layout()
        self._embed_fig(body, fig)


if __name__ == "__main__":
    App().mainloop()
