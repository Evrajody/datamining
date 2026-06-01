"""
Démo : flowchart de process ISOMÉTRIQUE et INTERACTIF dans Streamlit.

Lancement :
    pip install streamlit
    streamlit run iso_flow_app.py

Principe :
- Streamlit ne sait pas dessiner en isométrique -> on injecte du HTML/JS
  via st.components.v1.html.
- Les "cartes" d'étapes sont posées sur une grille puis basculées en
  projection isométrique avec des transforms CSS 3D.
- Le JS ajoute le zoom (molette) et le pan (clic-glisser), plus un clic
  sur une étape qui renvoie l'info à Streamlit.
"""

import json
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Iso Flow", layout="wide")
st.title("Flowchart de process — isométrique & interactif")

# --- 1. Tes étapes : éditables ici, ou via les widgets Streamlit ----------
default_steps = [
    {"id": "ingest",   "label": "Ingestion",      "col": 0, "row": 0, "color": "#4f8cff"},
    {"id": "clean",    "label": "Nettoyage",      "col": 1, "row": 0, "color": "#4f8cff"},
    {"id": "transform","label": "Transformation", "col": 2, "row": 1, "color": "#7c5cff"},
    {"id": "load",     "label": "Chargement DWH", "col": 3, "row": 1, "color": "#2ec28b"},
    {"id": "report",   "label": "Reporting",      "col": 4, "row": 2, "color": "#ff9f43"},
]
edges = [("ingest", "clean"), ("clean", "transform"),
         ("transform", "load"), ("load", "report")]

with st.sidebar:
    st.subheader("Paramètres")
    tile = st.slider("Taille des tuiles (px)", 80, 200, 120)
    gap = st.slider("Espacement", 1.0, 2.5, 1.6, 0.1)

steps_json = json.dumps(default_steps)
edges_json = json.dumps(edges)

# --- 2. Le rendu isométrique (HTML + CSS 3D + JS pan/zoom) ----------------
html = f"""
<div id="viewport" style="width:100%;height:640px;overflow:hidden;
     background:#0e1117;border-radius:12px;cursor:grab;position:relative;">
  <div id="scene" style="position:absolute;top:50%;left:50%;
       transform-style:preserve-3d;
       transform:rotateX(60deg) rotateZ(-45deg) scale(1);"></div>
</div>

<script>
const TILE = {tile};
const GAP  = {gap};
const steps = {steps_json};
const edges = {edges_json};
const scene = document.getElementById("scene");
const viewport = document.getElementById("viewport");

// Position d'une étape en coordonnées "monde" (avant projection iso)
function pos(s) {{
  return {{ x: s.col * TILE * GAP, y: s.row * TILE * GAP }};
}}

// Cartes
steps.forEach(s => {{
  const p = pos(s);
  const card = document.createElement("div");
  card.style.cssText = `position:absolute;width:${{TILE}}px;height:${{TILE}}px;
    left:${{p.x}}px;top:${{p.y}}px;
    background:${{s.color}};border-radius:10px;
    box-shadow:0 ${{TILE*0.25}}px 0 rgba(0,0,0,0.35);
    display:flex;align-items:center;justify-content:center;
    color:#fff;font:600 14px sans-serif;text-align:center;
    transform:translateZ(0);transition:transform .15s;cursor:pointer;`;
  card.textContent = s.label;
  card.onmouseenter = () => card.style.transform = "translateZ(30px)";
  card.onmouseleave = () => card.style.transform = "translateZ(0)";
  card.onclick = () => {{
    if (window.parent) window.parent.postMessage(
      {{type:"streamlit:setComponentValue", value:s.id}}, "*");
    alert("Étape sélectionnée : " + s.label);
  }};
  scene.appendChild(card);
}});

// Flèches (lignes SVG entre centres de tuiles, dans le plan de la grille)
const svgNS = "http://www.w3.org/2000/svg";
const svg = document.createElementNS(svgNS, "svg");
svg.style.cssText = "position:absolute;overflow:visible;left:0;top:0;";
scene.appendChild(svg);
const byId = Object.fromEntries(steps.map(s => [s.id, s]));
edges.forEach(([a, b]) => {{
  const pa = pos(byId[a]), pb = pos(byId[b]);
  const line = document.createElementNS(svgNS, "line");
  line.setAttribute("x1", pa.x + TILE/2); line.setAttribute("y1", pa.y + TILE/2);
  line.setAttribute("x2", pb.x + TILE/2); line.setAttribute("y2", pb.y + TILE/2);
  line.setAttribute("stroke", "#5b6472");
  line.setAttribute("stroke-width", "4");
  svg.appendChild(line);
}});

// --- Pan + zoom ---
let scale = 0.8, ox = 0, oy = 0, dragging = false, sx, sy;
function apply() {{
  scene.style.transform =
    `translate(${{ox}}px,${{oy}}px) rotateX(60deg) rotateZ(-45deg) scale(${{scale}})`;
}}
apply();
viewport.addEventListener("wheel", e => {{
  e.preventDefault();
  scale *= e.deltaY < 0 ? 1.1 : 0.9;
  scale = Math.min(Math.max(scale, 0.2), 3);
  apply();
}}, {{passive:false}});
viewport.addEventListener("mousedown", e => {{
  dragging = true; sx = e.clientX - ox; sy = e.clientY - oy;
  viewport.style.cursor = "grabbing";
}});
window.addEventListener("mouseup", () => {{
  dragging = false; viewport.style.cursor = "grab";
}});
window.addEventListener("mousemove", e => {{
  if (!dragging) return;
  ox = e.clientX - sx; oy = e.clientY - sy; apply();
}});
</script>
"""

components.html(html, height=660)

st.caption("Molette = zoom · clic-glissé = déplacement · clic sur une étape = sélection")
