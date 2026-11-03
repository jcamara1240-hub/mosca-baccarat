"""
Visor 3D del cerebro REAL de Kenyon cells de FlyWire.
Muestra las 5177 neuronas con sus posiciones reales y las conexiones
más fuertes entre ellas.

Por ahora solo visualiza, no juega al baccarat.
En la siguiente fase, cuando llegue el acceso CAVE, mapearemos
grupos de Kenyon a Player/Banker/Tie y la mosca jugará con cerebro real.
"""

import pickle
import numpy as np
import pyvista as pv
from pyvistaqt import BackgroundPlotter
from PyQt5.QtWidgets import QApplication
import sys


# ---------- Cargar los datos ----------
print("Cargando cerebro_full.pkl...")
with open("cerebro_full.pkl", "rb") as f:
    datos = pickle.load(f)

ids = datos["kenyon_ids"]
coords = datos["coords"]
subtipos = datos["subtipos"]
kc = set(ids)
conex = datos["conexiones"]
conex = conex[conex["pre"].isin(kc) & conex["post"].isin(kc)]   # solo Kenyon -> Kenyon
print(f"  {len(ids)} neuronas, {len(conex):,} sinapsis")

# ---------- Normalizar coordenadas ----------
# Las coords de FlyWire están en nanómetros, hay que escalar a algo manejable
posiciones = np.array([coords[i] for i in ids if i in coords])
centro = posiciones.mean(axis=0)
escala = posiciones.std()
pos_norm = {}
for i in ids:
    if i in coords:
        p = (np.array(coords[i]) - centro) / escala * 10
        pos_norm[i] = p

# ---------- Colores por subtipo ----------
COLORES_SUB = {
    "KCg-m":   (0.98, 0.50, 0.30),   # naranja
    "KCab":    (0.40, 0.75, 1.00),   # azul
    "KCab-p":  (0.30, 0.55, 0.90),
    "KCa'b'":  (0.70, 0.50, 1.00),   # violeta
    "KCapbp-m":(0.80, 0.40, 0.90),
    "KCg-s1":  (0.95, 0.85, 0.30),
    "KCg-s2":  (0.95, 0.90, 0.20),
}
DEFAULT_COLOR = (0.75, 0.75, 0.80)

def color_de(subtipo):
    for k, c in COLORES_SUB.items():
        if subtipo.startswith(k):
            return c
    return DEFAULT_COLOR


# ---------- Construir la nube de puntos (más rápido que esferas sueltas) ----------
puntos = []
colores_puntos = []
for i in ids:
    if i not in pos_norm:
        continue
    puntos.append(pos_norm[i])
    colores_puntos.append(color_de(subtipos.get(i, "")))
puntos = np.array(puntos)
colores_puntos = (np.array(colores_puntos) * 255).astype(np.uint8)

print(f"  posiciones listas: {len(puntos)}")


# ---------- Elegir conexiones a dibujar ----------
# Nos quedamos con las más fuertes para no saturar la escena
UMBRAL = 5  # al menos 5 sinapsis para dibujar la conexión
conex_fuertes = conex[conex["syn_count"] >= UMBRAL]
# Y limitamos el número total que mostramos
MAX_LINEAS = 3000
if len(conex_fuertes) > MAX_LINEAS:
    conex_fuertes = conex_fuertes.nlargest(MAX_LINEAS, "syn_count")
print(f"  conexiones a dibujar: {len(conex_fuertes):,} (umbral {UMBRAL} sinapsis)")


# ---------- Montar la ventana ----------
qapp = QApplication.instance() or QApplication(sys.argv)
plotter = BackgroundPlotter(
    title=f"Cerebro Kenyon real de FlyWire — {len(puntos)} neuronas",
    window_size=(1200, 900),
)
plotter.set_background("#0a0a12")

# Dibujar las neuronas como una sola nube de puntos (súper rápido)
pc = pv.PolyData(puntos)
pc["colores"] = colores_puntos
plotter.add_points(
    pc,
    scalars="colores",
    rgb=True,
    point_size=6,
    render_points_as_spheres=True,
)

# Dibujar las conexiones como líneas finitas
if len(conex_fuertes) > 0:
    print("  construyendo líneas...")
    puntos_lineas = []
    indices_lineas = []
    k = 0
    for _, row in conex_fuertes.iterrows():
        pre, post = int(row["pre"]), int(row["post"])
        if pre not in pos_norm or post not in pos_norm:
            continue
        puntos_lineas.append(pos_norm[pre])
        puntos_lineas.append(pos_norm[post])
        indices_lineas.extend([2, k, k + 1])
        k += 2
    if puntos_lineas:
        puntos_lineas = np.array(puntos_lineas)
        lineas = pv.PolyData(puntos_lineas)
        lineas.lines = np.array(indices_lineas)
        plotter.add_mesh(
            lineas,
            color=(0.3, 0.4, 0.6),
            opacity=0.15,
            line_width=1,
        )

# HUD
from collections import Counter
subs_mostrados = Counter(subtipos.get(i, "?") for i in ids if i in pos_norm)
leyenda = "CÉLULAS DE KENYON (FlyWire)\n" + "\n".join(
    f"{s}: {n}" for s, n in subs_mostrados.most_common(6)
)
plotter.add_text(leyenda, position="upper_left", font_size=10, color="white")
plotter.add_text(
    "Rota con el ratón · Rueda = zoom\nBotón derecho = mover",
    position="lower_left", font_size=9, color=(0.6, 0.6, 0.7),
)

# Cámara inicial
plotter.camera_position = "iso"
plotter.reset_camera()

print("\nAbriendo visor. Si tarda, es porque la primera vez pyvista inicializa OpenGL.")
qapp.exec_()