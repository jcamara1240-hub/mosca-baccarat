"""
VISOR FINAL: la mosca con cerebro REAL de FlyWire juega al baccarat.
Ventana 3D con las neuronas reales iluminándose al disparar + panel de mesa.
"""

import pickle
import numpy as np
import random
import time
import pyvista as pv
from pyvistaqt import BackgroundPlotter
from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication
import sys
from pathlib import Path

# El juego está en la carpeta 01
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "01_cerebro_simulado"))
from baccarat import jugar_mano
from mosca_real import MoscaReal


# ---------- Montar la mosca ----------
print("Montando mosca real...", flush=True)
mosca = MoscaReal(semilla=random.randint(0, 10000))


# ---------- Posiciones 3D de las neuronas seleccionadas ----------
coords = mosca.d["coords"]
clases = mosca.d["clases"]

pos_pn   = np.array([coords[r] for r in mosca.pn_sel])
pos_kc   = np.array([coords[r] for r in mosca.kc_sel])
pos_mbon = np.array([coords[r] for r in mosca.mbon_sel])
pos_todo = np.vstack([pos_pn, pos_kc, pos_mbon])
centro = pos_todo.mean(axis=0)
escala = pos_todo.std()

def norm(arr):
    return (arr - centro) / escala * 10

pos_pn_n   = norm(pos_pn)
pos_kc_n   = norm(pos_kc)
pos_mbon_n = norm(pos_mbon)


# ---------- Ventana ----------
qapp = QApplication.instance() or QApplication(sys.argv)
plotter = BackgroundPlotter(
    title="Mosca real de FlyWire jugando al baccarat",
    window_size=(1400, 900),
)
plotter.set_background("#070710")

# PN (verde)
pn_pc = pv.PolyData(pos_pn_n)
plotter.add_points(pn_pc, color=(0.4, 0.95, 0.55), point_size=14,
                   render_points_as_spheres=True, name="pn")

# KC (azul)
kc_pc = pv.PolyData(pos_kc_n)
kc_pc["activ"] = np.zeros(len(pos_kc_n))
kc_actor = plotter.add_points(
    kc_pc, scalars="activ", cmap="viridis", clim=[0, 1],
    point_size=5, render_points_as_spheres=True, name="kc",
    show_scalar_bar=False,
)

# MBON (grandes, con color según apuesta)
mbon_colores = [(1.0, 0.4, 0.4), (0.4, 0.95, 0.55), (1.0, 0.9, 0.3)]  # player, banker, tie
mbon_actores = []
for i, (p, c) in enumerate(zip(pos_mbon_n, mbon_colores)):
    esf = pv.Sphere(radius=0.6, center=p)
    actor = plotter.add_mesh(esf, color=c, smooth_shading=True, specular=0.5)
    mbon_actores.append((actor, c))

# Etiquetas MBON
for i, (p, nombre) in enumerate(zip(pos_mbon_n, ["PLAYER", "BANKER", "TIE"])):
    plotter.add_point_labels([p + np.array([0, 0, 1.5])], [nombre],
                             font_size=14, text_color="white",
                             shape_color="#111122", shape_opacity=0.6)

# Sinapsis KC->MBON (líneas)
pre = np.array(mosca.S_kc_mbon.i)
post = np.array(mosca.S_kc_mbon.j)
puntos_lin, idx_lin = [], []
k = 0
for a, b in zip(pre, post):
    puntos_lin.append(pos_kc_n[a])
    puntos_lin.append(pos_mbon_n[b])
    idx_lin.extend([2, k, k+1])
    k += 2
if puntos_lin:
    poly = pv.PolyData(np.array(puntos_lin))
    poly.lines = np.array(idx_lin)
    plotter.add_mesh(poly, color=(0.6, 0.85, 1.0), opacity=0.9, line_width=3, name="kc_mbon")
    # Sinapsis PN->KC (líneas verdes)
pre2 = np.array(mosca.S_pn_kc.i)
post2 = np.array(mosca.S_pn_kc.j)
puntos2, idx2 = [], []
k2 = 0
for a, b in zip(pre2, post2):
    puntos2.append(pos_pn_n[a])
    puntos2.append(pos_kc_n[b])
    idx2.extend([2, k2, k2+1])
    k2 += 2
if puntos2:
    poly2 = pv.PolyData(np.array(puntos2))
    poly2.lines = np.array(idx2)
    plotter.add_mesh(poly2, color=(0.5, 1.0, 0.6), opacity=0.7, line_width=2, name="pn_kc")
# HUD
hud = plotter.add_text("Arrancando...", position="upper_left",
                       font_size=11, color="white")

plotter.camera_position = "iso"
plotter.reset_camera()


# ---------- Lógica del juego ----------
historico = []
dinero = 1000.0
dinero_inicial = dinero
aciertos = 0
ronda = 0

def sensores_desde_historico():
    ultimos = historico[-10:] if historico else []
    n = max(len(ultimos), 1)
    s = [
        ultimos.count("player") / n,
        ultimos.count("banker") / n,
        ultimos.count("tie")    / n,
    ]
    codif = {"player": 0.2, "banker": 0.7, "tie": 1.0}
    ultimas8 = historico[-8:] if historico else []
    for h in ultimas8:
        s.append(codif[h])
    while len(s) < 11:
        s.append(0.0)
    return np.array(s)


def jugar_mano_vis():
    global dinero, aciertos, ronda
    sensores = sensores_desde_historico()
    apuesta = mosca.decidir(sensores)
    resultado = jugar_mano()
    historico.append(resultado)
    ronda += 1

    pagos = {"player": 1.0, "banker": 0.95, "tie": 8.0}
    if apuesta == resultado:
        dinero += pagos[apuesta]
        aciertos += 1
        mosca.dopamina(apuesta, 0.03)
        msg = f"GANÓ +{pagos[apuesta]:.2f}"
    else:
        dinero -= 1.0
        mosca.dopamina(apuesta, -0.008)
        msg = "PERDIÓ -1.00"

    # Pintar la actividad de las KC en el escalar
    act = mosca.ultima_act_kc
    if act.max() > 0:
        kc_pc["activ"] = act / act.max()
    else:
        kc_pc["activ"] = np.zeros_like(act)

    # Resaltar MBON ganadora
    orden = ["player", "banker", "tie"]
    idx_apuesta = orden.index(apuesta)
    for i, (actor, color_base) in enumerate(mbon_actores):
        prop = actor.GetProperty()
        if i == idx_apuesta:
            prop.SetColor(1.0, 1.0, 1.0)
            prop.SetAmbient(0.7)
        else:
            prop.SetColor(*color_base)
            prop.SetAmbient(0.15)

    tasa = aciertos / ronda * 100
    hud.SetText(2,
        f"CEREBRO REAL DE FLYWIRE\n"
        f"Neuronas: {len(mosca.pn_sel)} PN + {len(mosca.kc_sel)} KC + {len(mosca.mbon_sel)} MBON\n"
        f"Sinapsis reales: {len(mosca.S_pn_kc) + len(mosca.S_kc_mbon)}\n"
        f"─────────────\n"
        f"Ronda {ronda}  |  Dinero: {dinero:.1f}  |  Inicio: {dinero_inicial:.0f}\n"
        f"Aciertos: {aciertos} ({tasa:.1f}%)\n"
        f"Apostó: {apuesta.upper()}  |  Salió: {resultado.upper()}\n"
        f"{msg}"
    )
    plotter.render()


plotter.add_callback(jugar_mano_vis, interval=1500)

print("Visor abierto. Rota con el ratón, cierra la ventana para terminar.", flush=True)
plotter.app.exec_()