"""
Visor 3D del circuito completo de aprendizaje de la mosca.
PN (verde) -> Kenyon (azul/naranja según subtipo) -> MBON (amarillo).
DAN (rojo) = dopamina.
"""

import pickle
import numpy as np
import pyvista as pv
from pyvistaqt import BackgroundPlotter
from PyQt5.QtWidgets import QApplication
import sys

print("Cargando cerebro_full.pkl...")
with open("cerebro_full.pkl", "rb") as f:
    d = pickle.load(f)

coords   = d["coords"]
clases   = d["clases"]
subtipos = d["subtipos"]
conex    = d["conexiones"]

# Normalizar coordenadas
posiciones = np.array([coords[i] for i in coords])
centro = posiciones.mean(axis=0)
escala = posiciones.std()
pos = {}
for rid, p in coords.items():
    pos[rid] = (np.array(p) - centro) / escala * 10

# Colores por clase
COLORES = {
    "ALPN":        (0.40, 0.95, 0.55),  # verde: entrada olfativa
    "Kenyon_Cell": (0.40, 0.75, 1.00),  # azul: memoria
    "MBON":        (1.00, 0.88, 0.30),  # amarillo: salida
    "DAN":         (1.00, 0.35, 0.35),  # rojo: dopamina
}
TAMANO = {
    "ALPN":        5,
    "Kenyon_Cell": 4,
    "MBON":        10,
    "DAN":         8,
}

# Ventana
qapp = QApplication.instance() or QApplication(sys.argv)
plotter = BackgroundPlotter(
    title="Circuito de aprendizaje de la mosca (FlyWire)",
    window_size=(1300, 950),
)
plotter.set_background("#070710")

# Dibujar cada tipo como su propia nube
for cls, color in COLORES.items():
    pts = [pos[r] for r, c in clases.items() if c == cls and r in pos]
    if not pts:
        continue
    pts = np.array(pts)
    pc = pv.PolyData(pts)
    plotter.add_points(
        pc, color=color,
        point_size=TAMANO[cls],
        render_points_as_spheres=True,
    )
    print(f"  {cls}: {len(pts)} puntos")

# Dibujar conexiones clave
def dibujar_lineas(filtro_func, color, opacidad, max_lineas=2000, umbral=3):
    seleccion = []
    for _, row in conex.iterrows():
        pre, post = int(row["pre"]), int(row["post"])
        if pre not in pos or post not in pos:
            continue
        if row["syn_count"] < umbral:
            continue
        cpre = clases.get(pre, "?")
        cpost = clases.get(post, "?")
        if filtro_func(cpre, cpost):
            seleccion.append((pre, post, int(row["syn_count"])))
    # Nos quedamos con las más fuertes
    seleccion.sort(key=lambda x: -x[2])
    seleccion = seleccion[:max_lineas]
    if not seleccion:
        return
    puntos_lin, idx_lin = [], []
    k = 0
    for pre, post, _ in seleccion:
        puntos_lin.append(pos[pre])
        puntos_lin.append(pos[post])
        idx_lin.extend([2, k, k + 1])
        k += 2
    poly = pv.PolyData(np.array(puntos_lin))
    poly.lines = np.array(idx_lin)
    plotter.add_mesh(poly, color=color, opacity=opacidad, line_width=1)
    print(f"    líneas dibujadas: {len(seleccion)}")

print("\nDibujando conexiones...")
print("  PN -> Kenyon (verde)")
dibujar_lineas(lambda a, b: a == "ALPN" and b == "Kenyon_Cell",
               color=(0.3, 0.9, 0.4), opacidad=0.08, max_lineas=1500)
print("  Kenyon -> MBON (azul claro)")
dibujar_lineas(lambda a, b: a == "Kenyon_Cell" and b == "MBON",
               color=(0.5, 0.7, 1.0), opacidad=0.12, max_lineas=2000)
print("  DAN -> Kenyon y DAN -> MBON (rojo)")
dibujar_lineas(lambda a, b: a == "DAN" and b in ("Kenyon_Cell", "MBON"),
               color=(1.0, 0.4, 0.4), opacidad=0.20, max_lineas=1000)

# Leyenda
texto = (
    "CIRCUITO DE APRENDIZAJE (FlyWire)\n"
    "─────────────────\n"
    f"ALPN (verde): {len(d['pn_ids'])} entradas olfativas\n"
    f"Kenyon (azul): {len(d['kenyon_ids'])} memoria\n"
    f"MBON (amarillo): {len(d['mbon_ids'])} decisión\n"
    f"DAN (rojo): {len(d['dan_ids'])} dopamina"
)
plotter.add_text(texto, position="upper_left", font_size=10, color="white")
plotter.add_text(
    "Ratón = rotar · Rueda = zoom · Botón der = mover",
    position="lower_left", font_size=9, color=(0.6, 0.6, 0.7),
)

plotter.camera_position = "iso"
plotter.reset_camera()

print("\nAbriendo visor...")
qapp.exec_()