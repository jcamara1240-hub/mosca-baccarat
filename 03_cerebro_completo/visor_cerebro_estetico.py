"""
CEREBRO DE LA MOSCA · panel estético
-------------------------------------
139.248 neuronas reales de FlyWire (v783) en su posición real.
La actividad NO es aleatoria: se propaga por los 15 millones de
conexiones reales (excitadoras e inhibidoras) del modelo de
Shiu et al. 2024. Las neuronas sensoriales reciben estímulos y
la señal viaja por el cerebro.

Arriba: vista frontal (de cara a la mosca), girando suave.
Abajo:  vista dorsal (desde arriba).

Teclas:  ESPACIO = pausa · S = guardar captura PNG
Uso:     python visor_cerebro_estetico.py              (ventana en vivo)
         python visor_cerebro_estetico.py --captura    (guarda un PNG)
         python visor_cerebro_estetico.py --video      (graba un MP4 de 20 s)
"""

import sys
import time
import numpy as np
import pandas as pd
import scipy.sparse as sp
import pyvista as pv

CARPETA = "datos_flywire"
CAPTURA = "--captura" in sys.argv
VIDEO = "--video" in sys.argv

# ------------------------------------------------------------------
# 1. Neuronas y posiciones
# ------------------------------------------------------------------
print("Cargando neuronas...", flush=True)
ann = pd.read_csv(f"{CARPETA}/Supplemental_file1_neuron_annotations.tsv",
                  sep="\t", low_memory=False,
                  usecols=["root_id", "pos_x", "pos_y", "pos_z", "super_class"])
ann = ann.dropna(subset=["pos_x", "pos_y", "pos_z"]).reset_index(drop=True)
N = len(ann)

# FlyWire guarda X e Y en píxeles de 4 nm y Z en cortes de 40 nm.
# Pasamos todo a micras para que el cerebro tenga su forma real.
pos = ann[["pos_x", "pos_y", "pos_z"]].values.astype(np.float64)
pos *= np.array([4.0, 4.0, 40.0]) / 1000.0
p_lo, p_hi = np.percentile(pos, 0.5, axis=0), np.percentile(pos, 99.5, axis=0)
pos -= (p_lo + p_hi) / 2       # centrado en la caja del cerebro
pos[:, 1] *= -1          # en el microscopio Y crece hacia abajo
pos = pos.astype(np.float32)
print(f"  {N:,} neuronas", flush=True)

# ------------------------------------------------------------------
# 2. Conectoma real (con signo)
# ------------------------------------------------------------------
print("Cargando conexiones (15M, tarda un poco)...", flush=True)
con = pd.read_parquet(f"{CARPETA}/Connectivity_783.parquet",
                      columns=["Presynaptic_ID", "Postsynaptic_ID",
                               "Excitatory x Connectivity"])
idx = pd.Series(np.arange(N), index=ann["root_id"].values)
pre = idx.reindex(con["Presynaptic_ID"].values).values
post = idx.reindex(con["Postsynaptic_ID"].values).values
w = con["Excitatory x Connectivity"].values.astype(np.float32)
ok = ~(np.isnan(pre) | np.isnan(post))
pre, post, w = pre[ok].astype(np.int32), post[ok].astype(np.int32), w[ok]
del con
# W[post, pre]: entrada que recibe 'post' cuando 'pre' dispara
W = sp.csr_matrix((w, (post, pre)), shape=(N, N), dtype=np.float32)
print(f"  {W.nnz:,} conexiones", flush=True)

# Neuronas que reciben estímulo externo (ojos, antenas...)
sensoriales = np.where(ann["super_class"].astype(str).str.contains("sensory"))[0]

# ------------------------------------------------------------------
# 3. Simulación sencilla (integra-y-dispara)
# ------------------------------------------------------------------
v = np.zeros(N, np.float32)          # voltaje
refractario = np.zeros(N, np.int8)   # tras disparar, descansa unos pasos
brillo = np.zeros(N, np.float32)     # rastro visual de disparos
GANANCIA, FUGA, UMBRAL = 0.016, 0.80, 1.0
rng = np.random.default_rng(7)
foco = {"centro": rng.choice(sensoriales), "t": 0}

def paso():
    global v
    disparos = (v > UMBRAL) & (refractario == 0)
    v[disparos] = 0.0
    refractario[disparos] = 4
    refractario[refractario > 0] -= 1
    entrada = W @ disparos.astype(np.float32)
    v = FUGA * v + GANANCIA * entrada
    v[refractario > 0] = 0.0
    # Estímulo: un "golpe" sensorial (ojo, antena...) que cambia de sitio
    foco["t"] += 1
    if foco["t"] % 45 == 0:
        foco["centro"] = rng.choice(sensoriales)
    cerca = sensoriales[np.linalg.norm(pos[sensoriales] - pos[foco["centro"]], axis=1) < 70]
    v[rng.choice(cerca, size=min(len(cerca), 60))] += 1.5
    v[rng.integers(0, N, 80)] += 1.2            # ruido de fondo
    np.clip(v, -2, 3, out=v)
    brillo[:] = brillo * 0.72
    brillo[disparos] = 1.0
    return int(disparos.sum())

# ------------------------------------------------------------------
# 4. Estética
# ------------------------------------------------------------------
FONDO = "#000000"
BASE = np.array([62, 68, 82], np.float32)       # gris azulado muy oscuro
ACTIVO = np.array([235, 242, 255], np.float32)  # blanco frío

def colores(direccion_camara):
    # Sombreado por profundidad: lo cercano más claro, lo lejano se hunde.
    d = pos @ direccion_camara
    prof = (d - d.min()) / (np.ptp(d) + 1e-6)          # 0 lejos, 1 cerca
    base = BASE * (0.35 + 0.65 * prof[:, None])
    b = brillo[:, None] ** 0.8
    c = base * (1 - b) + ACTIVO * b
    return c.clip(0, 255).astype(np.uint8)

if CAPTURA or VIDEO:
    pv.OFF_SCREEN = True
    pl = pv.Plotter(shape=(2, 1), window_size=(720, 1100), border=False,
                    off_screen=True)
else:
    from pyvistaqt import BackgroundPlotter
    pl = BackgroundPlotter(shape=(2, 1), window_size=(720, 1100),
                           border=False, title="Neural Activity · Drosophila",
                           toolbar=False, menu_bar=False)

nubes, halos = [], []
for fila in (0, 1):
    pl.subplot(fila, 0)
    pl.set_background(FONDO)
    nube = pv.PolyData(pos)
    nube["c"] = colores(np.array([0, 0, 1], np.float32))
    pl.add_points(nube, scalars="c", rgb=True, point_size=1.6,
                  render_points_as_spheres=True, lighting=False)
    nubes.append(nube)
    # Halo: puntos grandes y translúcidos solo donde hay actividad
    halo = pv.PolyData(pos[:1].copy())
    halo["c"] = np.zeros((1, 3), np.uint8)
    pl.add_points(halo, scalars="c", rgb=True, point_size=6,
                  render_points_as_spheres=True, lighting=False,
                  opacity=0.13)
    halos.append(halo)

COLOR_HALO = np.array([150, 190, 255], np.float32)

def actualizar_halos():
    activos = np.where(brillo > 0.25)[0]
    if len(activos) == 0:
        activos = np.array([0])
    nueva = pv.PolyData(pos[activos])
    nueva["c"] = (COLOR_HALO * brillo[activos, None]).clip(0, 255).astype(np.uint8)
    for h in halos:
        h.copy_from(nueva)

GRIS = (0.45, 0.47, 0.55)
pl.subplot(0, 0)
pl.add_text("NEURAL ACTIVITY", position=(26, 505), font_size=11,
            color=(0.92, 0.93, 0.96), font="arial")
pl.add_text(f"{N:,} neurons  ·  {W.nnz/1e6:.1f}M connections  ·  FlyWire v783",
            position=(26, 482), font_size=7, color=GRIS, font="arial")
pl.add_text("FRONTAL", position=(26, 18), font_size=7, color=GRIS, font="arial")
hud = pl.add_text("", position=(560, 18), font_size=7, color=GRIS, font="arial")
pl.subplot(1, 0)
pl.add_text("DORSAL", position=(26, 18), font_size=7, color=GRIS, font="arial")
pl.add_text("Dorkenwald et al. 2024  ·  Shiu et al. 2024",
            position=(26, 505), font_size=7, color=(0.30, 0.31, 0.37), font="arial")

# Cámaras
def camara_frontal(angulo):
    a = np.deg2rad(angulo)
    direc = np.array([np.sin(a), 0.0, -np.cos(a)], np.float32)   # mira a la cara
    pl.subplot(0, 0)
    pl.camera_position = [tuple(-direc * 1500), (0, 0, 0), (0, 1, 0)]
    pl.camera.parallel_projection = True
    pl.camera.parallel_scale = 335
    return direc

def camara_dorsal():
    pl.subplot(1, 0)
    pl.camera_position = [(0, 1500, 0), (0, 0, 0), (0, 0, -1)]
    pl.camera.parallel_projection = True
    pl.camera.parallel_scale = 335
    return np.array([0, 1, 0], np.float32)

estado = {"t": 0.0, "pausa": False}
dir_dorsal = camara_dorsal()

def frame():
    if estado["pausa"]:
        return
    n = paso()
    estado["t"] += 1
    angulo = 28 * np.sin(estado["t"] / 90.0)      # vaivén suave
    dir_front = camara_frontal(angulo)
    nubes[0]["c"] = colores(dir_front)
    nubes[1]["c"] = colores(dir_dorsal)
    actualizar_halos()
    hud.SetInput(f"{n:>5} spikes / step")
    pl.render()

camara_frontal(0)

if VIDEO:
    # Graba 20 s a 24 fps (480 fotogramas). Tarda varios minutos.
    for _ in range(40):
        frame()                     # calentar la simulación
    pl.open_movie("cerebro_mosca.mp4", framerate=24, quality=9)
    for i in range(480):
        frame()
        pl.write_frame()
        if i % 48 == 0:
            print(f"  grabando... {i // 24} s", flush=True)
    pl.close()
    print("Guardado cerebro_mosca.mp4")
elif CAPTURA:
    for _ in range(60):
        frame()
    pl.screenshot("captura_cerebro.png")
    print("Guardado captura_cerebro.png")
else:
    def pausa():
        estado["pausa"] = not estado["pausa"]
    def guardar():
        nombre = f"cerebro_{int(time.time())}.png"
        pl.screenshot(nombre)
        print("Captura guardada:", nombre)
    pl.add_key_event("space", pausa)
    pl.add_key_event("s", guardar)
    pl.add_callback(frame, interval=50)
    pl.app.exec_()
