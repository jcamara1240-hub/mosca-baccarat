"""
LA MOSCA JUEGA AL BACCARAT  ·  EN VIVO
======================================
Ventana en tiempo real con tres zonas:
  arriba-izq   su cerebro (139.248 neuronas de FlyWire) disparando
  abajo-izq    la mesa de baccarat en 3D con la mosca (NeuroMechFly, MuJoCo)
  derecha      este mismo código ejecutándose, con su consola

Juega rondas sin fin. Aprende con dopamina y, al cerrar, guarda lo aprendido
en memoria_mosca.npz (la misma memoria que usa mosca_baccarat_3d.py).

Teclas:  ESPACIO pausa  ·  ESC o Q salir (y guardar)  ·  F pantalla completa

Uso (entorno 'moscafly', en la carpeta mosca):
    python mosca_en_vivo.py
    python mosca_en_vivo.py --nuevo      (empieza con un cerebro sin experiencia)

Por defecto usa ANIMACIÓN CINEMÁTICA: las patas se mueven con el generador de
ritmo (CPG) real de NeuroMechFly y el cuerpo avanza a la velocidad que mide la
física, pero sin calcular cada contacto. Es ~150 veces más rápido.
    python mosca_en_vivo.py --fisica     (física completa: realista pero ~1 fps)
"""

import os
import sys
import time
import re
from collections import deque
import numpy as np
import pandas as pd
import scipy.sparse as sp
import cv2
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageDraw, ImageFont

PRUEBA = "--prueba" in sys.argv          # solo para comprobar sin pantalla
FISICA = "--fisica" in sys.argv          # física completa (lenta)
CARPETA = "datos_flywire"
TMP = "tmp_vivo"
os.makedirs(TMP, exist_ok=True)
ANCHO, ALTO = 1920, 1080
IZQ_W, IZQ_H = 960, 540
COD_W = ANCHO - IZQ_W
FPS = 30
VELOCIDAD = 0.25                   # cámara lenta x0.25 (como el vídeo)
DT_FRAME = VELOCIDAD / FPS         # segundos de mosca por fotograma
DT_FISICA = 2e-4                   # x2 más rápido que el vídeo (con 4e-4 la mosca tropieza)
MEMORIA = "memoria_mosca.npz"
if "--nuevo" in sys.argv and os.path.exists(MEMORIA):
    os.remove(MEMORIA)
DIA = int(np.load(MEMORIA)["dia"]) + 1 if os.path.exists(MEMORIA) else 1
SEMILLA = 11 + 7 * DIA

def fuente(tam, negrita=True):
    nombres = (["arialbd.ttf", "DejaVuSans-Bold.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"] if negrita else
               ["arial.ttf", "DejaVuSans.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"])
    for f in nombres:
        try:
            return ImageFont.truetype(f, tam)
        except OSError:
            pass
    return ImageFont.load_default()

def fuente_mono(tam, negrita=False):
    nombres = (["consolab.ttf", "DejaVuSansMono-Bold.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"] if negrita else
               ["consola.ttf", "DejaVuSansMono.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"])
    for f in nombres:
        try:
            return ImageFont.truetype(f, tam)
        except OSError:
            pass
    return ImageFont.load_default()

def fuente_palos(tam):
    for f in ["seguisym.ttf", "arial.ttf", "DejaVuSans.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(f, tam)
        except OSError:
            pass
    return ImageFont.load_default()

# ==================================================================
# 1. CEREBRO: neuronas, conectoma y modelo integra-y-dispara
# ==================================================================
print("Cargando cerebro...", flush=True)
ann = pd.read_csv(f"{CARPETA}/Supplemental_file1_neuron_annotations.tsv",
                  sep="\t", low_memory=False,
                  usecols=["root_id", "pos_x", "pos_y", "pos_z", "super_class",
                           "cell_class", "cell_type", "side"])
ann = ann.dropna(subset=["pos_x", "pos_y", "pos_z"]).reset_index(drop=True)
N = len(ann)
pos = ann[["pos_x", "pos_y", "pos_z"]].values.astype(np.float64)
pos *= np.array([4.0, 4.0, 40.0]) / 1000.0          # a micras
lo, hi = np.percentile(pos, 0.5, axis=0), np.percentile(pos, 99.5, axis=0)
pos -= (lo + hi) / 2
pos[:, 1] *= -1
pos = pos.astype(np.float32)

print("Cargando conexiones...", flush=True)
con = pd.read_parquet(f"{CARPETA}/Connectivity_783.parquet",
                      columns=["Presynaptic_ID", "Postsynaptic_ID",
                               "Excitatory x Connectivity"])
idx = pd.Series(np.arange(N), index=ann["root_id"].values)
pre = idx.reindex(con["Presynaptic_ID"].values).values
post = idx.reindex(con["Postsynaptic_ID"].values).values
w = con["Excitatory x Connectivity"].values.astype(np.float32)
ok = ~(np.isnan(pre) | np.isnan(post))
W = sp.csr_matrix((w[ok], (post[ok].astype(np.int32), pre[ok].astype(np.int32))),
                  shape=(N, N), dtype=np.float32)
del con, pre, post, w
print(f"  {N:,} neuronas · {W.nnz:,} conexiones", flush=True)

clase = ann["cell_class"].astype(str)
tipo = ann["cell_type"].astype(str)
KC = clase.str.contains("Kenyon").values
sensoriales = np.where(ann["super_class"].astype(str).str.contains("sensory"))[0]
PAM = np.where(tipo.str.startswith("PAM"))[0]          # dopamina de recompensa
PPL1 = np.where(tipo.str.startswith("PPL1"))[0]        # dopamina de castigo
mbon = np.where(clase.str.contains("MBON").values)[0]
entrada_kc = np.asarray(W[mbon][:, np.where(KC)[0]].sum(axis=1)).ravel()
tipos_mbon = pd.Series(entrada_kc, index=tipo.values[mbon]).groupby(level=0).sum()
top3 = list(tipos_mbon.sort_values(ascending=False).index[:3])
APUESTAS = ["PLAYER", "BANKER", "TIE"]
GRUPOS = {ap: mbon[tipo.values[mbon] == t] for ap, t in zip(APUESTAS, top3)}

# Guardamos la matriz por COLUMNAS (una columna = las salidas de una neurona).
# Así, en cada paso solo se suman las conexiones de las neuronas que disparan
# (1-2 %) en vez de recorrer los 15 millones: mismo resultado, mucho más rápido.
W = W.tocsc()
_col_de = np.repeat(np.arange(N), np.diff(W.indptr))       # columna de cada entrada
FILAS = np.concatenate([GRUPOS[a] for a in APUESTAS])      # MBON que aprenden
_rango = np.full(N, -1); _rango[FILAS] = np.arange(len(FILAS))
_sel = np.flatnonzero(_rango[W.indices] >= 0)
_sel = _sel[np.lexsort((_col_de[_sel], _rango[W.indices[_sel]]))]   # orden fila, col
POS_GRUPO = {ap: _sel[np.isin(W.indices[_sel], GRUPOS[ap])] for ap in APUESTAS}
del _rango

DINERO_INICIAL = 1000.0
if os.path.exists(MEMORIA):
    mem = np.load(MEMORIA)
    W.data[_sel] = mem["pesos"]
    DINERO_INICIAL = float(mem["dinero"])
    print(f"  memoria cargada: día {DIA}", flush=True)

v = np.zeros(N, np.float32)
refractario = np.zeros(N, np.int8)
brillo = np.zeros(N, np.float32)
extra = np.zeros(N, np.float32)
rng = np.random.default_rng(SEMILLA)
foco = {"c": rng.choice(sensoriales), "t": 0}

def paso_cerebro():
    global v
    disparos = (v > 1.0) & (refractario == 0)
    v[disparos] = 0.0
    refractario[disparos] = 4
    refractario[refractario > 0] -= 1
    act = np.flatnonzero(disparos)
    ini, lon = W.indptr[act], np.diff(W.indptr)[act]
    k = np.repeat(ini - np.cumsum(lon) + lon, lon) + np.arange(lon.sum())
    entrada = np.bincount(W.indices[k], weights=W.data[k], minlength=N)
    v = 0.80 * v + 0.016 * entrada.astype(np.float32)   # <- paso
    v[refractario > 0] = 0.0
    foco["t"] += 1
    if foco["t"] % 45 == 0:
        foco["c"] = rng.choice(sensoriales)
    cerca = sensoriales[np.linalg.norm(pos[sensoriales] - pos[foco["c"]], axis=1) < 70]
    v[rng.choice(cerca, size=min(len(cerca), 60))] += 1.5
    v[rng.integers(0, N, 80)] += 1.2
    v += extra
    np.clip(v, -2, 3, out=v)
    brillo[:] = brillo * 0.72
    brillo[disparos] = 1.0
    return disparos

def aprender(apuesta, kc_activas, factor):
    """Plasticidad dopaminérgica: escala Kenyon->MBON de la apuesta elegida."""
    p = POS_GRUPO[apuesta]
    W.data[p[kc_activas[_col_de[p]]]] *= factor

def guardar_memoria(dinero):
    np.savez(MEMORIA, dia=DIA, dinero=dinero, filas=FILAS, pesos=W.data[_sel])
    print(f"Memoria guardada (día {DIA}, fichas {dinero:.2f})", flush=True)

# ------------------------------------------------------------------
# Dibujo del cerebro sin GPU (numpy): rápido y sin conflictos
# ------------------------------------------------------------------
ESCALA = IZQ_H / 520.0            # píxeles por micra
BASE = np.array([62, 68, 82], np.float32)
ACTIVO = np.array([235, 242, 255], np.float32)
HALO = np.array([0.45, 0.62, 1.0], np.float32)

def dibujar_cerebro(t):
    a = np.deg2rad(25 * np.sin(t / 90.0))
    ca, sa = np.cos(a), np.sin(a)
    x = pos[:, 0] * ca + pos[:, 2] * sa                # gira alrededor del eje vertical
    prof = -pos[:, 0] * sa + pos[:, 2] * ca
    u = (IZQ_W / 2 + x * ESCALA).astype(np.int32)
    vv = (IZQ_H / 2 + 12 - pos[:, 1] * ESCALA).astype(np.int32)
    dentro = (u >= 0) & (u < IZQ_W) & (vv >= 0) & (vv < IZQ_H)
    p = (prof.max() - prof) / (np.ptp(prof) + 1e-6)     # 1 = cerca de la cámara
    b = brillo ** 0.8
    col = BASE * (0.35 + 0.65 * p[:, None])
    col = col * (1 - b[:, None]) + ACTIVO * b[:, None]
    img = np.zeros((IZQ_H, IZQ_W, 3), np.float32)
    todas = np.flatnonzero(dentro)
    img[vv[todas], u[todas]] = col[todas]
    vivas = todas[b[todas] > 0.02]
    vivas = vivas[np.argsort(b[vivas])]                 # lo brillante, encima
    img[vv[vivas], u[vivas]] = col[vivas]
    act = np.zeros((IZQ_H, IZQ_W), np.float32)
    act[vv[vivas], u[vivas]] = b[vivas]
    glow = cv2.GaussianBlur(act, (0, 0), 4.0)
    img += glow[..., None] * HALO * 160
    return np.clip(img, 0, 255).astype(np.uint8)

# ==================================================================
# 2. MESA DE BACCARAT (reglas reales) Y ESCENA 3D
# ==================================================================
rng_cartas = np.random.default_rng(SEMILLA + 1)
PALOS = ["S", "H", "D", "C"]
RANGOS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]

def valor(c):
    r = c[0]
    return 1 if r == "A" else (0 if r in ("10", "J", "Q", "K") else int(r))

def puntos(mano):
    return sum(valor(c) for c in mano) % 10

def jugar_mano(baraja):
    p = [baraja.pop(), baraja.pop()]
    b = [baraja.pop(), baraja.pop()]
    orden = [("P", 0), ("B", 0), ("P", 1), ("B", 1)]
    if puntos(p) < 8 and puntos(b) < 8:                    # sin "natural"
        p3 = None
        if puntos(p) <= 5:
            p3 = baraja.pop(); p.append(p3); orden.append(("P", 2))
        pb = puntos(b)
        if p3 is None:
            pide = pb <= 5
        else:
            t = valor(p3)
            pide = (pb <= 2 or (pb == 3 and t != 8) or (pb == 4 and 2 <= t <= 7)
                    or (pb == 5 and 4 <= t <= 7) or (pb == 6 and t in (6, 7)))
        if pide:
            b.append(baraja.pop()); orden.append(("B", 2))
    pp, pb = puntos(p), puntos(b)
    gana = "PLAYER" if pp > pb else ("BANKER" if pb > pp else "TIE")
    return p, b, orden, gana

def zapato_nuevo():
    z = [(r, s) for r in RANGOS for s in PALOS] * 8
    rng_cartas.shuffle(z)
    return z

baraja = zapato_nuevo()
MESA_X, MESA_Y = (-8.0, 28.0), (-15.0, 15.0)
CASILLAS = {"PLAYER": (10.5, 7.0), "TIE": (13.0, 0.0), "BANKER": (10.5, -7.0)}
TAM_CASILLA = (3.2, 5.6)                 # (largo en X, ancho en Y)
MANO_P, MANO_B = (21.0, 6.0), (21.0, -6.0)
CARTA = (3.4, 2.4)                       # (largo en X, ancho en Y)
ZAPATO = (27.0, 0.0)

PX_MM = 40
TW = int((MESA_Y[1] - MESA_Y[0]) * PX_MM)
TH = int((MESA_X[1] - MESA_X[0]) * PX_MM)

def a_px(x, y):
    """mm de la mesa -> píxel de la textura (arriba = lejos, izquierda = +Y)."""
    return (MESA_Y[1] - y) * PX_MM, (MESA_X[1] - x) * PX_MM

def textura_tapete(ruta):
    im = Image.new("RGB", (TW, TH))
    yy, xx = np.mgrid[0:TH, 0:TW]
    cx, cy = a_px(10, 0)
    r = np.hypot((xx - cx) / TW, (yy - cy) / TH)
    base = np.array([16, 84, 52]) * (1.05 - 0.75 * np.clip(r, 0, 1))[..., None]
    ruido = rng_cartas.normal(0, 2.0, (TH, TW, 1))
    im = Image.fromarray(np.clip(base + ruido, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    oro = (196, 164, 92)
    # Arcos dorados
    for radio, grosor in [(16.5, 5), (17.2, 2)]:
        x0, y0 = a_px(1.0 + radio, 0 + radio)
        x1, y1 = a_px(1.0 - radio, 0 - radio)
        d.arc((x0, y0, x1, y1), start=200, end=340, fill=oro, width=grosor)
    f_big, f_sm = fuente(int(1.2 * PX_MM)), fuente(int(0.42 * PX_MM), False)
    cols = {"PLAYER": (110, 170, 255), "BANKER": (255, 110, 110), "TIE": (255, 210, 90)}
    paga = {"PLAYER": "PAGA 1 A 1", "BANKER": "PAGA 0.95 A 1", "TIE": "PAGA 8 A 1"}
    for nombre, (x, y) in CASILLAS.items():
        x0, y0 = a_px(x + TAM_CASILLA[0] / 2, y + TAM_CASILLA[1] / 2)
        x1, y1 = a_px(x - TAM_CASILLA[0] / 2, y - TAM_CASILLA[1] / 2)
        d.rounded_rectangle((x0, y0, x1, y1), radius=int(0.5 * PX_MM),
                            outline=cols[nombre], width=int(0.12 * PX_MM))
        cx, cy = a_px(x, y)
        w = d.textlength(nombre, font=f_big)
        d.text((cx - w / 2, cy - 0.95 * PX_MM), nombre, font=f_big, fill=cols[nombre])
        w = d.textlength(paga[nombre], font=f_sm)
        d.text((cx - w / 2, cy + 0.45 * PX_MM), paga[nombre], font=f_sm, fill=oro)
    for nombre, (x, y) in [("PLAYER", MANO_P), ("BANKER", MANO_B)]:
        ymax, ymin = (y + 5.9, y - 3.0) if nombre == "PLAYER" else (y + 3.0, y - 5.9)
        x0, y0 = a_px(x + 2.3, ymax)
        x1, y1 = a_px(x - 2.3, ymin)
        d.rounded_rectangle((x0, y0, x1, y1), radius=int(0.4 * PX_MM),
                            outline=oro, width=int(0.06 * PX_MM))
        cx, cy = a_px(x - 2.9, y)
        w = d.textlength(nombre, font=f_sm)
        d.text((cx - w / 2, cy), nombre, font=f_sm, fill=oro)
    f_t = fuente(int(1.6 * PX_MM))
    cx, cy = a_px(26.0, 0)
    w = d.textlength("BACCARAT", font=f_t)
    d.text((cx - w / 2, cy - 0.8 * PX_MM), "BACCARAT", font=f_t, fill=(28, 106, 68))
    im.save(ruta)

def textura_carta(c, ruta):
    W_, H_ = 240, 340
    im = Image.new("RGB", (W_, H_), (16, 84, 52))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((2, 2, W_ - 3, H_ - 3), radius=22, fill=(246, 244, 238),
                        outline=(200, 198, 190), width=2)
    rojo = c[1] in ("H", "D")
    col = (196, 30, 46) if rojo else (22, 22, 26)
    simbolo = {"S": "♠", "H": "♥", "D": "♦", "C": "♣"}[c[1]]
    f_r, f_p, f_c = fuente(64), fuente_palos(54), fuente_palos(150)
    d.text((18, 10), c[0], font=f_r, fill=col)
    d.text((22, 80), simbolo, font=f_p, fill=col)
    w = d.textlength(simbolo, font=f_c)
    d.text(((W_ - w) / 2, 120), simbolo, font=f_c, fill=col)
    im.save(ruta)

def hueco(mano, j):
    """Posición (x, y, giro) del hueco j de la mano P o B."""
    cx, cy = MANO_P if mano == "P" else MANO_B
    if j < 2:
        return cx, cy + (1.4 if j == 0 else -1.4), 0.0
    return cx, cy + (4.0 if mano == "P" else -4.0), np.pi / 2   # 3ª carta, de lado

print("Preparando texturas...", flush=True)
textura_tapete(os.path.abspath(f"{TMP}/tapete.png"))
TODAS = [(r, s) for r in RANGOS for s in PALOS]
for c in TODAS:
    textura_carta(c, os.path.abspath(f"{TMP}/carta_{c[0]}{c[1]}.png"))

from flygym_gymnasium import Fly
from flygym_gymnasium.camera import Camera
from flygym_gymnasium.arena import FlatTerrain
from flygym_gymnasium.examples.locomotion import HybridTurningController
from flygym_gymnasium.preprogrammed import default_leg_sensor_placements

ESCONDIDO = (0.0, 0.0, -3.0)
HUECOS = [("P", 0), ("P", 1), ("P", 2), ("B", 0), ("B", 1), ("B", 2)]

class MesaBaccarat(FlatTerrain):
    def __init__(self):
        super().__init__(size=(400, 400))
        tex = self.root_element.asset.texture[0]
        tex.rgb1 = tex.rgb2 = (0.01, 0.03, 0.02)
        self.root_element.asset.material[0].reflectance = 0.0
        self.root_element.asset.add("texture", type="skybox", builtin="gradient",
                                    rgb1=(0.01, 0.01, 0.015), rgb2=(0, 0, 0),
                                    width=64, height=64)
        def material(nombre, ruta):
            t = self.root_element.asset.add("texture", name=f"t_{nombre}", type="2d",
                                            file=os.path.abspath(ruta))
            return self.root_element.asset.add("material", name=f"m_{nombre}", texture=t,
                                               specular=0.15, shininess=0.3)
        m = material("tapete", f"{TMP}/tapete.png")
        self.root_element.worldbody.add(
            "geom", name="tapete", type="plane",
            size=((MESA_Y[1] - MESA_Y[0]) / 2, (MESA_X[1] - MESA_X[0]) / 2, 0.1),
            pos=((MESA_X[0] + MESA_X[1]) / 2, (MESA_Y[0] + MESA_Y[1]) / 2, 0.002),
            euler=(0, 0, -np.pi / 2), material=m, contype=0, conaffinity=0)
        mats = {c: material(f"c_{c[0]}{c[1]}", f"{TMP}/carta_{c[0]}{c[1]}.png") for c in TODAS}
        for mano, j in HUECOS:      # 6 huecos; su imagen se cambia en cada reparto
            self.root_element.worldbody.add(
                "geom", name=f"hueco_{mano}{j}", type="plane",
                size=(CARTA[1] / 2, CARTA[0] / 2, 0.1), pos=ESCONDIDO,
                euler=(0, 0, -np.pi / 2), material=mats[TODAS[0]],
                contype=0, conaffinity=0)
        self.root_element.worldbody.add("geom", name="ficha", type="cylinder",
                                        size=(0.9, 0.12), pos=ESCONDIDO,
                                        rgba=(0.85, 0.72, 0.30, 1), contype=0, conaffinity=0)
        self.root_element.worldbody.add("geom", name="ficha_borde", type="cylinder",
                                        size=(0.7, 0.13), pos=ESCONDIDO,
                                        rgba=(0.95, 0.95, 0.92, 1), contype=0, conaffinity=0)
        self.root_element.worldbody.add("light", name="foco", pos=(10, 0, 40),
                                        dir=(0, 0, -1), diffuse=(0.55, 0.55, 0.5),
                                        castshadow=True)

def parametros_camara(pos, mira, fovy):
    pos, mira = np.array(pos, float), np.array(mira, float)
    zc = pos - mira; zc /= np.linalg.norm(zc)
    xc = np.cross([0, 0, 1], zc); xc /= np.linalg.norm(xc)
    yc = np.cross(zc, xc)
    return {"mode": "fixed", "pos": tuple(pos), "xyaxes": (*xc, *yc), "fovy": fovy}

print("Montando la mesa y la mosca...", flush=True)
fly = Fly(enable_adhesion=True, draw_adhesion=False,
          contact_sensor_placements=default_leg_sensor_placements, spawn_pos=(0, 0, 0.2))
glob = getattr(fly.model.visual, "global")
glob.offwidth, glob.offheight = ANCHO, ALTO
mesa = MesaBaccarat()
cam = Camera(attachment_point=mesa.root_element.worldbody, camera_name="cam_mesa",
             window_size=(IZQ_W, IZQ_H), play_speed=VELOCIDAD, fps=FPS,
             play_speed_text=False,
             camera_parameters=parametros_camara((-10, 0, 19), (12.5, 0, 0), 40))
sim = HybridTurningController(fly=fly, cameras=[cam], timestep=DT_FISICA, arena=mesa)
obs, _ = sim.reset(seed=0)
fis = sim.physics
for _ in range(int(0.05 / DT_FISICA)):     # que se asiente de pie
    obs, *_ = sim.step(np.zeros(2))

# ---------- Cuerpo cinemático: CPG real + desplazamiento calibrado ----------
RAIZ = next(fis.model.id2name(j, "joint") for j in range(fis.model.njnt)
            if fis.model.jnt_type[j] == 0)
ADR_RAIZ = fis.model.jnt_qposadr[fis.model.name2id(RAIZ, "joint")]
Q_RAIZ = fis.data.qpos[ADR_RAIZ:ADR_RAIZ + 7].copy()
ADR_PATAS = [fis.model.jnt_qposadr[fis.model.name2id(f"{fly.name}/{j}", "joint")]
             for j in fly.actuated_joints]
CPG, PASOS = sim.cpg_network, sim.preprogrammed_steps
N_CPG = max(1, int(round(DT_FRAME / CPG.timestep)))
cuerpo = {"p": Q_RAIZ[:2].copy(), "th": 0.0}
V_MAX, GIRO_MAX = 12.0, 2.9      # mm/s y rad/s, medidos con la física completa

def _qz(t):
    return np.array([np.cos(t / 2), 0, 0, np.sin(t / 2)])

def _qmul(a, b):
    w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
    return np.array([w1*w2 - x1*x2 - y1*y2 - z1*z2, w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2, w1*z2 + x1*y2 - y1*x2 + z1*w2])

def mover_cuerpo(accion):
    """Un fotograma de marcha: el CPG mueve las 42 articulaciones de las patas."""
    CPG.intrinsic_amps = np.repeat(np.abs(accion[:, None]), 3, axis=1).ravel()
    CPG.intrinsic_freqs = sim.intrinsic_freqs.copy()
    for _ in range(N_CPG):
        CPG.step()
    fis.data.qpos[ADR_PATAS] = np.concatenate(
        [PASOS.get_joint_angles(leg, CPG.curr_phases[i], CPG.curr_magnitudes[i])
         for i, leg in enumerate(PASOS.legs)])
    empuje = CPG.curr_magnitudes.mean()               # anda cuando las patas se mueven
    cuerpo["th"] += GIRO_MAX * (accion[1] - accion[0]) * min(1.0, empuje) * DT_FRAME
    cuerpo["p"] += V_MAX * empuje * DT_FRAME * np.array([np.cos(cuerpo["th"]),
                                                         np.sin(cuerpo["th"])])
    fis.data.qpos[ADR_RAIZ:ADR_RAIZ + 7] = np.r_[cuerpo["p"], Q_RAIZ[2],
                                                 _qmul(_qz(cuerpo["th"]), Q_RAIZ[3:7])]
    fis.forward()

MAT = {c: fis.model.name2id(f"m_c_{c[0]}{c[1]}", "material") for c in TODAS}
GEOM_HUECO = {(m, j): fis.model.name2id(f"hueco_{m}{j}", "geom") for m, j in HUECOS}

def mover(nombre, xyz, giro=0.0):
    fis.named.model.geom_pos[nombre] = xyz
    c, s = np.cos((-np.pi / 2 + giro) / 2), np.sin((-np.pi / 2 + giro) / 2)
    fis.named.model.geom_quat[nombre] = (c, 0, 0, s)

def poner_imagen(mano, j, carta):
    fis.model.geom_matid[GEOM_HUECO[(mano, j)]] = MAT[carta]

def posicion():
    return obs["fly"][0][:2] if FISICA else cuerpo["p"]

def orientacion():
    if not FISICA:
        return cuerpo["th"]
    o = obs["fly_orientation"]
    return np.arctan2(o[1], o[0])

def accion_hacia(objetivo):
    p = posicion()
    d = np.array(objetivo) - p
    err = np.arctan2(d[1], d[0]) - orientacion()
    err = (err + np.pi) % (2 * np.pi) - np.pi
    giro = np.clip(1.6 * err, -1, 1)           # >0 = girar a la izquierda
    avance = 1.0 if abs(err) < 1.2 else 0.6
    return (np.clip([avance * (1 - 0.8 * giro), avance * (1 + 0.8 * giro)], 0.1, 1.4),
            np.linalg.norm(d))

# ==================================================================
# 3. INTERFAZ: rótulos de la mesa, código en vivo y consola
# ==================================================================
F_T, F_t, F_G = fuente(22), fuente(15, False), fuente(40)
F_H, F_h = fuente(20, False), fuente(12, False)
COL = {"PLAYER": (110, 170, 255), "BANKER": (255, 110, 110),
       "TIE": (255, 210, 90), None: (220, 222, 230)}

def rotular_cerebro(img, spikes, texto):
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.text((26, 20), "NEURAL ACTIVITY", font=F_H, fill=(235, 237, 244))
    d.text((26, 48), f"{N:,} neurons  ·  FlyWire v783  ·  Shiu et al. 2024",
           font=F_h, fill=(115, 120, 140))
    t = f"{spikes:>5} spikes / step"
    d.text((IZQ_W - 26 - d.textlength(t, font=F_h), 26), t, font=F_h, fill=(115, 120, 140))
    d.text((26, IZQ_H - 36), texto, font=F_t, fill=(205, 209, 224))
    return np.asarray(im)

def rotular_mesa(img, e):
    im = Image.fromarray(np.ascontiguousarray(img))
    d = ImageDraw.Draw(im, "RGBA")
    d.rectangle((0, 0, IZQ_W, 64), fill=(0, 0, 0, 160))
    d.text((22, 10), f"DÍA {DIA}  ·  RONDA {e['ronda'] + 1}", font=F_T, fill=(235, 236, 242))
    d.text((22, 38), TEXTOS_FASE[e["fase"]], font=F_t, fill=(150, 154, 168))
    if e["apuesta"]:
        d.text((300, 10), "APUESTA", font=F_t, fill=(150, 154, 168))
        d.text((300, 30), e["apuesta"], font=F_T, fill=COL[e["apuesta"]])
    pp = e["p_pts"] if e["p_pts"] is not None else "-"
    bp = e["b_pts"] if e["b_pts"] is not None else "-"
    d.text((500, 10), f"PLAYER  {pp}", font=F_T, fill=COL["PLAYER"])
    d.text((500, 36), f"BANKER  {bp}", font=F_T, fill=COL["BANKER"])
    txt = f"{e['dinero']:.2f}"
    d.text((IZQ_W - 22 - d.textlength(txt, font=F_T), 10), txt, font=F_T, fill=(235, 236, 242))
    d.text((IZQ_W - 22 - d.textlength("FICHAS", font=F_t), 38), "FICHAS", font=F_t,
           fill=(150, 154, 168))
    if e["resultado"]:
        col = (120, 235, 150) if e["gana"] else (240, 110, 110)
        w = d.textlength(e["resultado"], font=F_G)
        d.rectangle(((IZQ_W - w) / 2 - 22, IZQ_H - 92, (IZQ_W + w) / 2 + 22, IZQ_H - 34),
                    fill=(0, 0, 0, 170))
        d.text(((IZQ_W - w) / 2, IZQ_H - 88), e["resultado"], font=F_G, fill=col)
    return np.asarray(im)

FUENTE = [l.rstrip("\n") for l in open(__file__, encoding="utf-8")]
_cache_lineas = {}
def linea_de(marca):
    if marca not in _cache_lineas:
        _cache_lineas[marca] = next((n for n, l in enumerate(FUENTE) if marca in l), 0)
    return _cache_lineas[marca]

FM, FMb, FC = fuente_mono(15), fuente_mono(15, True), fuente_mono(14)
C = {"kw": (198, 120, 221), "str": (152, 195, 121), "num": (209, 154, 102),
     "com": (92, 99, 112), "fn": (97, 175, 239), "txt": (171, 178, 191), "nro": (70, 76, 90)}
KW = set("if elif else for while in not and or def return import from as with "
         "True False None global break continue lambda class try except".split())
TOK = re.compile(r'(#.*$)|(f?"[^"]*"|f?\'[^\']*\')|(\b\d+\.?\d*\b)|([A-Za-z_]\w*)|(\s+)|(.)')
COLOR_LOG = {"dec": (97, 175, 239), "win": (120, 235, 150), "lose": (240, 110, 110),
             "dopa": (230, 190, 90), "card": (220, 222, 230), "info": (130, 136, 150)}
LH, N_COD = 21, 30
consola = deque(maxlen=16)
hist_spk = deque(maxlen=240)
scroll = {"top": 0.0}

def pintar_linea(d, x, y, texto):
    for m in TOK.finditer(texto):
        t = m.group(0)
        if m.group(1):   col = C["com"]
        elif m.group(2): col = C["str"]
        elif m.group(3): col = C["num"]
        elif m.group(4):
            col = C["kw"] if t in KW else (C["fn"] if texto[m.end():m.end() + 1] == "(" else C["txt"])
        else:            col = C["txt"]
        d.text((x, y), t, font=FM, fill=col)
        x += d.textlength(t, font=FM)

_cache_cod = {"clave": None, "img": None}

def dibujar_codigo(t_mosca, marca, fps):
    activa = linea_de(marca)
    scroll["top"] += (max(0, activa - 9) - scroll["top"]) * 0.25
    clave = (int(round(scroll["top"])), activa, len(consola) and consola[-1],
             round(fps, 1), int(t_mosca * 20), int(time.time() * 2) % 2)
    if clave == _cache_cod["clave"]:
        return _cache_cod["img"]
    _cache_cod["clave"] = clave
    im = Image.new("RGB", (COD_W, ALTO), (10, 12, 16))
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, COD_W, 44), fill=(16, 19, 25))
    for k, col in enumerate([(237, 106, 94), (245, 191, 79), (98, 197, 84)]):
        d.ellipse((18 + 22 * k, 16, 30 + 22 * k, 28), fill=col)
    d.text((100, 13), "mosca_en_vivo.py", font=FMb, fill=(200, 204, 214))
    modo = "FÍSICA" if FISICA else "CINEMÁTICA"
    etiqueta = f"● EN VIVO · {modo}   {fps:4.1f} fps   t = {t_mosca:7.3f} s"
    d.text((COD_W - 24 - d.textlength(etiqueta, font=FM), 13), etiqueta, font=FM,
           fill=(98, 197, 84))
    top = int(round(scroll["top"]))
    y0 = 62
    for n in range(top, min(top + N_COD, len(FUENTE))):
        y = y0 + (n - top) * LH
        if n == activa:
            d.rectangle((0, y - 2, COD_W, y + LH - 3), fill=(30, 38, 54))
            d.rectangle((0, y - 2, 4, y + LH - 3), fill=(97, 175, 239))
        d.text((14, y), f"{n + 1:>4}", font=FM,
               fill=(140, 150, 170) if n == activa else C["nro"])
        pintar_linea(d, 70, y, FUENTE[n][:92])
    yc = y0 + N_COD * LH + 16
    d.rectangle((0, yc, COD_W, ALTO), fill=(7, 8, 11))
    d.line((0, yc, COD_W, yc), fill=(32, 36, 46), width=1)
    d.text((22, yc + 12), "CONSOLA", font=FMb, fill=(130, 136, 150))
    gx0, gx1, gy0, gy1 = COD_W - 330, COD_W - 24, yc + 10, yc + 34
    if len(hist_spk) > 1:
        mx = max(max(hist_spk), 1)
        d.line([(gx0 + (gx1 - gx0) * k / (hist_spk.maxlen - 1), gy1 - (gy1 - gy0) * s / mx)
                for k, s in enumerate(hist_spk)], fill=(97, 175, 239), width=1)
    d.text((gx0 - 120, yc + 12), "spikes/step", font=FC, fill=(90, 96, 110))
    y = yc + 48
    for tipo_, texto in consola:
        d.text((22, y), texto[:100], font=FC, fill=COLOR_LOG.get(tipo_, C["txt"]))
        y += 19
    if int(time.time() * 2) % 2 == 0:
        d.rectangle((22, y + 2, 31, y + 17), fill=(171, 178, 191))
    _cache_cod["img"] = np.asarray(im)
    return _cache_cod["img"]

# ==================================================================
# 4. BUCLE EN VIVO: cerebro -> decisión -> cuerpo -> pantalla
# ==================================================================
PAGOS = {"PLAYER": 1.0, "BANKER": 0.95, "TIE": 8.0}
PALO_TXT = {"S": "♠", "H": "♥", "D": "♦", "C": "♣"}
TEXTOS_FASE = {"DECIDE": "el cerebro decide...", "CAMINA": "camina a su apuesta",
               "REPARTO": "reparto de cartas", "RESULTADO": "resultado"}

def carta_txt(c):
    return f"{c[0]}{PALO_TXT[c[1]]}"

e = {"ronda": 0, "fase": "DECIDE", "t_fase": 0.0, "apuesta": None,
     "dinero": DINERO_INICIAL, "evidencia": np.zeros(3), "kc_act": np.zeros(N, bool),
     "repartidas": 0, "p_pts": None, "b_pts": None, "resultado": None, "gana": False,
     "dopa": 0, "cerebro": "", "linea_res": "premio", "mano": None}

def log(tipo_, texto):
    consola.append((tipo_, texto))

def nueva_fase(f):
    e["fase"], e["t_fase"] = f, 0.0

def nueva_mano():
    global baraja
    if len(baraja) < 20:
        baraja = zapato_nuevo()
        log("info", ">>> zapato nuevo (8 barajas)")
    p, b, orden, ganador = jugar_mano(baraja)
    for mano, cartas in (("P", p), ("B", b)):
        for j, c in enumerate(cartas):
            poner_imagen(mano, j, c)
    e["mano"] = (p, b, orden, ganador)

for _ in range(40):
    paso_cerebro()
log("info", f">>> día {DIA}: cerebro cargado ({N:,} neuronas, {W.nnz:,} sinapsis)")
log("info", f">>> MBON  PLAYER={top3[0]}  BANKER={top3[1]}  TIE={top3[2]}")
nueva_mano()

if PRUEBA:
    import imageio
    salida = imageio.get_writer("prueba_vivo.mp4", fps=FPS, macro_block_size=8)
    N_PRUEBA = int(next((a for a in sys.argv[1:] if a.isdigit()), "90"))
else:
    cv2.namedWindow("Mosca en vivo", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Mosca en vivo", 1600, 900)

def preparar_dopamina():
    extra[:] = 0.0
    if e["dopa"] > 0:
        extra[PAM if e["gana"] else PPL1] = 1.4                # <- dopa
        e["dopa"] -= 1
        return "dopa"
    return "paso"

def trabajo_cerebro(n):
    """Un paso de cerebro por fotograma, y su dibujo."""
    d = paso_cerebro()
    return d, dibujar_cerebro(n)

HILO = ThreadPoolExecutor(max_workers=1)
accion = np.zeros(2)
pausa, cuadros, t_ini, fps = False, 0, time.time(), 0.0
t_mosca = 0.0
print("¡En marcha! (ESPACIO pausa, ESC sale)", flush=True)
while True:
    if not PRUEBA:
        tecla = cv2.waitKey(1) & 0xFF
        if tecla in (27, ord("q")):
            break
        if tecla == ord(" "):
            pausa = not pausa
        if tecla == ord("f"):
            completo = cv2.getWindowProperty("Mosca en vivo", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("Mosca en vivo", cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_NORMAL if completo == 1 else cv2.WINDOW_FULLSCREEN)
        if cv2.getWindowProperty("Mosca en vivo", cv2.WND_PROP_VISIBLE) < 1:
            break
        if pausa:
            continue

    if FISICA:
        obs, *_ = sim.step(accion)
        frames = sim.render()
        if not frames or frames[0] is None:
            continue
        frame_mesa = frames[0]
        linea = preparar_dopamina()
        disp, img_cerebro = trabajo_cerebro(cuadros + 1)
    else:
        # El cerebro se calcula en otro hilo MIENTRAS la tarjeta gráfica dibuja la mesa
        linea = preparar_dopamina()
        tarea = HILO.submit(trabajo_cerebro, cuadros + 1)
        mover_cuerpo(accion)
        frame_mesa = fis.render(height=IZQ_H, width=IZQ_W, camera_id=cam.camera_id)
        disp, img_cerebro = tarea.result()
    t_mosca += DT_FRAME
    e["t_fase"] += DT_FRAME
    p, b, orden, ganador = e["mano"]
    f = e["fase"]
    t_txt = f"[{t_mosca:7.3f}s]"

    if f == "DECIDE":
        accion = np.zeros(2)
        if e["t_fase"] <= DT_FRAME * 1.01:
            log("info", f"{t_txt} ronda {e['ronda'] + 1}: estimulando neuronas sensoriales...")
        for i, ap in enumerate(APUESTAS):
            g = GRUPOS[ap]
            e["evidencia"][i] += disp[g].sum() + 0.05 * v[g].clip(0).sum()  # <- vota
        e["kc_act"] |= disp & KC
        ev = e["evidencia"]
        linea = "vota"
        e["cerebro"] = "MBON   P {:.1f}   B {:.1f}   T {:.1f}".format(*ev)
        if cuadros % 6 == 0:
            log("info", f"{t_txt} votos MBON  P={ev[0]:5.1f}  B={ev[1]:5.1f}  T={ev[2]:5.1f}")
        if e["t_fase"] > 0.12:
            if ev.max() <= 0:
                ev = ev + rng.random(3) * 1e-3
            e["apuesta"] = APUESTAS[int(np.argmax(ev))]           # <- decide
            linea = "decide"
            e["cerebro"] = f"DECISION  ->  {e['apuesta']}"
            log("dec", f"{t_txt} apuesta = \"{e['apuesta']}\"   "
                       f"({top3[APUESTAS.index(e['apuesta'])]} gana la votación)")
            nueva_fase("CAMINA")
    elif f == "CAMINA":
        accion, dist = accion_hacia(CASILLAS[e["apuesta"]])       # <- camina
        linea = "camina"
        if cuadros % 10 == 0:
            log("info", f"{t_txt} caminando  dist={dist:5.2f} mm  "
                        f"patas izq={accion[0]:.2f} der={accion[1]:.2f}")
        if dist < 1.0 or e["t_fase"] > 1.6:
            accion = np.zeros(2)
            x, y = posicion()
            mover("ficha", (x - 1.6, y, 0.12)); mover("ficha_borde", (x - 1.6, y, 0.125))
            log("dec", f"{t_txt} ficha colocada en {e['apuesta']}")
            nueva_fase("REPARTO")
            e["repartidas"] = 0
    elif f == "REPARTO":
        accion = np.zeros(2)
        DUR = 0.07
        n = min(len(orden), int(e["t_fase"] / DUR) + 1)
        for idx_c in range(n):
            mano, j = orden[idx_c]
            hx, hy, giro = hueco(mano, j)
            a = min(1.0, (e["t_fase"] - idx_c * DUR) / DUR)
            a = 1 - (1 - a) ** 3
            mover(f"hueco_{mano}{j}", (ZAPATO[0] + (hx - ZAPATO[0]) * a,
                                       ZAPATO[1] + (hy - ZAPATO[1]) * a,
                                       0.006 + 0.002 * idx_c), giro * a)  # <- reparte
        linea = "reparte"
        if n > e["repartidas"]:
            mano, j = orden[n - 1]
            c = (p if mano == "P" else b)[j]
            log("card", f"{t_txt} carta {n}: {'PLAYER' if mano == 'P' else 'BANKER':<6} "
                        f"recibe {carta_txt(c)}")
            e["repartidas"] = n
        e["p_pts"] = puntos([p[j] for (m, j) in orden[:n] if m == "P"])
        e["b_pts"] = puntos([b[j] for (m, j) in orden[:n] if m == "B"])
        e["cerebro"] = f"REPARTO   carta {n} / {len(orden)}"
        if e["t_fase"] > DUR * (len(orden) + 1.5):
            e["gana"] = gana = e["apuesta"] == ganador
            mb = top3[APUESTAS.index(e["apuesta"])]
            log("info", f"{t_txt} PLAYER {puntos(p)} vs BANKER {puntos(b)}  ->  gana {ganador}")
            if gana:
                e["dinero"] += PAGOS[ganador]
                e["resultado"] = f"GANA {ganador}   +{PAGOS[ganador]:.2f}"
                aprender(e["apuesta"], e["kc_act"], 1.15)          # <- premio
                linea = e["linea_res"] = "premio"
                e["cerebro"] = "DOPAMINA (PAM)  ->  refuerza Kenyon-MBON"
                log("win", f"{t_txt} ¡acierto! +{PAGOS[ganador]:.2f}  fichas={e['dinero']:.2f}")
                log("dopa", f"{t_txt} dopamina PAM x{len(PAM)}  ->  W[KC->{mb}] *= 1.15")
            else:
                e["dinero"] -= 1.0
                e["resultado"] = f"GANA {ganador}   -1.00"
                aprender(e["apuesta"], e["kc_act"], 0.93)          # <- castigo
                linea = e["linea_res"] = "castigo"
                e["cerebro"] = "DOPAMINA (PPL1)  ->  debilita Kenyon-MBON"
                log("lose", f"{t_txt} fallo  -1.00  fichas={e['dinero']:.2f}")
                log("dopa", f"{t_txt} dopamina PPL1 x{len(PPL1)}  ->  W[KC->{mb}] *= 0.93")
            e["dopa"] = 10
            nueva_fase("RESULTADO")
    elif f == "RESULTADO":
        accion = np.zeros(2)
        linea = e["linea_res"]
        if e["t_fase"] > 0.30:
            for m_, j in HUECOS:
                mover(f"hueco_{m_}{j}", ESCONDIDO)
            mover("ficha", ESCONDIDO); mover("ficha_borde", ESCONDIDO)
            e.update(ronda=e["ronda"] + 1, apuesta=None, p_pts=None, b_pts=None,
                     resultado=None, evidencia=np.zeros(3), kc_act=np.zeros(N, bool),
                     cerebro="")
            nueva_mano()
            nueva_fase("DECIDE")

    # ---------- componer la pantalla ----------
    cuadros += 1
    hist_spk.append(int(disp.sum()))
    if cuadros % 10 == 0:
        ahora = time.time()
        fps = 10 / (ahora - t_ini)
        t_ini = ahora
    lienzo = np.empty((ALTO, ANCHO, 3), np.uint8)
    lienzo[:IZQ_H, :IZQ_W] = rotular_cerebro(img_cerebro, int(disp.sum()),
                                             e["cerebro"])
    lienzo[IZQ_H:, :IZQ_W] = rotular_mesa(frame_mesa[:IZQ_H, :IZQ_W, :3], e)
    lienzo[:, IZQ_W:] = dibujar_codigo(t_mosca, f"# <- {linea}", fps)
    lienzo[IZQ_H:IZQ_H + 1, :IZQ_W] = 30
    lienzo[:, IZQ_W:IZQ_W + 1] = 30
    if PRUEBA:
        salida.append_data(lienzo)
        if cuadros >= N_PRUEBA:
            break
    else:
        cv2.imshow("Mosca en vivo", cv2.cvtColor(lienzo, cv2.COLOR_RGB2BGR))

if PRUEBA:
    salida.close()
    print(f"prueba: {cuadros} cuadros, último fps medido {fps:.1f}")
else:
    cv2.destroyAllWindows()
guardar_memoria(e["dinero"])
sim.close()
