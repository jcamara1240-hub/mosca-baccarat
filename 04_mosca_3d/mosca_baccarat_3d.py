"""
LA MOSCA JUEGA AL BACCARAT  ·  cuerpo + cerebro real
====================================================
Izquierda: mesa de baccarat en 3D. La mosca (NeuroMechFly v2, física MuJoCo)
           decide su apuesta con su cerebro, camina hasta la casilla,
           se reparten las cartas y se resuelve la mano.
Derecha:   su cerebro (139.248 neuronas de FlyWire) disparando en vivo.

Cada ronda:
  1. DECIDE   El cerebro (modelo de Shiu et al. 2024 sobre el conectoma real)
              recibe estímulos. Gana la apuesta cuyo grupo de neuronas de
              salida del cuerpo fungiforme (MBON) se activa más.
  2. CAMINA   La mosca anda hasta la casilla elegida (el controlador de patas
              de NeuroMechFly hace los pasos).
  3. REPARTO  Cartas con las reglas reales del baccarat (tercera carta incluida).
  4. RESULTADO Si gana: se disparan sus neuronas de dopamina de recompensa (PAM)
              y se refuerzan las conexiones Kenyon -> MBON que llevaron a esa
              apuesta. Si pierde: dopamina de castigo (PPL1) y se debilitan.

Interfaz: arriba-izq el cerebro, abajo-izq la mesa, derecha el código
ejecutándose en tiempo real con su consola.

Memoria entre vídeos: al terminar guarda lo aprendido en memoria_mosca.npz.
El siguiente vídeo empieza en el DÍA siguiente con ese cerebro.

Uso (entorno 'moscafly', en la carpeta mosca):
    python mosca_baccarat_3d.py           -> 24 s, día siguiente  (mosca_baccarat_diaN.mp4)
    python mosca_baccarat_3d.py 8         -> 8 s, para probar
    python mosca_baccarat_3d.py --nuevo   -> borra la memoria y empieza en el día 1
"""

import os
import sys
import json
import subprocess
import numpy as np
import pandas as pd
import scipy.sparse as sp
import imageio
from PIL import Image, ImageDraw, ImageFont

PASADA_PANEL = "--panel" in sys.argv
_nums = [a for a in sys.argv[1:] if not a.startswith("--")]
SEGUNDOS = float(_nums[0]) if _nums else 24.0

CARPETA = "datos_flywire"
TMP = "tmp_baccarat"
os.makedirs(TMP, exist_ok=True)
FPS = 30
VELOCIDAD = 0.25                    # cámara lenta x0.25
DT_FRAME = VELOCIDAD / FPS          # segundos de simulación por fotograma
ANCHO, ALTO = 1920, 1080
IZQ_W, IZQ_H = 960, 540            # cerebro (arriba) y mesa (abajo)
COD_W = ANCHO - IZQ_W              # código a la derecha
FOTOGRAMAS = int(SEGUNDOS * FPS)
MEMORIA = "memoria_mosca.npz"
if "--nuevo" in sys.argv and os.path.exists(MEMORIA):
    os.remove(MEMORIA)
DIA = int(np.load(MEMORIA)["dia"]) + 1 if os.path.exists(MEMORIA) else 1
SEMILLA = 11 + 7 * DIA             # cada día, otras cartas y otros estímulos

# ------------------------------------------------------------------
# Utilidades de dibujo
# ------------------------------------------------------------------
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
# 1. CEREBRO (se usa en las dos pasadas)
# ==================================================================
print("Cargando cerebro...", flush=True)
ann = pd.read_csv(f"{CARPETA}/Supplemental_file1_neuron_annotations.tsv",
                  sep="\t", low_memory=False,
                  usecols=["root_id", "pos_x", "pos_y", "pos_z", "super_class",
                           "cell_class", "cell_type", "side"])
ann = ann.dropna(subset=["pos_x", "pos_y", "pos_z"]).reset_index(drop=True)
N = len(ann)
pos = ann[["pos_x", "pos_y", "pos_z"]].values.astype(np.float64)
pos *= np.array([4.0, 4.0, 40.0]) / 1000.0
lo, hi = np.percentile(pos, 0.5, axis=0), np.percentile(pos, 99.5, axis=0)
pos -= (lo + hi) / 2
pos[:, 1] *= -1
pos = pos.astype(np.float32)

# ==================================================================
# 2. PASADA 2: cerebro + mesa + código en vivo -> vídeo final
# ==================================================================
if PASADA_PANEL:
    import re
    from collections import deque
    import cv2
    brillos = np.memmap(f"{TMP}/brillo.u8", dtype=np.uint8, mode="r").reshape(-1, N)
    meta = json.load(open(f"{TMP}/hud.json"))
    hud, dia = meta["frames"], meta["dia"]

    # ---------------- Cerebro (arriba a la izquierda) ----------------
    # Se dibuja con numpy (sin tarjeta gráfica): funciona igual en cualquier
    # ordenador y es más rápido que un render 3D para 139.000 puntos.
    ESCALA = IZQ_H / 520.0                       # píxeles por micra
    BASE = np.array([62, 68, 82], np.float32)
    ACTIVO = np.array([235, 242, 255], np.float32)
    HALO = np.array([0.45, 0.62, 1.0], np.float32)
    F_H, F_h, F_e = fuente(20, False), fuente(12, False), fuente(15, False)

    def render_cerebro(i, brillo, h):
        a = np.deg2rad(25 * np.sin(i / 90.0))
        ca, sa = np.cos(a), np.sin(a)
        x = pos[:, 0] * ca + pos[:, 2] * sa          # gira alrededor del eje vertical
        prof = -pos[:, 0] * sa + pos[:, 2] * ca
        u = (IZQ_W / 2 + x * ESCALA).astype(np.int32)
        vv = (IZQ_H / 2 + 12 - pos[:, 1] * ESCALA).astype(np.int32)
        dentro = (u >= 0) & (u < IZQ_W) & (vv >= 0) & (vv < IZQ_H)
        p = (prof.max() - prof) / (np.ptp(prof) + 1e-6)    # 1 = cerca de la cámara
        b = brillo ** 0.8
        col = BASE * (0.35 + 0.65 * p[:, None])
        col = col * (1 - b[:, None]) + ACTIVO * b[:, None]
        img = np.zeros((IZQ_H, IZQ_W, 3), np.float32)
        todas = np.flatnonzero(dentro)
        img[vv[todas], u[todas]] = col[todas]
        vivas = todas[b[todas] > 0.02]
        vivas = vivas[np.argsort(b[vivas])]               # lo brillante, encima
        img[vv[vivas], u[vivas]] = col[vivas]
        act = np.zeros((IZQ_H, IZQ_W), np.float32)
        act[vv[vivas], u[vivas]] = b[vivas]
        img += cv2.GaussianBlur(act, (0, 0), 4.0)[..., None] * HALO * 160
        im = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
        d = ImageDraw.Draw(im)
        d.text((26, 20), "NEURAL ACTIVITY", font=F_H, fill=(235, 237, 244))
        d.text((26, 48), f"{N:,} neurons  ·  FlyWire v783  ·  Shiu et al. 2024",
               font=F_h, fill=(115, 120, 140))
        t = f"{h['spikes']:>5} spikes / step"
        d.text((IZQ_W - 26 - d.textlength(t, font=F_h), 26), t, font=F_h, fill=(115, 120, 140))
        d.text((26, IZQ_H - 36), h["cerebro"], font=F_e, fill=(205, 209, 224))
        return np.asarray(im)

    # ---------------- Mesa (abajo a la izquierda) ----------------
    F_T, F_t, F_G = fuente(22), fuente(15, False), fuente(40)
    COL = {"PLAYER": (110, 170, 255), "BANKER": (255, 110, 110),
           "TIE": (255, 210, 90), None: (220, 222, 230)}

    def rotular(img, h):
        im = Image.fromarray(np.ascontiguousarray(img))
        d = ImageDraw.Draw(im, "RGBA")
        d.rectangle((0, 0, IZQ_W, 64), fill=(0, 0, 0, 160))
        d.text((22, 10), f"DÍA {dia}  ·  RONDA {h['ronda']}", font=F_T, fill=(235, 236, 242))
        d.text((22, 38), h["fase"], font=F_t, fill=(150, 154, 168))
        if h["apuesta"]:
            d.text((300, 10), "APUESTA", font=F_t, fill=(150, 154, 168))
            d.text((300, 30), h["apuesta"], font=F_T, fill=COL[h["apuesta"]])
        pp = h["p_pts"] if h["p_pts"] is not None else "-"
        bp = h["b_pts"] if h["b_pts"] is not None else "-"
        d.text((500, 10), f"PLAYER  {pp}", font=F_T, fill=COL["PLAYER"])
        d.text((500, 36), f"BANKER  {bp}", font=F_T, fill=COL["BANKER"])
        txt = f"{h['dinero']:.2f}"
        d.text((IZQ_W - 22 - d.textlength(txt, font=F_T), 10), txt, font=F_T,
               fill=(235, 236, 242))
        d.text((IZQ_W - 22 - d.textlength("FICHAS", font=F_t), 38), "FICHAS",
               font=F_t, fill=(150, 154, 168))
        if h["resultado"]:
            col = (120, 235, 150) if h["gana"] else (240, 110, 110)
            w = d.textlength(h["resultado"], font=F_G)
            d.rectangle(((IZQ_W - w) / 2 - 22, IZQ_H - 92, (IZQ_W + w) / 2 + 22,
                         IZQ_H - 34), fill=(0, 0, 0, 170))
            d.text(((IZQ_W - w) / 2, IZQ_H - 88), h["resultado"], font=F_G, fill=col)
        return np.asarray(im)

    # ---------------- Código en vivo (derecha) ----------------
    FUENTE = [l.rstrip("\n") for l in open(__file__, encoding="utf-8")]
    def linea_de(marca):
        for n, l in enumerate(FUENTE):
            if marca in l and not l.strip().startswith('"linea"'):
                return n
        return 0

    FM, FMb = fuente_mono(15), fuente_mono(15, True)
    FC = fuente_mono(14)
    C_FONDO = (10, 12, 16)
    C = {"kw": (198, 120, 221), "str": (152, 195, 121), "num": (209, 154, 102),
         "com": (92, 99, 112), "fn": (97, 175, 239), "txt": (171, 178, 191),
         "nro": (70, 76, 90)}
    KW = set("if elif else for while in not and or def return import from as with "
             "True False None global break continue lambda class try except".split())
    TOK = re.compile(r'(#.*$)|(f?"[^"]*"|f?\'[^\']*\')|(\b\d+\.?\d*\b)|([A-Za-z_]\w*)|(\s+)|(.)')

    def pintar_linea(d, x, y, texto):
        for m in TOK.finditer(texto):
            t = m.group(0)
            if m.group(1):   col = C["com"]
            elif m.group(2): col = C["str"]
            elif m.group(3): col = C["num"]
            elif m.group(4):
                resto = texto[m.end():m.end() + 1]
                col = C["kw"] if t in KW else (C["fn"] if resto == "(" else C["txt"])
            else:            col = C["txt"]
            d.text((x, y), t, font=FM, fill=col)
            x += d.textlength(t, font=FM)

    LH = 21                                   # alto de línea
    N_COD = 30                                # líneas de código visibles
    consola = deque(maxlen=16)
    estado_scroll = {"top": 0.0}
    hist_spk = deque(maxlen=240)
    COLOR_LOG = {"dec": (97, 175, 239), "win": (120, 235, 150), "lose": (240, 110, 110),
                 "dopa": (230, 190, 90), "card": (220, 222, 230), "info": (130, 136, 150)}

    def render_codigo(i, h):
        im = Image.new("RGB", (COD_W, ALTO), C_FONDO)
        d = ImageDraw.Draw(im)
        # Barra superior
        d.rectangle((0, 0, COD_W, 44), fill=(16, 19, 25))
        for k, col in enumerate([(237, 106, 94), (245, 191, 79), (98, 197, 84)]):
            d.ellipse((18 + 22 * k, 16, 30 + 22 * k, 28), fill=col)
        d.text((100, 13), "mosca_baccarat_3d.py", font=FMb, fill=(200, 204, 214))
        etiqueta = f"● EN EJECUCIÓN   t = {i * DT_FRAME:6.3f} s"
        d.text((COD_W - 24 - d.textlength(etiqueta, font=FM), 13), etiqueta, font=FM,
               fill=(98, 197, 84) if (i // 15) % 2 == 0 else (60, 120, 52))
        # Código, con scroll suave hacia la línea activa
        activa = linea_de(h["linea"])
        objetivo = max(0, activa - 9)
        estado_scroll["top"] += (objetivo - estado_scroll["top"]) * 0.18
        top = int(round(estado_scroll["top"]))
        y0 = 62
        for n in range(top, min(top + N_COD, len(FUENTE))):
            y = y0 + (n - top) * LH
            if n == activa:
                d.rectangle((0, y - 2, COD_W, y + LH - 3), fill=(30, 38, 54))
                d.rectangle((0, y - 2, 4, y + LH - 3), fill=(97, 175, 239))
            num = f"{n + 1:>4}"
            d.text((14, y), num, font=FM, fill=(140, 150, 170) if n == activa else C["nro"])
            pintar_linea(d, 70, y, FUENTE[n][:92].replace("\t", "    "))
        # Consola
        yc = y0 + N_COD * LH + 16
        d.rectangle((0, yc, COD_W, ALTO), fill=(7, 8, 11))
        d.line((0, yc, COD_W, yc), fill=(32, 36, 46), width=1)
        d.text((22, yc + 12), "CONSOLA", font=FMb, fill=(130, 136, 150))
        # Mini gráfica de disparos
        hist_spk.append(h["spikes"])
        gx0, gx1, gy0, gy1 = COD_W - 330, COD_W - 24, yc + 10, yc + 34
        if len(hist_spk) > 1:
            mx = max(max(hist_spk), 1)
            pts = [(gx0 + (gx1 - gx0) * k / (hist_spk.maxlen - 1),
                    gy1 - (gy1 - gy0) * v_ / mx) for k, v_ in enumerate(hist_spk)]
            d.line(pts, fill=(97, 175, 239), width=1)
        d.text((gx0 - 120, yc + 12), "spikes/step", font=FC, fill=(90, 96, 110))
        for tipo_, texto in h["log"]:
            consola.append((tipo_, texto))
        y = yc + 48
        for tipo_, texto in consola:
            d.text((22, y), texto[:100], font=FC, fill=COLOR_LOG.get(tipo_, C["txt"]))
            y += 19
        # Cursor parpadeante
        if (i // 15) % 2 == 0:
            d.rectangle((22, y + 2, 31, y + 17), fill=(171, 178, 191))
        return np.asarray(im)

    lector = imageio.get_reader(f"{TMP}/mesa.mp4")
    nombre_video = f"mosca_baccarat_dia{dia}.mp4"
    salida = imageio.get_writer(nombre_video, fps=FPS, quality=8, macro_block_size=8)
    print("Pasada 2: dibujando cerebro, mesa y código...", flush=True)
    for i, frame_mesa in enumerate(lector):
        if i >= len(hud):
            break
        h = hud[i]
        b = brillos[i].astype(np.float32) / 255.0
        lienzo = np.zeros((ALTO, ANCHO, 3), np.uint8)
        lienzo[:IZQ_H, :IZQ_W] = render_cerebro(i, b, h)
        lienzo[IZQ_H:, :IZQ_W] = rotular(frame_mesa[:IZQ_H, :IZQ_W, :3], h)
        lienzo[:, IZQ_W:] = render_codigo(i, h)
        lienzo[IZQ_H:IZQ_H + 1, :IZQ_W] = 30           # separadores finos
        lienzo[:, IZQ_W:IZQ_W + 1] = 30
        salida.append_data(lienzo)
        if (i + 1) % FPS == 0:
            print(f"  {(i + 1) // FPS} s", flush=True)
    salida.close()
    print(f"\nGuardado {nombre_video}  <-- este es el bueno")
    sys.exit(0)

# ==================================================================
# 3. PASADA 1: conectoma, juego y cuerpo
# ==================================================================
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
es_kc = KC
sensoriales = np.where(ann["super_class"].astype(str).str.contains("sensory"))[0]
PAM = np.where(tipo.str.startswith("PAM"))[0]          # dopamina de recompensa
PPL1 = np.where(tipo.str.startswith("PPL1"))[0]        # dopamina de castigo

# Las 3 familias de MBON que más entrada reciben de las células de Kenyon
mbon = np.where(clase.str.contains("MBON").values)[0]
entrada_kc = np.asarray(W[mbon][:, np.where(es_kc)[0]].sum(axis=1)).ravel()
tipos_mbon = pd.Series(entrada_kc, index=tipo.values[mbon]).groupby(level=0).sum()
top3 = list(tipos_mbon.sort_values(ascending=False).index[:3])
GRUPOS = {ap: mbon[tipo.values[mbon] == t] for ap, t in zip(["PLAYER", "BANKER", "TIE"], top3)}
print(f"  MBON -> PLAYER={top3[0]}, BANKER={top3[1]}, TIE={top3[2]}", flush=True)
print(f"  dopamina: {len(PAM)} PAM (premio), {len(PPL1)} PPL1 (castigo)", flush=True)

DINERO_INICIAL = 1000.0
if os.path.exists(MEMORIA):
    mem = np.load(MEMORIA)
    k0 = 0
    for r in mem["filas"]:
        a, b_ = W.indptr[r], W.indptr[r + 1]
        W.data[a:b_] = mem["pesos"][k0:k0 + (b_ - a)]
        k0 += b_ - a
    DINERO_INICIAL = float(mem["dinero"])
    print(f"  memoria cargada: día {DIA}, la mosca recuerda lo aprendido", flush=True)
else:
    print("  día 1: cerebro sin experiencia", flush=True)

v = np.zeros(N, np.float32)
refractario = np.zeros(N, np.int8)
brillo = np.zeros(N, np.float32)
rng = np.random.default_rng(SEMILLA)
foco = {"c": rng.choice(sensoriales), "t": 0}
extra = np.zeros(N, np.float32)          # estímulo puntual (dopamina)

def paso_cerebro():
    global v
    disparos = (v > 1.0) & (refractario == 0)
    v[disparos] = 0.0
    refractario[disparos] = 4
    refractario[refractario > 0] -= 1
    v = 0.80 * v + 0.016 * (W @ disparos.astype(np.float32))   # <- paso
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
    for r in GRUPOS[apuesta]:
        a, b = W.indptr[r], W.indptr[r + 1]
        cols = W.indices[a:b]
        m = kc_activas[cols]
        W.data[a:b][m] *= factor

# ------------------------------------------------------------------
# Baccarat con reglas reales
# ------------------------------------------------------------------
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

baraja = [(r, s) for r in RANGOS for s in PALOS] * 8
rng_cartas = np.random.default_rng(SEMILLA + 1)
rng_cartas.shuffle(baraja)
N_RONDAS = 8
manos = [jugar_mano(baraja) for _ in range(N_RONDAS)]

# ------------------------------------------------------------------
# Texturas: tapete y cartas
# ------------------------------------------------------------------
# Geometría de la mesa (en mm). La mosca empieza en (0,0) mirando a +X.
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

textura_tapete(os.path.abspath(f"{TMP}/tapete.png"))
cartas_ronda = []
for k, (p, b, orden, gana) in enumerate(manos):
    for j, c in enumerate(p):
        textura_carta(c, os.path.abspath(f"{TMP}/c_{k}_P{j}.png"))
    for j, c in enumerate(b):
        textura_carta(c, os.path.abspath(f"{TMP}/c_{k}_B{j}.png"))

def hueco(mano, j):
    """Posición (x, y, giro) del hueco j de la mano P o B."""
    cx, cy = MANO_P if mano == "P" else MANO_B
    if j < 2:
        return cx, cy + (1.4 if j == 0 else -1.4), 0.0
    return cx, cy + (4.0 if mano == "P" else -4.0), np.pi / 2   # 3ª carta, de lado

# ------------------------------------------------------------------
# Escena MuJoCo
# ------------------------------------------------------------------
from flygym_gymnasium import Fly
from flygym_gymnasium.camera import Camera
from flygym_gymnasium.arena import FlatTerrain
from flygym_gymnasium.examples.locomotion import HybridTurningController
from flygym_gymnasium.preprogrammed import default_leg_sensor_placements

ESCONDIDO = (0.0, 0.0, -3.0)            # debajo del suelo

class MesaBaccarat(FlatTerrain):
    def __init__(self):
        super().__init__(size=(400, 400))
        tex = self.root_element.asset.texture[0]
        tex.rgb1 = tex.rgb2 = (0.01, 0.03, 0.02)
        self.root_element.asset.material[0].reflectance = 0.0
        self.root_element.asset.add("texture", type="skybox", builtin="gradient",
                                    rgb1=(0.01, 0.01, 0.015), rgb2=(0, 0, 0),
                                    width=64, height=64)
        def plano(nombre, ruta, centro, medio, giro=0.0, z=0.004, pos=None):
            t = self.root_element.asset.add("texture", name=f"t_{nombre}", type="2d",
                                            file=os.path.abspath(ruta))
            m = self.root_element.asset.add("material", name=f"m_{nombre}", texture=t,
                                            specular=0.15, shininess=0.3)
            self.root_element.worldbody.add(
                "geom", name=nombre, type="plane", size=(*medio, 0.1),
                pos=pos if pos is not None else (*centro, z),
                euler=(0, 0, -np.pi / 2 + giro), material=m,
                contype=0, conaffinity=0)
        plano("tapete", f"{TMP}/tapete.png",
              ((MESA_X[0] + MESA_X[1]) / 2, (MESA_Y[0] + MESA_Y[1]) / 2),
              ((MESA_Y[1] - MESA_Y[0]) / 2, (MESA_X[1] - MESA_X[0]) / 2), z=0.002)
        for k, (p, b, _, _) in enumerate(manos):
            for mano, cartas in (("P", p), ("B", b)):
                for j in range(len(cartas)):
                    plano(f"carta_{k}_{mano}{j}", f"{TMP}/c_{k}_{mano}{j}.png", None,
                          (CARTA[1] / 2, CARTA[0] / 2), pos=ESCONDIDO)
        # Ficha de apuesta
        self.root_element.worldbody.add("geom", name="ficha", type="cylinder",
                                        size=(0.9, 0.12), pos=ESCONDIDO,
                                        rgba=(0.85, 0.72, 0.30, 1), contype=0,
                                        conaffinity=0)
        self.root_element.worldbody.add("geom", name="ficha_borde", type="cylinder",
                                        size=(0.7, 0.13), pos=ESCONDIDO,
                                        rgba=(0.95, 0.95, 0.92, 1), contype=0,
                                        conaffinity=0)
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
          contact_sensor_placements=default_leg_sensor_placements,
          spawn_pos=(0, 0, 0.2))
glob = getattr(fly.model.visual, "global")
glob.offwidth, glob.offheight = ANCHO, ALTO
mesa = MesaBaccarat()
cam = Camera(attachment_point=mesa.root_element.worldbody, camera_name="cam_mesa",
             window_size=(IZQ_W, IZQ_H), play_speed=VELOCIDAD, fps=FPS,
             play_speed_text=False,
             camera_parameters=parametros_camara((-10, 0, 19), (12.5, 0, 0), 40))
sim = HybridTurningController(fly=fly, cameras=[cam], timestep=1e-4, arena=mesa)
obs, _ = sim.reset(seed=0)
fis = sim.physics

def mover(nombre, xyz, giro=0.0):
    fis.named.model.geom_pos[nombre] = xyz
    c, s = np.cos((-np.pi / 2 + giro) / 2), np.sin((-np.pi / 2 + giro) / 2)
    fis.named.model.geom_quat[nombre] = (c, 0, 0, s)

# ------------------------------------------------------------------
# Bucle del juego
# ------------------------------------------------------------------
writer = imageio.get_writer(f"{TMP}/mesa.mp4", fps=FPS, quality=9, macro_block_size=8)
brillo_disco = np.memmap(f"{TMP}/brillo.u8", dtype=np.uint8, mode="w+",
                         shape=(FOTOGRAMAS, N))
hud = []
estado = {"ronda": 0, "fase": "DECIDE", "t_fase": 0.0, "apuesta": None,
          "dinero": DINERO_INICIAL, "evidencia": np.zeros(3), "kc_act": np.zeros(N, bool),
          "repartidas": 0, "p_pts": None, "b_pts": None, "resultado": None,
          "gana": False, "dopa": 0, "cerebro": "", "linea_res": "premio"}
APUESTAS = ["PLAYER", "BANKER", "TIE"]
PAGOS = {"PLAYER": 1.0, "BANKER": 0.95, "TIE": 8.0}

for _ in range(40):
    paso_cerebro()

def orientacion():
    o = obs["fly_orientation"]
    return np.arctan2(o[1], o[0])

def accion_hacia(objetivo):
    p = obs["fly"][0][:2]
    d = np.array(objetivo) - p
    err = np.arctan2(d[1], d[0]) - orientacion()
    err = (err + np.pi) % (2 * np.pi) - np.pi
    giro = np.clip(1.6 * err, -1, 1)           # >0 = girar a la izquierda
    avance = 1.0 if abs(err) < 1.2 else 0.6
    return np.clip([avance * (1 - 0.8 * giro), avance * (1 + 0.8 * giro)], 0.1, 1.4), np.linalg.norm(d)

def nueva_fase(f):
    estado["fase"] = f
    estado["t_fase"] = 0.0

print(f"Pasada 1: jugando y grabando {SEGUNDOS:.0f} s ({FOTOGRAMAS} fotogramas)...", flush=True)
accion = np.zeros(2)
hechos = 0
PALO_TXT = {"S": "♠", "H": "♥", "D": "♦", "C": "♣"}

def carta_txt(c):
    return f"{c[0]}{PALO_TXT[c[1]]}"

log = [("info", f">>> día {DIA}: cerebro cargado ({N:,} neuronas, {W.nnz:,} sinapsis)"),
       ("info", f">>> fichas iniciales: {estado['dinero']:.2f}")]
while hechos < FOTOGRAMAS and estado["ronda"] < N_RONDAS:
    obs, *_ = sim.step(accion)
    frames = sim.render()
    if not frames or frames[0] is None:
        continue

    # ---------- un paso de cerebro por fotograma ----------
    estado["t_fase"] += DT_FRAME
    extra[:] = 0.0
    linea = "paso"
    if estado["dopa"] > 0:
        extra[PAM if estado["gana"] else PPL1] = 1.4          # <- dopa
        estado["dopa"] -= 1
        linea = "dopa"
    disp = paso_cerebro()
    k = estado["ronda"]
    p, b, orden, ganador = manos[k]
    f = estado["fase"]
    t_txt = f"[{hechos * DT_FRAME:6.3f}s]"

    if f == "DECIDE":
        accion = np.zeros(2)
        if estado["t_fase"] <= DT_FRAME * 1.01:
            log.append(("info", f"{t_txt} ronda {k + 1}: estimulando neuronas sensoriales..."))
        for i, ap in enumerate(APUESTAS):
            g = GRUPOS[ap]
            estado["evidencia"][i] += disp[g].sum() + 0.05 * v[g].clip(0).sum()  # <- vota
        estado["kc_act"] |= disp & KC
        ev = estado["evidencia"]
        linea = "vota"
        estado["cerebro"] = ("MBON   P {:.1f}   B {:.1f}   T {:.1f}".format(*ev))
        if hechos % 6 == 0:
            log.append(("info", f"{t_txt} votos MBON  P={ev[0]:5.1f}  B={ev[1]:5.1f}  T={ev[2]:5.1f}"))
        if estado["t_fase"] > 0.12:
            if ev.max() <= 0:
                ev = ev + rng.random(3) * 1e-3
            estado["apuesta"] = APUESTAS[int(np.argmax(ev))]       # <- decide
            linea = "decide"
            estado["cerebro"] = f"DECISION  ->  {estado['apuesta']}"
            log.append(("dec", f"{t_txt} apuesta = \"{estado['apuesta']}\"   "
                               f"({top3[APUESTAS.index(estado['apuesta'])]} gana la votación)"))
            nueva_fase("CAMINA")
    elif f == "CAMINA":
        objetivo = CASILLAS[estado["apuesta"]]
        accion, dist = accion_hacia(objetivo)                     # <- camina
        linea = "camina"
        if hechos % 10 == 0:
            log.append(("info", f"{t_txt} caminando  dist={dist:5.2f} mm  "
                                f"patas izq={accion[0]:.2f} der={accion[1]:.2f}"))
        if dist < 1.0 or estado["t_fase"] > 1.6:
            accion = np.zeros(2)
            x, y = obs["fly"][0][:2]
            mover("ficha", (x - 1.6, y, 0.12)); mover("ficha_borde", (x - 1.6, y, 0.125))
            log.append(("dec", f"{t_txt} ficha colocada en {estado['apuesta']}"))
            nueva_fase("REPARTO")
            estado["repartidas"] = 0
    elif f == "REPARTO":
        accion = np.zeros(2)
        DUR = 0.07                                   # s de simulación por carta
        n = min(len(orden), int(estado["t_fase"] / DUR) + 1)
        for idx_c in range(n):
            mano, j = orden[idx_c]
            hx, hy, giro = hueco(mano, j)
            a = min(1.0, (estado["t_fase"] - idx_c * DUR) / DUR)
            a = 1 - (1 - a) ** 3                     # frena al llegar
            x = ZAPATO[0] + (hx - ZAPATO[0]) * a
            y = ZAPATO[1] + (hy - ZAPATO[1]) * a
            mover(f"carta_{k}_{mano}{j}", (x, y, 0.006 + 0.002 * idx_c), giro * a)  # <- reparte
        linea = "reparte"
        if n > estado["repartidas"]:
            mano, j = orden[n - 1]
            c = (p if mano == "P" else b)[j]
            quien = "PLAYER" if mano == "P" else "BANKER"
            log.append(("card", f"{t_txt} carta {n}: {quien:<6} recibe {carta_txt(c)}"))
            estado["repartidas"] = n
        cart_p = [p[j] for (m, j) in orden[:n] if m == "P"]
        cart_b = [b[j] for (m, j) in orden[:n] if m == "B"]
        estado["p_pts"], estado["b_pts"] = puntos(cart_p), puntos(cart_b)
        estado["cerebro"] = f"REPARTO   carta {n} / {len(orden)}"
        if estado["t_fase"] > DUR * (len(orden) + 1.5):
            gana = estado["apuesta"] == ganador
            estado["gana"] = gana
            log.append(("info", f"{t_txt} PLAYER {puntos(p)} vs BANKER {puntos(b)}  ->  gana {ganador}"))
            if gana:
                estado["dinero"] += PAGOS[ganador]
                estado["resultado"] = f"GANA {ganador}   +{PAGOS[ganador]:.2f}"
                aprender(estado["apuesta"], estado["kc_act"], 1.15)   # <- premio
                linea = estado["linea_res"] = "premio"
                estado["cerebro"] = "DOPAMINA (PAM)  ->  refuerza Kenyon-MBON"
                log.append(("win", f"{t_txt} ¡acierto! +{PAGOS[ganador]:.2f}  fichas={estado['dinero']:.2f}"))
                log.append(("dopa", f"{t_txt} dopamina PAM x{len(PAM)}  ->  "
                                    f"W[KC->{top3[APUESTAS.index(estado['apuesta'])]}] *= 1.15"))
            else:
                estado["dinero"] -= 1.0
                estado["resultado"] = f"GANA {ganador}   -1.00"
                aprender(estado["apuesta"], estado["kc_act"], 0.93)   # <- castigo
                linea = estado["linea_res"] = "castigo"
                estado["cerebro"] = "DOPAMINA (PPL1)  ->  debilita Kenyon-MBON"
                log.append(("lose", f"{t_txt} fallo  -1.00  fichas={estado['dinero']:.2f}"))
                log.append(("dopa", f"{t_txt} dopamina PPL1 x{len(PPL1)}  ->  "
                                    f"W[KC->{top3[APUESTAS.index(estado['apuesta'])]}] *= 0.93"))
            estado["dopa"] = 10
            nueva_fase("RESULTADO")
    elif f == "RESULTADO":
        accion = np.zeros(2)
        linea = estado["linea_res"]
        if estado["t_fase"] > 0.30:
            for mano, j in orden:
                mover(f"carta_{k}_{mano}{j}", ESCONDIDO)
            mover("ficha", ESCONDIDO); mover("ficha_borde", ESCONDIDO)
            estado.update(ronda=k + 1, apuesta=None, p_pts=None, b_pts=None,
                          resultado=None, evidencia=np.zeros(3),
                          kc_act=np.zeros(N, bool), cerebro="")
            nueva_fase("DECIDE")

    textos_fase = {"DECIDE": "el cerebro decide...", "CAMINA": "camina a su apuesta",
                   "REPARTO": "reparto de cartas", "RESULTADO": "resultado"}
    hud.append({"ronda": min(estado["ronda"], N_RONDAS - 1) + 1,
                "fase": textos_fase[estado["fase"]], "apuesta": estado["apuesta"],
                "p_pts": estado["p_pts"], "b_pts": estado["b_pts"],
                "dinero": estado["dinero"], "resultado": estado["resultado"],
                "gana": estado["gana"], "spikes": int(disp.sum()),
                "cerebro": estado["cerebro"], "linea": f"# <- {linea}", "log": log})
    log = []
    brillo_disco[hechos] = (brillo * 255).astype(np.uint8)
    writer.append_data(frames[0][:IZQ_H, :IZQ_W, :3])
    hechos += 1
    if hechos % FPS == 0:
        print(f"  {hechos // FPS} s / {SEGUNDOS:.0f} s   ronda {estado['ronda'] + 1}"
              f"   fase {estado['fase']}", flush=True)

writer.close()
brillo_disco.flush()
json.dump({"dia": DIA, "frames": hud}, open(f"{TMP}/hud.json", "w"))
sim.close()

# Guardar lo aprendido para el día siguiente
filas = np.concatenate([GRUPOS[a] for a in APUESTAS])
pesos = np.concatenate([W.data[W.indptr[r]:W.indptr[r + 1]] for r in filas])
np.savez(MEMORIA, dia=DIA, dinero=estado["dinero"], filas=filas, pesos=pesos)
print(f"Memoria guardada (día {DIA}, fichas {estado['dinero']:.2f})", flush=True)

subprocess.run([sys.executable, __file__, str(SEGUNDOS), "--panel"], check=True)
