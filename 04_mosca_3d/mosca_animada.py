"""
MOSCA ANIMADA CON CEREBRO REAL
------------------------------
Izquierda: cuerpo de mosca NeuroMechFly v2 (87 articulaciones, física MuJoCo).
Derecha:   su cerebro (139.248 neuronas de FlyWire) disparando en tiempo real.

Cómo se conectan (igual que Eon Systems, 2026):
  · El cerebro (modelo de Shiu et al. 2024) recibe estímulos sensoriales.
  · Leemos sus neuronas descendentes, que en la mosca real mandan órdenes
    al cuerpo:
        DNa01 / DNa02 (izq. y der.) -> hacia dónde gira
        DNp09                       -> si anda hacia delante
  · Esas órdenes mueven el controlador de patas de NeuroMechFly.

Uso (en el entorno 'moscafly'):
    python mosca_animada.py              -> graba 20 s en mosca_animada.mp4
    python mosca_animada.py 8            -> graba 8 s (para probar rápido)
"""

import sys
import subprocess
import numpy as np
import pandas as pd
import scipy.sparse as sp
import imageio

# Se hace en dos pasadas (cada una en su propio proceso) para que MuJoCo y
# PyVista no se peleen por la tarjeta gráfica:
#   1) cuerpo: simula cerebro + mosca y graba mosca_cuerpo.mp4
#   2) panel:  repite el MISMO cerebro (misma semilla), dibuja el panel
#              y lo junta todo en mosca_animada.mp4
PASADA_PANEL = "--panel" in sys.argv
_nums = [a for a in sys.argv[1:] if not a.startswith("--")]

CARPETA = "datos_flywire"
SEGUNDOS_VIDEO = float(_nums[0]) if _nums else 20.0
FPS = 30
VELOCIDAD = 0.2          # el vídeo va a cámara lenta x0.2, como los originales
ANCHO, ALTO = 1920, 1080
ANCHO_MOSCA = 1280       # el resto (640 px) es el panel del cerebro

# ==================================================================
# 1. CEREBRO
# ==================================================================
print("Cargando cerebro...", flush=True)
ann = pd.read_csv(f"{CARPETA}/Supplemental_file1_neuron_annotations.tsv",
                  sep="\t", low_memory=False,
                  usecols=["root_id", "pos_x", "pos_y", "pos_z",
                           "super_class", "cell_type", "side"])
ann = ann.dropna(subset=["pos_x", "pos_y", "pos_z"]).reset_index(drop=True)
N = len(ann)

pos = ann[["pos_x", "pos_y", "pos_z"]].values.astype(np.float64)
pos *= np.array([4.0, 4.0, 40.0]) / 1000.0        # a micras
lo, hi = np.percentile(pos, 0.5, axis=0), np.percentile(pos, 99.5, axis=0)
pos -= (lo + hi) / 2
pos[:, 1] *= -1
pos = pos.astype(np.float32)

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

tipo = ann["cell_type"].astype(str)
lado = ann["side"].astype(str)
DN_GIRO_IZQ = np.where(tipo.isin(["DNa01", "DNa02"]) & (lado == "left"))[0]
DN_GIRO_DER = np.where(tipo.isin(["DNa01", "DNa02"]) & (lado == "right"))[0]
DN_AVANZA = np.where(tipo == "DNp09")[0]
sensoriales = np.where(ann["super_class"].astype(str).str.contains("sensory"))[0]
print(f"  neuronas de control: giro izq {len(DN_GIRO_IZQ)}, "
      f"giro der {len(DN_GIRO_DER)}, avance {len(DN_AVANZA)}", flush=True)

v = np.zeros(N, np.float32)
refractario = np.zeros(N, np.int8)
brillo = np.zeros(N, np.float32)
rng = np.random.default_rng(3)
foco = {"centro": rng.choice(sensoriales), "t": 0}
traza = {"izq": 0.0, "der": 0.0, "avanza": 0.0}

def paso_cerebro():
    """Un paso del modelo integra-y-dispara sobre el conectoma real."""
    global v
    disparos = (v > 1.0) & (refractario == 0)
    v[disparos] = 0.0
    refractario[disparos] = 4
    refractario[refractario > 0] -= 1
    v = 0.80 * v + 0.016 * (W @ disparos.astype(np.float32))
    v[refractario > 0] = 0.0
    foco["t"] += 1
    if foco["t"] % 45 == 0:
        foco["centro"] = rng.choice(sensoriales)
    cerca = sensoriales[np.linalg.norm(pos[sensoriales] - pos[foco["centro"]], axis=1) < 70]
    v[rng.choice(cerca, size=min(len(cerca), 60))] += 1.5
    v[rng.integers(0, N, 80)] += 1.2
    np.clip(v, -2, 3, out=v)
    brillo[:] = brillo * 0.72
    brillo[disparos] = 1.0
    # Rastro suavizado de las neuronas descendentes
    k = 0.06
    traza["izq"] += k * (disparos[DN_GIRO_IZQ].sum() - traza["izq"])
    traza["der"] += k * (disparos[DN_GIRO_DER].sum() - traza["der"])
    traza["avanza"] += k * (disparos[DN_AVANZA].sum() - traza["avanza"])
    return int(disparos.sum())

def ordenes_al_cuerpo():
    """Traduce la actividad descendente a la entrada [izq, der] del controlador."""
    giro = np.tanh(4.0 * (traza["izq"] - traza["der"]))   # >0 gira a la izquierda
    avance = 0.75 + 0.45 * np.tanh(3.0 * traza["avanza"])
    izq = avance * (1.0 - 0.55 * giro)
    der = avance * (1.0 + 0.55 * giro)
    return np.clip([izq, der], 0.2, 1.4), giro, avance

# ==================================================================
# 2. PASADA 2: PANEL DEL CEREBRO (PyVista) + montaje final
# ==================================================================
if PASADA_PANEL:
    import cv2
    from PIL import Image, ImageDraw, ImageFont
    ANCHO_PANEL = ANCHO - ANCHO_MOSCA
    MITAD = ALTO // 2

    def fuente(tam):
        for f in ("arial.ttf", "DejaVuSans.ttf",
                  "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
            try:
                return ImageFont.truetype(f, tam)
            except OSError:
                pass
        return ImageFont.load_default()

    # El cerebro se dibuja con numpy (sin tarjeta gráfica): igual en todos los
    # ordenadores y más rápido que un render 3D de 139.000 puntos.
    ESCALA = ANCHO_PANEL / 860.0                 # píxeles por micra
    BASE = np.array([62, 68, 82], np.float32)
    ACTIVO = np.array([235, 242, 255], np.float32)
    HALO = np.array([0.45, 0.62, 1.0], np.float32)
    F_T, F_t, F_c = fuente(20), fuente(12), fuente(11)

    def vista(horiz, vert, prof, alto):
        """Proyecta las neuronas en un plano y las pinta con su actividad."""
        u = (ANCHO_PANEL / 2 + horiz * ESCALA).astype(np.int32)
        vv = (alto / 2 - vert * ESCALA).astype(np.int32)
        dentro = (u >= 0) & (u < ANCHO_PANEL) & (vv >= 0) & (vv < alto)
        p = (prof.max() - prof) / (np.ptp(prof) + 1e-6)
        b = brillo ** 0.8
        col = BASE * (0.35 + 0.65 * p[:, None])
        col = col * (1 - b[:, None]) + ACTIVO * b[:, None]
        img = np.zeros((alto, ANCHO_PANEL, 3), np.float32)
        todas = np.flatnonzero(dentro)
        img[vv[todas], u[todas]] = col[todas]
        vivas = todas[b[todas] > 0.02]
        vivas = vivas[np.argsort(b[vivas])]
        img[vv[vivas], u[vivas]] = col[vivas]
        act = np.zeros((alto, ANCHO_PANEL), np.float32)
        act[vv[vivas], u[vivas]] = b[vivas]
        img += cv2.GaussianBlur(act, (0, 0), 3.0)[..., None] * HALO * 160
        return np.clip(img, 0, 255).astype(np.uint8)

    def render_panel(t, n_spk, giro, avance):
        a = np.deg2rad(22 * np.sin(t / 90.0))
        ca, sa = np.cos(a), np.sin(a)
        frontal = vista(pos[:, 0] * ca + pos[:, 2] * sa, pos[:, 1],
                        -pos[:, 0] * sa + pos[:, 2] * ca, MITAD)
        dorsal = vista(pos[:, 0], pos[:, 2], -pos[:, 1], MITAD)
        im = Image.fromarray(np.vstack([frontal, dorsal]))
        d = ImageDraw.Draw(im)
        gris = (115, 120, 140)
        d.text((28, 24), "NEURAL ACTIVITY", font=F_T, fill=(235, 237, 244))
        d.text((28, 52), f"{N:,} neurons  ·  {W.nnz / 1e6:.1f}M connections",
               font=F_t, fill=gris)
        d.text((28, MITAD - 30), "FRONTAL", font=F_t, fill=gris)
        t_ = f"{n_spk:>5} spikes / step"
        d.text((ANCHO_PANEL - 28 - d.textlength(t_, font=F_t), MITAD - 30), t_,
               font=F_t, fill=gris)
        flecha = "<" if giro > 0.15 else (">" if giro < -0.15 else "|")
        d.text((28, MITAD + 24), f"DESCENDING   steer {flecha} {giro:+.2f}   "
                                f"forward {avance:.2f}", font=F_t, fill=(205, 209, 224))
        d.text((28, MITAD + 46), "FlyWire v783 · Shiu et al. 2024 · NeuroMechFly v2",
               font=F_c, fill=(77, 79, 95))
        d.text((28, ALTO - 30), "DORSAL", font=F_t, fill=gris)
        return np.asarray(im)

    lector = imageio.get_reader("mosca_cuerpo.mp4")
    salida = imageio.get_writer("mosca_animada.mp4", fps=FPS, quality=8,
                                macro_block_size=8)
    for _ in range(40):
        paso_cerebro()
    print("Pasada 2: dibujando el cerebro y montando el vídeo...", flush=True)
    for i, frame_mosca in enumerate(lector):
        n_spk = paso_cerebro()
        _, giro, avance = ordenes_al_cuerpo()
        panel = render_panel(i + 1, n_spk, giro, avance)
        lienzo = np.zeros((ALTO, ANCHO, 3), np.uint8)
        lienzo[:, :ANCHO_MOSCA] = frame_mosca[:ALTO, :ANCHO_MOSCA, :3]
        lienzo[:, ANCHO_MOSCA:] = panel[:ALTO, :ANCHO_PANEL]
        lienzo[:, ANCHO_MOSCA:ANCHO_MOSCA + 1] = 28      # línea separadora
        salida.append_data(lienzo)
        if (i + 1) % FPS == 0:
            print(f"  {(i + 1) // FPS} s", flush=True)
    salida.close()
    print("Guardado mosca_animada.mp4  <-- este es el bueno")
    sys.exit(0)

# ==================================================================
# 3. CUERPO DE LA MOSCA (NeuroMechFly)
# ==================================================================
from flygym_gymnasium import Fly, YawOnlyCamera
from flygym_gymnasium.arena import FlatTerrain
from flygym_gymnasium.examples.locomotion import HybridTurningController
from flygym_gymnasium.preprogrammed import default_leg_sensor_placements


class SueloOscuro(FlatTerrain):
    """Suelo casi negro, brillante, a juego con el panel."""
    def __init__(self):
        super().__init__(size=(400, 400))
        tex = self.root_element.asset.texture[0]
        tex.rgb1 = (0.055, 0.058, 0.066)
        tex.rgb2 = (0.075, 0.078, 0.088)
        mat = self.root_element.asset.material[0]
        mat.reflectance = 0.22
        mat.texrepeat = (80, 80)
        self.root_element.asset.add("texture", type="skybox", builtin="gradient",
                                    rgb1=(0.02, 0.02, 0.03), rgb2=(0, 0, 0),
                                    width=256, height=256)

print("Pasada 1: montando el cuerpo de la mosca...", flush=True)
fly = Fly(enable_adhesion=True, draw_adhesion=False,
          contact_sensor_placements=default_leg_sensor_placements,
          spawn_pos=(0, 0, 0.2))
glob = getattr(fly.model.visual, "global")      # permite renderizar en HD
glob.offwidth, glob.offheight = ANCHO, ALTO
cam = YawOnlyCamera(attachment_point=fly.model.worldbody,
                    camera_name="camera_left_top_zoomout",
                    targeted_fly_names=fly.name, play_speed=VELOCIDAD, fps=FPS,
                    window_size=(ANCHO_MOSCA, ALTO), play_speed_text=False)
sim = HybridTurningController(fly=fly, cameras=[cam], timestep=1e-4,
                              arena=SueloOscuro())
sim.reset(seed=0)

# ==================================================================
# 4. BUCLE: cerebro -> órdenes -> cuerpo -> vídeo
# ==================================================================
salida = imageio.get_writer("mosca_cuerpo.mp4", fps=FPS, quality=9,
                            macro_block_size=8)
fotogramas = int(SEGUNDOS_VIDEO * FPS)
print(f"Grabando {SEGUNDOS_VIDEO:.0f} s de vídeo ({fotogramas} fotogramas)...", flush=True)

for _ in range(40):                 # calentar el cerebro
    paso_cerebro()

accion, giro, avance = ordenes_al_cuerpo()
hechos = 0
while hechos < fotogramas:
    sim.step(accion)
    frames = sim.render()
    frame_mosca = frames[0] if frames else None
    if frame_mosca is None:
        continue
    paso_cerebro()                  # un paso de cerebro por fotograma
    accion, giro, avance = ordenes_al_cuerpo()
    salida.append_data(frame_mosca[:ALTO, :ANCHO_MOSCA, :3])
    hechos += 1
    if hechos % FPS == 0:
        print(f"  {hechos // FPS} s / {SEGUNDOS_VIDEO:.0f} s", flush=True)
salida.close()
sim.close()

# Lanza la pasada 2 en un proceso nuevo
subprocess.run([sys.executable, __file__, str(SEGUNDOS_VIDEO), "--panel"], check=True)
