"""
Visor 3D v2.
Tres capas: sensores (azul), intermedia (morado), decisiones (rojo/verde/amarillo).
Las sinapsis engordan con el aprendizaje.
Al disparar una neurona se ilumina en blanco.
"""

import numpy as np
import random
import pyvista as pv
from pyvistaqt import BackgroundPlotter
from PyQt5 import QtCore

from baccarat import jugar_mano
from mosca_v2 import MoscaBaccarat


# ---------- Posiciones 3D ----------
def layout_3d():
    pos = {}

    # 11 sensores a la izquierda, columna
    for i in range(11):
        pos[("sensor", i)] = np.array([-9.0, 0.0, (i - 5) * 1.2])

    # 20 intermedias en el centro, en racimo
    rng = np.random.default_rng(10)
    for i in range(20):
        base = np.array([-2.0, 0.0, 0.0])
        pos[("inter", i)] = base + rng.normal(0, 2.2, 3)

    # 10 player, 10 banker, 10 tie a la derecha
    for grupo, y in [("player", -6), ("banker", 0), ("tie", 6)]:
        rng = np.random.default_rng(hash(grupo) % 1000)
        for i in range(10):
            base = np.array([6.0, y, 0.0])
            pos[(grupo, i)] = base + rng.normal(0, 1.1, 3)

    return pos


COLORES = {
    "sensor": (0.47, 0.70, 1.00),
    "inter":  (0.75, 0.55, 1.00),
    "player": (1.00, 0.43, 0.43),
    "banker": (0.47, 0.90, 0.55),
    "tie":    (1.00, 0.90, 0.43),
}


def estado_a_sensores(historico):
    ultimos = historico[-10:] if historico else []
    n = max(len(ultimos), 1)
    sensores = [
        ultimos.count("player") / n,
        ultimos.count("banker") / n,
        ultimos.count("tie")    / n,
    ]
    codif = {"player": 0.2, "banker": 0.7, "tie": 1.0}
    ultimas8 = historico[-8:] if historico else []
    for h in ultimas8:
        sensores.append(codif[h])
    while len(sensores) < 11:
        sensores.append(0.0)
    return np.array(sensores)


class Visor3D:
    def __init__(self):
        self.mosca = MoscaBaccarat(semilla=random.randint(0, 10000))
        self.pos = layout_3d()
        self.historico = []
        self.dinero = 1000.0
        self.aciertos = 0
        self.ronda = 0

        self.plotter = BackgroundPlotter(
            title="Cerebro 3D de la mosca v2",
            window_size=(1100, 800),
        )
        self.plotter.set_background("#121218")

        # Neuronas
        self.actores_neuronas = {}
        for clave, p in self.pos.items():
            grupo, _ = clave
            radio = 0.30 if grupo == "inter" else 0.35
            esfera = pv.Sphere(radius=radio, center=p)
            color = COLORES[grupo]
            actor = self.plotter.add_mesh(
                esfera, color=color, smooth_shading=True,
                specular=0.4, specular_power=15
            )
            self.actores_neuronas[clave] = (actor, color)

        # Sinapsis
        self.actores_sinapsis = []
        self._dibujar_sinapsis()

        # HUD
        self.hud = self.plotter.add_text(
            "Preparando...", position="upper_left",
            font_size=11, color="white"
        )

        self.plotter.camera_position = [(20, -20, 14), (0, 0, 0), (0, 0, 1)]

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.jugar_una_mano)
        self.timer.start(1000)

    def _dibujar_sinapsis(self):
        for actor in self.actores_sinapsis:
            try:
                self.plotter.remove_actor(actor)
            except Exception:
                pass
        self.actores_sinapsis = []

        # Capa 1: sensor -> inter
        S = self.mosca.S_in
        pre, post = np.array(S.i), np.array(S.j)
        ws = np.array(S.w)
        for a, b, w in zip(pre, post, ws):
            if w < 0.08:
                continue
            p1 = self.pos[("sensor", int(a))]
            p2 = self.pos[("inter",  int(b))]
            self._tubo(p1, p2, w, tono="azul")

        # Capa 2: inter -> cada decisión
        for S, nombre, tono in [
            (self.mosca.S_player, "player", "rojo"),
            (self.mosca.S_banker, "banker", "verde"),
            (self.mosca.S_tie,    "tie",    "amarillo"),
        ]:
            pre, post = np.array(S.i), np.array(S.j)
            ws = np.array(S.w)
            for a, b, w in zip(pre, post, ws):
                if w < 0.08:
                    continue
                p1 = self.pos[("inter", int(a))]
                p2 = self.pos[(nombre,  int(b))]
                self._tubo(p1, p2, w, tono=tono)

    def _tubo(self, p1, p2, w, tono="gris"):
        linea = pv.Line(p1, p2)
        radio = float(0.02 + w * 0.14)
        tubo = linea.tube(radius=radio)
        intensidad = float(np.clip(w, 0, 1))
        if tono == "verde":
            color = (0.3 + 0.4*intensidad, 0.7 + 0.3*intensidad, 0.3 + 0.2*intensidad)
        elif tono == "rojo":
            color = (0.8 + 0.2*intensidad, 0.3 + 0.3*intensidad, 0.3 + 0.2*intensidad)
        elif tono == "amarillo":
            color = (0.8 + 0.2*intensidad, 0.75 + 0.2*intensidad, 0.3 + 0.2*intensidad)
        elif tono == "azul":
            color = (0.3 + 0.3*intensidad, 0.5 + 0.3*intensidad, 0.8 + 0.2*intensidad)
        else:
            color = (0.5, 0.5, 0.5)
        actor = self.plotter.add_mesh(tubo, color=color, smooth_shading=True, opacity=0.85)
        self.actores_sinapsis.append(actor)

    def resaltar(self, grupo_activo, intermedias_activas):
        for clave, (actor, color_base) in self.actores_neuronas.items():
            grupo, idx = clave
            if grupo == grupo_activo:
                actor.GetProperty().SetColor(1.0, 1.0, 1.0)
                actor.GetProperty().SetAmbient(0.6)
            elif grupo == "inter" and intermedias_activas[idx] > 0:
                nivel = min(1.0, 0.3 + intermedias_activas[idx] / 5.0)
                r = color_base[0] + (1 - color_base[0]) * nivel
                g = color_base[1] + (1 - color_base[1]) * nivel
                b = color_base[2] + (1 - color_base[2]) * nivel
                actor.GetProperty().SetColor(r, g, b)
                actor.GetProperty().SetAmbient(0.3)
            else:
                actor.GetProperty().SetColor(*color_base)
                actor.GetProperty().SetAmbient(0.1)

    def jugar_una_mano(self):
        sensores = estado_a_sensores(self.historico)
        apuesta = self.mosca.decidir(sensores)
        resultado = jugar_mano()
        self.historico.append(resultado)
        self.ronda += 1

        pagos = {"player": 1.0, "banker": 0.95, "tie": 8.0}
        if apuesta == resultado:
            self.dinero += pagos[apuesta]
            self.aciertos += 1
            self.mosca.dopamina(apuesta, 0.03)
            msg = f"GANÓ +{pagos[apuesta]:.2f}"
        else:
            self.dinero -= 1.0
            self.mosca.dopamina(apuesta, -0.008)
            msg = "PERDIÓ -1.00"

        self.resaltar(apuesta, self.mosca.ultima_actividad_intermedia)

        if self.ronda % 10 == 0:
            self._dibujar_sinapsis()

        tasa = self.aciertos / self.ronda * 100
        self.hud.SetText(2,
            f"Ronda {self.ronda}\n"
            f"Dinero: {self.dinero:.1f}\n"
            f"Aciertos: {self.aciertos} ({tasa:.1f}%)\n"
            f"Apostó: {apuesta.upper()}\n"
            f"Salió:  {resultado.upper()}\n"
            f"{msg}"
        )
        self.plotter.render()

def main():
    v = Visor3D()
    # Fuerza un primer disparo manual para que arranque
    v.jugar_una_mano()
    v.plotter.app.exec_()


if __name__ == "__main__":
    main()


