"""
Visor 3D del cerebro de la mosca jugando al baccarat.
Las neuronas son esferas en el espacio y las sinapsis son tubos finos.
Al disparar una neurona se ilumina. Al reforzar una sinapsis, su tubo engorda.

Esta ventana se puede abrir junto al visor 2D normal.
Dentro de la ventana 3D puedes rotar con el ratón.
"""

import numpy as np
import random
import time
import pyvista as pv
from pyvistaqt import BackgroundPlotter
from PyQt5 import QtCore

from baccarat import jugar_mano
from mosca_v2 import MoscaBaccarat

# ---------- Posiciones 3D ----------
def layout_3d():
    """Coloca las neuronas en el espacio como racimos."""
    pos = {}

    # 5 sensores en una línea vertical en x=-6
    for i in range(5):
        pos[("sensor", i)] = np.array([-6, 0, (i - 2) * 1.5])

    # 10 player en un racimo cerca de (0, -5, z)
    rng = np.random.default_rng(1)
    for i in range(10):
        base = np.array([0, -5, 0])
        pos[("player", i)] = base + rng.normal(0, 1.0, 3)

    # 10 banker en racimo cerca de (0, 0, z)
    rng = np.random.default_rng(2)
    for i in range(10):
        base = np.array([0, 0, 0])
        pos[("banker", i)] = base + rng.normal(0, 1.0, 3)

    # 10 tie en racimo cerca de (0, 5, z)
    rng = np.random.default_rng(3)
    for i in range(10):
        base = np.array([0, 5, 0])
        pos[("tie", i)] = base + rng.normal(0, 1.0, 3)

    return pos


COLORES = {
    "sensor": (0.47, 0.70, 1.00),
    "player": (1.00, 0.43, 0.43),
    "banker": (0.47, 0.90, 0.55),
    "tie":    (1.00, 0.90, 0.43),
}


def estado_a_sensores(historico):
    ultimos = historico[-10:] if historico else []
    n = max(len(ultimos), 1)
    return np.array([
        ultimos.count("player") / n,
        ultimos.count("banker") / n,
        ultimos.count("tie")    / n,
        min(len(historico) / 500, 1.0),
        random.random()
    ])


class Visor3D:
    def __init__(self):
        self.mosca = MoscaBaccarat(semilla=random.randint(0, 10000))
        self.pos = layout_3d()
        self.historico = []
        self.dinero = 1000.0
        self.aciertos = 0
        self.ronda = 0

        self.plotter = BackgroundPlotter(
            title="Cerebro 3D de la mosca",
            window_size=(900, 700),
        )
        self.plotter.set_background("#121218")

        # Esferas de las neuronas
        self.actores_neuronas = {}
        for clave, p in self.pos.items():
            grupo, _ = clave
            esfera = pv.Sphere(radius=0.35, center=p)
            color = COLORES[grupo]
            actor = self.plotter.add_mesh(
                esfera, color=color, smooth_shading=True,
                specular=0.5, specular_power=15
            )
            self.actores_neuronas[clave] = (actor, color)

        # Tubos de las sinapsis
        self.actores_sinapsis = {}  # (grupo, i_pre, i_post) -> actor
        self._dibujar_sinapsis()

        # Texto HUD
        self.hud = self.plotter.add_text(
            "Preparando...", position="upper_left",
            font_size=11, color="white"
        )

        # Camara inicial
        self.plotter.camera_position = [(14, -14, 10), (0, 0, 0), (0, 0, 1)]

        # Timer para jugar manos
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.jugar_una_mano)
        self.timer.start(1200)  # ms entre manos

    def _dibujar_sinapsis(self):
        """Dibuja todas las sinapsis como tubos. Se vuelven a dibujar al actualizar."""
        # Borrar las que haya
        for actor in list(self.actores_sinapsis.values()):
            try:
                self.plotter.remove_actor(actor)
            except Exception:
                pass
        self.actores_sinapsis = {}

        sinapsis = [(self.mosca.S_player, "player"),
                    (self.mosca.S_banker, "banker"),
                    (self.mosca.S_tie,    "tie")]

        for S, nombre in sinapsis:
            pre = np.array(S.i)
            post = np.array(S.j)
            ws = np.array(S.w)
            for a, b, w in zip(pre, post, ws):
                if w < 0.05:
                    continue  # ocultar sinapsis irrelevantes
                p1 = self.pos[("sensor", int(a))]
                p2 = self.pos[(nombre, int(b))]
                linea = pv.Line(p1, p2)
                radio = float(0.02 + w * 0.14)
                tubo = linea.tube(radius=radio)
                intensidad = float(np.clip(w, 0, 1))
                color = (0.4 + intensidad * 0.6,
                         0.4 + intensidad * 0.6,
                         0.5 + intensidad * 0.3)
                actor = self.plotter.add_mesh(
                    tubo, color=color, smooth_shading=True, opacity=0.85
                )
                self.actores_sinapsis[(nombre, int(a), int(b))] = actor

    def resaltar(self, grupo_activo):
        """Pone en blanco las neuronas del grupo que ha disparado."""
        for clave, (actor, color_base) in self.actores_neuronas.items():
            grupo, _ = clave
            if grupo == grupo_activo:
                actor.GetProperty().SetColor(1.0, 1.0, 1.0)
                actor.GetProperty().SetAmbient(0.6)
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
            self.mosca.dopamina(apuesta, 0.02)
            msg = f"GANÓ +{pagos[apuesta]:.2f}"
        else:
            self.dinero -= 1.0
            self.mosca.dopamina(apuesta, -0.005)
            msg = "PERDIÓ -1.00"

        # Resaltar visualmente
        self.resaltar(apuesta)

        # Redibujar sinapsis cada 10 manos (es más costoso)
        if self.ronda % 10 == 0:
            self._dibujar_sinapsis()

        # Actualizar HUD
        tasa = self.aciertos / self.ronda * 100
        texto = (
            f"Ronda {self.ronda}\n"
            f"Dinero: {self.dinero:.1f}\n"
            f"Aciertos: {self.aciertos} ({tasa:.1f}%)\n"
            f"Apostó: {apuesta.upper()}\n"
            f"Salió:  {resultado.upper()}\n"
            f"{msg}"
        )
        self.hud.SetText(2, texto)  # 2 = esquina superior izquierda

        self.plotter.render()


def main():
    app = pv.global_theme  # noqa: forzar inicialización
    from PyQt5.QtWidgets import QApplication
    import sys
    qapp = QApplication.instance() or QApplication(sys.argv)
    v = Visor3D()
    qapp.exec_()


if __name__ == "__main__":
    main()