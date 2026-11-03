"""
Visor en tiempo real.
A la izquierda: el cerebro de la mosca (neuronas que se iluminan al disparar).
A la derecha: la mesa de baccarat, con apuesta, dinero y resultados.
"""

import pygame
import numpy as np
import random
import time
from baccarat import jugar_mano
from mosca import MoscaBaccarat

# -------- Config de la ventana --------
ANCHO, ALTO = 1200, 700
FONDO = (18, 18, 24)
COLOR_TEXTO = (230, 230, 230)
COLOR_SENSOR = (120, 180, 255)
COLOR_PLAYER = (255, 110, 110)
COLOR_BANKER = (120, 230, 140)
COLOR_TIE    = (255, 230, 110)
COLOR_SINAPSIS = (90, 90, 110)

# -------- Posiciones de las neuronas en pantalla --------
def layout_neuronas():
    pos = {}
    # 5 sensores a la izquierda, en columna
    for i in range(5):
        pos[("sensor", i)] = (120, 120 + i * 80)
    # 10 player
    for i in range(10):
        pos[("player", i)] = (320, 70 + i * 55)
    # 10 banker
    for i in range(10):
        pos[("banker", i)] = (470, 70 + i * 55)
    # 10 tie
    for i in range(10):
        pos[("tie",    i)] = (620, 70 + i * 55)
    return pos


def dibujar_cerebro(pantalla, mosca, activaciones, pos):
    # Dibujar sinapsis primero (para que queden detrás)
    sinapsis = [("S_player", mosca.S_player, "player"),
                ("S_banker", mosca.S_banker, "banker"),
                ("S_tie",    mosca.S_tie,    "tie")]
    for _, S, nombre in sinapsis:
        pre = np.array(S.i)
        post = np.array(S.j)
        w = np.array(S.w)
        for a, b, ww in zip(pre, post, w):
            x1, y1 = pos[("sensor", int(a))]
            x2, y2 = pos[(nombre, int(b))]
            gris = int(60 + ww * 180)
            pygame.draw.line(pantalla, (gris, gris, gris + 20),
                             (x1, y1), (x2, y2), max(1, int(ww * 3)))

    # Neuronas
    for (grupo, idx), (x, y) in pos.items():
        if grupo == "sensor":
            color_base = COLOR_SENSOR
        elif grupo == "player":
            color_base = COLOR_PLAYER
        elif grupo == "banker":
            color_base = COLOR_BANKER
        else:
            color_base = COLOR_TIE

        actividad = activaciones.get((grupo, idx), 0.0)
        # Mezclar color base con blanco según actividad
        r = min(255, int(color_base[0] + actividad * (255 - color_base[0])))
        g = min(255, int(color_base[1] + actividad * (255 - color_base[1])))
        b = min(255, int(color_base[2] + actividad * (255 - color_base[2])))
        pygame.draw.circle(pantalla, (r, g, b), (x, y), 14)
        pygame.draw.circle(pantalla, (20, 20, 30), (x, y), 14, 2)


def dibujar_panel(pantalla, fuente, fuente_pequena,
                  estado, dinero, dinero_inicial,
                  aciertos, ronda, historico):
    x0 = 760
    y = 40
    pantalla.blit(fuente.render("MESA DE BACCARAT", True, COLOR_TEXTO), (x0, y)); y += 50

    pantalla.blit(fuente_pequena.render(f"Ronda: {ronda}", True, COLOR_TEXTO), (x0, y)); y += 30
    pantalla.blit(fuente_pequena.render(f"Dinero: {dinero:.1f} (inicio {dinero_inicial:.0f})",
                                        True, COLOR_TEXTO), (x0, y)); y += 30
    tasa = (aciertos / ronda * 100) if ronda > 0 else 0
    pantalla.blit(fuente_pequena.render(f"Aciertos: {aciertos} ({tasa:.1f}%)", True, COLOR_TEXTO),
                  (x0, y)); y += 40

    pantalla.blit(fuente_pequena.render("Última apuesta:", True, COLOR_TEXTO), (x0, y)); y += 30
    color_ap = {"player": COLOR_PLAYER, "banker": COLOR_BANKER, "tie": COLOR_TIE}.get(
        estado.get("apuesta"), COLOR_TEXTO)
    pantalla.blit(fuente.render(estado.get("apuesta", "-").upper(), True, color_ap), (x0, y)); y += 50

    pantalla.blit(fuente_pequena.render("Resultado:", True, COLOR_TEXTO), (x0, y)); y += 30
    color_res = {"player": COLOR_PLAYER, "banker": COLOR_BANKER, "tie": COLOR_TIE}.get(
        estado.get("resultado"), COLOR_TEXTO)
    pantalla.blit(fuente.render(estado.get("resultado", "-").upper(), True, color_res),
                  (x0, y)); y += 50

    msg = estado.get("mensaje", "")
    if msg:
        color_msg = (120, 230, 140) if "GAN" in msg else (230, 110, 110)
        pantalla.blit(fuente.render(msg, True, color_msg), (x0, y)); y += 50

    # Histórico (últimos 20)
    pantalla.blit(fuente_pequena.render("Histórico:", True, COLOR_TEXTO), (x0, y)); y += 30
    ultimos = historico[-20:]
    xx = x0
    for r in ultimos:
        c = {"player": COLOR_PLAYER, "banker": COLOR_BANKER, "tie": COLOR_TIE}[r]
        pygame.draw.circle(pantalla, c, (xx + 10, y + 10), 8)
        xx += 20
        if xx > x0 + 380:
            xx = x0
            y += 25


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


def main():
    pygame.init()
    pantalla = pygame.display.set_mode((ANCHO, ALTO))
    pygame.display.set_caption("Mosca jugando al baccarat")
    fuente = pygame.font.SysFont("arial", 24, bold=True)
    fuente_pequena = pygame.font.SysFont("arial", 18)
    reloj = pygame.time.Clock()

    mosca = MoscaBaccarat()
    pos = layout_neuronas()

    historico = []
    dinero = 1000.0
    dinero_inicial = dinero
    aciertos = 0
    ronda = 0
    estado = {"apuesta": "-", "resultado": "-", "mensaje": ""}
    activaciones = {}
    ultimo_tiempo = time.time()
    pausa = False

    corriendo = True
    while corriendo:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                corriendo = False
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_SPACE:
                    pausa = not pausa
                if ev.key == pygame.K_ESCAPE:
                    corriendo = False

        # Jugar una mano cada 1.2 segundos
        if not pausa and time.time() - ultimo_tiempo > 1.2:
            ultimo_tiempo = time.time()
            sensores = estado_a_sensores(historico)
            apuesta = mosca.decidir(sensores)

            # Pintar activaciones
            activaciones = {("sensor", i): float(sensores[i]) for i in range(5)}
            for i in range(10):
                activaciones[("player", i)] = 1.0 if apuesta == "player" else 0.1
                activaciones[("banker", i)] = 1.0 if apuesta == "banker" else 0.1
                activaciones[("tie",    i)] = 1.0 if apuesta == "tie"    else 0.1

            resultado = jugar_mano()
            historico.append(resultado)
            ronda += 1
            pagos = {"player": 1.0, "banker": 0.95, "tie": 8.0}
            if apuesta == resultado:
                dinero += pagos[apuesta]
                aciertos += 1
                mosca.dopamina(apuesta, 0.02)
                estado["mensaje"] = f"GANASTE +{pagos[apuesta]:.2f}"
            else:
                dinero -= 1.0
                mosca.dopamina(apuesta, -0.005)
                estado["mensaje"] = "PERDISTE -1.00"
            estado["apuesta"] = apuesta
            estado["resultado"] = resultado

        pantalla.fill(FONDO)
        pantalla.blit(fuente.render("CEREBRO DE LA MOSCA", True, COLOR_TEXTO), (90, 20))
        dibujar_cerebro(pantalla, mosca, activaciones, pos)
        dibujar_panel(pantalla, fuente, fuente_pequena,
                      estado, dinero, dinero_inicial,
                      aciertos, ronda, historico)
        pantalla.blit(fuente_pequena.render(
            "ESPACIO = pausa    ESC = salir", True, (120, 120, 140)),
            (20, ALTO - 30))

        pygame.display.flip()
        reloj.tick(30)

    pygame.quit()


if __name__ == "__main__":
    main()