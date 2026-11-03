"""
Visor en tiempo real v2.
- Sinapsis que engordan y cambian de color según el peso.
- Gráfica del dinero abajo.
- R = reiniciar | + y - = velocidad | ESPACIO = pausa | ESC = salir
"""

import pygame
import numpy as np
import random
import time
from baccarat import jugar_mano
from mosca import MoscaBaccarat

ANCHO, ALTO = 1300, 800
FONDO = (18, 18, 24)
COLOR_TEXTO = (230, 230, 230)
COLOR_SENSOR = (120, 180, 255)
COLOR_PLAYER = (255, 110, 110)
COLOR_BANKER = (120, 230, 140)
COLOR_TIE    = (255, 230, 110)


def layout():
    pos = {}
    for i in range(5):
        pos[("sensor", i)] = (120, 150 + i * 70)
    for i in range(10):
        pos[("player", i)] = (320, 90 + i * 45)
        pos[("banker", i)] = (470, 90 + i * 45)
        pos[("tie",    i)] = (620, 90 + i * 45)
    return pos


def color_segun_peso(peso):
    """De gris claro (peso 0) a amarillo brillante (peso 1)."""
    peso = float(np.clip(peso, 0, 1))
    r = int(80 + peso * 175)
    g = int(80 + peso * 175)
    b = int(100 + peso * 50)
    return (r, g, b)


def dibujar_cerebro(pantalla, mosca, activaciones, pos, tecla_resaltar):
    sinapsis = [(mosca.S_player, "player"),
                (mosca.S_banker, "banker"),
                (mosca.S_tie,    "tie")]

    for S, nombre in sinapsis:
        pre, post = np.array(S.i), np.array(S.j)
        ws = np.array(S.w)
        for a, b, w in zip(pre, post, ws):
            x1, y1 = pos[("sensor", int(a))]
            x2, y2 = pos[(nombre, int(b))]
            grosor = max(1, int(1 + w * 6))
            color = color_segun_peso(w)
            pygame.draw.line(pantalla, color, (x1, y1), (x2, y2), grosor)

    for (grupo, idx), (x, y) in pos.items():
        base = {"sensor": COLOR_SENSOR, "player": COLOR_PLAYER,
                "banker": COLOR_BANKER, "tie": COLOR_TIE}[grupo]
        act = activaciones.get((grupo, idx), 0.0)
        r = min(255, int(base[0] + act * (255 - base[0])))
        g = min(255, int(base[1] + act * (255 - base[1])))
        b = min(255, int(base[2] + act * (255 - base[2])))
        radio = 12 if grupo != "sensor" else 15
        pygame.draw.circle(pantalla, (r, g, b), (x, y), radio)
        pygame.draw.circle(pantalla, (20, 20, 30), (x, y), radio, 2)


def dibujar_grafica(pantalla, curva, x0, y0, w, h, dinero_ini):
    pygame.draw.rect(pantalla, (30, 30, 40), (x0, y0, w, h), border_radius=6)
    pygame.draw.rect(pantalla, (80, 80, 100), (x0, y0, w, h), 1, border_radius=6)
    if len(curva) < 2:
        return
    minv = min(min(curva), dinero_ini - 50)
    maxv = max(max(curva), dinero_ini + 50)
    rng = max(maxv - minv, 1)
    # Línea del dinero inicial
    y_ini = y0 + h - int((dinero_ini - minv) / rng * h)
    pygame.draw.line(pantalla, (90, 90, 120), (x0, y_ini), (x0 + w, y_ini), 1)
    puntos = []
    for i, v in enumerate(curva):
        x = x0 + int(i / max(len(curva) - 1, 1) * w)
        y = y0 + h - int((v - minv) / rng * h)
        puntos.append((x, y))
    if len(puntos) > 1:
        color = (120, 230, 140) if curva[-1] >= dinero_ini else (230, 110, 110)
        pygame.draw.lines(pantalla, color, False, puntos, 2)


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
    pygame.display.set_caption("Mosca jugando al baccarat v2")
    fuente = pygame.font.SysFont("arial", 24, bold=True)
    fp = pygame.font.SysFont("arial", 18)
    reloj = pygame.time.Clock()

    def nueva_partida():
        return (MoscaBaccarat(semilla=random.randint(0, 10000)), [],
                1000.0, 0, 0, {"apuesta": "-", "resultado": "-", "mensaje": ""})

    mosca, historico, dinero, aciertos, ronda, estado = nueva_partida()
    dinero_ini = 1000.0
    pos = layout()
    activaciones = {}
    curva = [dinero]
    ultimo = time.time()
    pausa = False
    velocidad = 1.2  # segundos entre manos

    corriendo = True
    while corriendo:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                corriendo = False
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_SPACE: pausa = not pausa
                if ev.key == pygame.K_ESCAPE: corriendo = False
                if ev.key == pygame.K_r:
                    mosca, historico, dinero, aciertos, ronda, estado = nueva_partida()
                    curva = [dinero]
                if ev.key in (pygame.K_PLUS, pygame.K_KP_PLUS, pygame.K_EQUALS):
                    velocidad = max(0.05, velocidad * 0.6)
                if ev.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    velocidad = min(3.0, velocidad * 1.6)

        if not pausa and time.time() - ultimo > velocidad:
            ultimo = time.time()
            sensores = estado_a_sensores(historico)
            apuesta = mosca.decidir(sensores)
            activaciones = {("sensor", i): float(sensores[i]) for i in range(5)}
            for i in range(10):
                for g in ("player", "banker", "tie"):
                    activaciones[(g, i)] = 1.0 if apuesta == g else 0.1
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
            curva.append(dinero)
            if len(curva) > 600:
                curva = curva[-600:]

        pantalla.fill(FONDO)
        pantalla.blit(fuente.render("CEREBRO DE LA MOSCA", True, COLOR_TEXTO), (90, 30))
        dibujar_cerebro(pantalla, mosca, activaciones, pos, None)

        # Panel derecha
        x0 = 760; y = 40
        pantalla.blit(fuente.render("MESA DE BACCARAT", True, COLOR_TEXTO), (x0, y)); y += 50
        pantalla.blit(fp.render(f"Ronda: {ronda}", True, COLOR_TEXTO), (x0, y)); y += 25
        pantalla.blit(fp.render(f"Dinero: {dinero:.1f} (inicio {dinero_ini:.0f})", True, COLOR_TEXTO), (x0, y)); y += 25
        tasa = (aciertos / ronda * 100) if ronda > 0 else 0
        pantalla.blit(fp.render(f"Aciertos: {aciertos} ({tasa:.1f}%)", True, COLOR_TEXTO), (x0, y)); y += 25
        pantalla.blit(fp.render(f"Velocidad: {velocidad:.2f}s/mano   Pausa: {'SI' if pausa else 'no'}", True, COLOR_TEXTO), (x0, y)); y += 40

        pantalla.blit(fp.render("Última apuesta:", True, COLOR_TEXTO), (x0, y)); y += 25
        color_ap = {"player": COLOR_PLAYER, "banker": COLOR_BANKER, "tie": COLOR_TIE}.get(estado["apuesta"], COLOR_TEXTO)
        pantalla.blit(fuente.render(estado["apuesta"].upper(), True, color_ap), (x0, y)); y += 40
        pantalla.blit(fp.render("Resultado:", True, COLOR_TEXTO), (x0, y)); y += 25
        color_res = {"player": COLOR_PLAYER, "banker": COLOR_BANKER, "tie": COLOR_TIE}.get(estado["resultado"], COLOR_TEXTO)
        pantalla.blit(fuente.render(estado["resultado"].upper(), True, color_res), (x0, y)); y += 40
        msg = estado["mensaje"]
        if msg:
            color_msg = (120, 230, 140) if "GAN" in msg else (230, 110, 110)
            pantalla.blit(fuente.render(msg, True, color_msg), (x0, y)); y += 40

        # Histórico últimos 25
        pantalla.blit(fp.render("Histórico:", True, COLOR_TEXTO), (x0, y)); y += 25
        xx = x0
        for r in historico[-25:]:
            c = {"player": COLOR_PLAYER, "banker": COLOR_BANKER, "tie": COLOR_TIE}[r]
            pygame.draw.circle(pantalla, c, (xx + 10, y + 10), 8)
            xx += 20

        # Gráfica
        pantalla.blit(fp.render("Dinero en el tiempo:", True, COLOR_TEXTO), (90, 620))
        dibujar_grafica(pantalla, curva, 90, 650, 1120, 110, dinero_ini)

        pantalla.blit(fp.render(
            "ESPACIO=pausa  R=reiniciar  +/-=velocidad  ESC=salir",
            True, (140, 140, 160)), (20, ALTO - 25))

        pygame.display.flip()
        reloj.tick(30)

    pygame.quit()


if __name__ == "__main__":
    main()