"""
Entrenamiento con el cerebro v2.
Ahora los sensores son 11:
  - 3 frecuencias (player, banker, tie en las últimas 10 manos)
  - 8 valores que codifican las 8 últimas manos una por una
"""

import numpy as np
import random
from baccarat import jugar_mano
from mosca_v2 import MoscaBaccarat


def estado_a_sensores(historico):
    ultimos = historico[-10:] if historico else []
    n = max(len(ultimos), 1)

    # Primeros 3: frecuencias
    sensores = [
        ultimos.count("player") / n,
        ultimos.count("banker") / n,
        ultimos.count("tie")    / n,
    ]

    # Siguientes 8: las 8 últimas manos codificadas
    # player = 0.2, banker = 0.7, tie = 1.0, vacío = 0.0
    codificacion = {"player": 0.2, "banker": 0.7, "tie": 1.0}
    ultimas8 = historico[-8:] if historico else []
    for h in ultimas8:
        sensores.append(codificacion[h])
    # Rellenar con ceros si hay menos de 8
    while len(sensores) < 11:
        sensores.append(0.0)

    return np.array(sensores)


def entrenar(n_manos=500, verboso=True):
    mosca = MoscaBaccarat()
    historico = []
    dinero = 1000.0
    dinero_inicial = dinero
    aciertos = 0

    for ronda in range(n_manos):
        sensores = estado_a_sensores(historico)
        apuesta = mosca.decidir(sensores)
        resultado = jugar_mano()
        historico.append(resultado)

        pagos = {"player": 1.0, "banker": 0.95, "tie": 8.0}
        if apuesta == resultado:
            dinero += pagos[apuesta]
            dopamina = 0.03
            aciertos += 1
        else:
            dinero -= 1.0
            dopamina = -0.008

        mosca.dopamina(apuesta, dopamina)

        if (ronda + 1) % 50 == 0 and verboso:
            tasa = aciertos / (ronda + 1) * 100
            print(f"Ronda {ronda+1:4d} | dinero: {dinero:7.1f} "
                  f"| aciertos: {tasa:5.1f}% | apostó: {apuesta} | salió: {resultado}")

    print("\n" + "=" * 50)
    print(f"Resultado tras {n_manos} manos:")
    print(f"  Dinero: {dinero:.1f} (empezó con {dinero_inicial:.1f})")
    print(f"  Aciertos: {aciertos}/{n_manos} = {aciertos/n_manos*100:.1f}%")
    print(f"  Salieron: Player={historico.count('player')}, "
          f"Banker={historico.count('banker')}, Tie={historico.count('tie')}")
    print("=" * 50)

    return mosca, dinero


if __name__ == "__main__":
    entrenar(500)