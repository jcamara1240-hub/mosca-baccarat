"""
Entrena a la mosca a jugar al baccarat.
Cada mano:
  1. La mosca mira el histórico y decide una apuesta.
  2. Se juega la mano.
  3. Si acierta, recibe dopamina (refuerzo).
  4. Si falla, recibe lo contrario (castigo pequeño).
"""

import numpy as np
import random
from baccarat import jugar_mano
from mosca import MoscaBaccarat


def estado_a_sensores(historico):
    """Convierte las últimas 10 manos en 5 señales entre 0 y 1."""
    ultimos = historico[-10:] if historico else []
    n = max(len(ultimos), 1)
    return np.array([
        ultimos.count("player") / n,
        ultimos.count("banker") / n,
        ultimos.count("tie")    / n,
        min(len(historico) / 500, 1.0),  # cuántas manos ha visto
        random.random()                  # un poco de ruido
    ])


def entrenar(n_manos=500, verboso=True):
    mosca = MoscaBaccarat()
    historico = []
    dinero = 1000.0
    dinero_inicial = dinero
    aciertos = 0

    # Para guardar el dinero cada 50 manos
    curva = []

    for ronda in range(n_manos):
        # 1. Decidir
        sensores = estado_a_sensores(historico)
        apuesta = mosca.decidir(sensores)

        # 2. Jugar
        resultado = jugar_mano()
        historico.append(resultado)

        # 3. Pagos reales del baccarat
        pagos = {"player": 1.0, "banker": 0.95, "tie": 8.0}

        if apuesta == resultado:
            dinero += pagos[apuesta]
            dopamina = 0.02   # refuerzo
            aciertos += 1
        else:
            dinero -= 1.0
            dopamina = -0.005  # castigo pequeño

        # 4. Aprendizaje: la "dopamina" refuerza la sinapsis que decidió
        mosca.dopamina(apuesta, dopamina)

        # Log cada 50 manos
        if (ronda + 1) % 50 == 0:
            tasa = aciertos / (ronda + 1) * 100
            if verboso:
                print(f"Ronda {ronda+1:4d} | dinero: {dinero:7.1f} "
                      f"| aciertos: {tasa:5.1f}% | última apuesta: {apuesta} "
                      f"| salió: {resultado}")
            curva.append((ronda + 1, dinero))

    print("\n" + "=" * 50)
    print(f"Resultado final tras {n_manos} manos:")
    print(f"  Dinero: {dinero:.1f} (empezó con {dinero_inicial:.1f})")
    print(f"  Aciertos: {aciertos}/{n_manos} = {aciertos/n_manos*100:.1f}%")
    print(f"  Veces que apostó cada opción:")
    print(f"    Player: {historico.count('player')} salieron")
    print(f"    Banker: {historico.count('banker')} salieron")
    print(f"    Tie:    {historico.count('tie')} salieron")
    print("=" * 50)

    return mosca, curva


if __name__ == "__main__":
    entrenar(n_manos=500)