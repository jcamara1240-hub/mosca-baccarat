from brian2 import *
import numpy as np

# Esto hace Brian2 más silencioso en consola
BrianLogger.suppress_hierarchy('brian2')
prefs.codegen.target = 'numpy'  # evita necesitar compilador en Windows


class MoscaBaccarat:
    """
    Cerebrito mínimo con dopamina.
    - 5 neuronas de "sensor" leen el estado del juego.
    - 3 grupos de 10 neuronas deciden: Player, Banker o Tie.
    - Las sinapsis se refuerzan cuando acierta (dopamina positiva)
      y se debilitan cuando falla (dopamina negativa).
    """

    def __init__(self, n_sensor=5, n_decision=10, semilla=42):
        np.random.seed(semilla)
        start_scope()

        # Modelo de neurona: se carga con la corriente I y dispara al pasar 1
        eqs = '''
        dv/dt = (I - v) / (10*ms) : 1
        I : 1
        '''

        self.sensor = NeuronGroup(n_sensor, eqs, threshold='v>1',
                                  reset='v=0', method='euler')
        self.player = NeuronGroup(n_decision, eqs, threshold='v>1',
                                  reset='v=0', method='euler')
        self.banker = NeuronGroup(n_decision, eqs, threshold='v>1',
                                  reset='v=0', method='euler')
        self.tie    = NeuronGroup(n_decision, eqs, threshold='v>1',
                                  reset='v=0', method='euler')

        # Sinapsis con peso aprendible
        self.S_player = self._conectar(self.sensor, self.player)
        self.S_banker = self._conectar(self.sensor, self.banker)
        self.S_tie    = self._conectar(self.sensor, self.tie)

        # Contadores de disparos
        self.mP = SpikeMonitor(self.player)
        self.mB = SpikeMonitor(self.banker)
        self.mT = SpikeMonitor(self.tie)

        self.net = Network(collect())

    def _conectar(self, pre, post):
        S = Synapses(pre, post, 'w : 1', on_pre='v_post += w')
        S.connect(p=0.6)
        S.w = 'rand() * 0.3'
        return S

    def decidir(self, sensores):
        """
        Pasa 'sensores' (array de 5 valores entre 0 y 1) por la red
        y devuelve 'player', 'banker' o 'tie' según quién dispare más.
        """
        self.sensor.I = sensores * 2.0

        # Guardamos conteos previos
        c0_p = int(sum(self.mP.count[:]))
        c0_b = int(sum(self.mB.count[:]))
        c0_t = int(sum(self.mT.count[:]))

        self.net.run(20*ms)

        disparos_p = int(sum(self.mP.count[:])) - c0_p
        disparos_b = int(sum(self.mB.count[:])) - c0_b
        disparos_t = int(sum(self.mT.count[:])) - c0_t

        disparos = {'player': disparos_p, 'banker': disparos_b, 'tie': disparos_t}

        # Si ninguna dispara, elegimos al azar
        if max(disparos.values()) == 0:
            return np.random.choice(['player', 'banker', 'tie'])

        return max(disparos, key=disparos.get)

    def dopamina(self, apuesta, cantidad):
        """
        Refuerza o debilita la sinapsis que llevó a 'apuesta'.
        cantidad > 0: ganó, refuerza.
        cantidad < 0: perdió, debilita.
        """
        S = {'player': self.S_player,
             'banker': self.S_banker,
             'tie':    self.S_tie}[apuesta]

        pesos = np.array(S.w)
        pesos = np.clip(pesos + cantidad, 0, 1)
        S.w = pesos


# Prueba rápida: solo comprobar que el cerebro decide algo
if __name__ == "__main__":
    m = MoscaBaccarat()
    for i in range(5):
        sensores = np.random.rand(5)
        decision = m.decidir(sensores)
        print(f"Prueba {i+1}: {decision}")