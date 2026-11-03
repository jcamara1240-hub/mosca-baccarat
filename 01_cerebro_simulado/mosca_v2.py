"""
Cerebro v2: más listo que el anterior.
- Capa intermedia de 20 neuronas (como una red profunda sencilla).
- Memoria de 8 manos recientes codificadas una a una.
- Dopamina que refuerza las dos capas de sinapsis con traza de elegibilidad:
  solo se refuerzan las que llevaron a esa decisión concreta.

Mantiene la MISMA interfaz que mosca.py (decidir, dopamina, S_player, S_banker, S_tie)
para que visor.py, visor2.py, visor3d.py y jugar.py funcionen sin cambios.
"""

from brian2 import *
import numpy as np

BrianLogger.suppress_hierarchy('brian2')
prefs.codegen.target = 'numpy'


class MoscaBaccarat:
    def __init__(self, n_sensor=11, n_intermedia=20, n_decision=10, semilla=42):
        """
        n_sensor = 3 (frecuencias) + 8 (últimas manos codificadas) = 11
        """
        np.random.seed(semilla)
        start_scope()

        eqs = '''
        dv/dt = (I - v) / (10*ms) : 1
        I : 1
        '''

        self.sensor      = NeuronGroup(n_sensor, eqs, threshold='v>1',
                                       reset='v=0', method='euler')
        self.intermedia  = NeuronGroup(n_intermedia, eqs, threshold='v>1',
                                       reset='v=0', method='euler')
        self.player      = NeuronGroup(n_decision, eqs, threshold='v>1',
                                       reset='v=0', method='euler')
        self.banker      = NeuronGroup(n_decision, eqs, threshold='v>1',
                                       reset='v=0', method='euler')
        self.tie         = NeuronGroup(n_decision, eqs, threshold='v>1',
                                       reset='v=0', method='euler')

        # Capa 1: sensores -> intermedia
        self.S_in = Synapses(self.sensor, self.intermedia,
                             'w : 1', on_pre='v_post += w')
        self.S_in.connect(p=0.6)
        self.S_in.w = 'rand() * 0.25'

        # Capa 2: intermedia -> cada decisión
        self.S_player = self._conectar(self.intermedia, self.player)
        self.S_banker = self._conectar(self.intermedia, self.banker)
        self.S_tie    = self._conectar(self.intermedia, self.tie)

        # Monitores de disparo
        self.m_inter = SpikeMonitor(self.intermedia)
        self.mP = SpikeMonitor(self.player)
        self.mB = SpikeMonitor(self.banker)
        self.mT = SpikeMonitor(self.tie)

        self.net = Network(collect())

        # Traza de elegibilidad: qué neuronas intermedias dispararon en la última decisión
        self.ultima_actividad_intermedia = np.zeros(n_intermedia)

    def _conectar(self, pre, post):
        S = Synapses(pre, post, 'w : 1', on_pre='v_post += w')
        S.connect(p=0.6)
        S.w = 'rand() * 0.25'
        return S

    def decidir(self, sensores):
        self.sensor.I = sensores * 2.0

        # Guardar contadores previos
        c0_i = int(sum(self.m_inter.count[:]))
        c0_p = int(sum(self.mP.count[:]))
        c0_b = int(sum(self.mB.count[:]))
        c0_t = int(sum(self.mT.count[:]))
        act_inter_antes = np.array(self.m_inter.count[:]).copy()

        self.net.run(25*ms)

        # Guardar qué intermedias dispararon (para la traza)
        act_inter_ahora = np.array(self.m_inter.count[:])
        self.ultima_actividad_intermedia = (
            act_inter_ahora - act_inter_antes
        ).astype(float)

        disparos = {
            'player': int(sum(self.mP.count[:])) - c0_p,
            'banker': int(sum(self.mB.count[:])) - c0_b,
            'tie':    int(sum(self.mT.count[:])) - c0_t,
        }

        if max(disparos.values()) == 0:
            return np.random.choice(['player', 'banker', 'tie'])

        return max(disparos, key=disparos.get)

    def dopamina(self, apuesta, cantidad):
        # Capa 2: la sinapsis que lleva a la apuesta elegida
        S2 = {'player': self.S_player,
              'banker': self.S_banker,
              'tie':    self.S_tie}[apuesta]

        # Refuerzo proporcional a cuánto disparó cada neurona intermedia
        pre_idx = np.array(S2.i)       # índice de la neurona pre (intermedia)
        traza = self.ultima_actividad_intermedia
        traza_norm = traza / max(traza.max(), 1e-6)

        pesos2 = np.array(S2.w)
        for k, pre in enumerate(pre_idx):
            pesos2[k] += cantidad * traza_norm[int(pre)]
        S2.w = np.clip(pesos2, 0, 1)

        # Capa 1: también refuerza (menos) las sinapsis que llevaron a esas intermedias
        pesos1 = np.array(self.S_in.w)
        post_idx = np.array(self.S_in.j)   # intermedia destino
        for k, post in enumerate(post_idx):
            pesos1[k] += 0.3 * cantidad * traza_norm[int(post)]
        self.S_in.w = np.clip(pesos1, 0, 1)


# Prueba rápida
if __name__ == "__main__":
    m = MoscaBaccarat()
    for i in range(5):
        s = np.random.rand(11)
        print(f"Prueba {i+1}: {m.decidir(s)}")