"""
Mosca con cerebro REAL de FlyWire jugando al baccarat.
Versión con tipos int64 consistentes.
"""

import numpy as np
import pickle
import brian2 as b2
from brian2 import ms

b2.BrianLogger.suppress_hierarchy('brian2')
b2.prefs.codegen.target = 'numpy'


class MoscaReal:
    def __init__(self, semilla=42, archivo="cerebro_full.pkl"):
        print("  cargando pkl...", flush=True)
        np.random.seed(semilla)
        with open(archivo, "rb") as f:
            d = pickle.load(f)
        self.d = d

        rng = np.random.default_rng(semilla)
        conex = d["conexiones"].copy()

        # Forzar tipos int64 en todo (clave para que isin funcione)
        conex["pre"]  = conex["pre"].astype(np.int64)
        conex["post"] = conex["post"].astype(np.int64)
        kenyon_ids = np.array(d["kenyon_ids"], dtype=np.int64)
        mbon_ids   = np.array(d["mbon_ids"],   dtype=np.int64)
        pn_ids     = np.array(d["pn_ids"],     dtype=np.int64)
        dan_ids    = np.array(d["dan_ids"],    dtype=np.int64)

        kenyon_set = set(kenyon_ids.tolist())
        mbon_set_all = set(mbon_ids.tolist())
        pn_set_all = set(pn_ids.tolist())

        # 1) Las 3 MBON más conectadas con Kenyon cells
        mask = conex["pre"].isin(kenyon_set) & conex["post"].isin(mbon_set_all)
        conx_kc_mbon = conex[mask]
        print(f"  KC->MBON totales en el dataset: {len(conx_kc_mbon)}", flush=True)
        mbon_rank = conx_kc_mbon.groupby("post")["syn_count"].sum().sort_values(ascending=False)
        self.mbon_sel = [int(x) for x in mbon_rank.head(3).index]
        print(f"  MBON elegidas: {self.mbon_sel}", flush=True)

        # 2) Las 400 Kenyon más conectadas con esas MBON
        mbon_set_sel = set(self.mbon_sel)
        mask = conex["pre"].isin(kenyon_set) & conex["post"].isin(mbon_set_sel)
        conx_filtrada = conex[mask]
        print(f"  KC->(3 MBON) encontradas: {len(conx_filtrada)}", flush=True)
        kc_rank = conx_filtrada.groupby("pre")["syn_count"].sum().sort_values(ascending=False)
        self.kc_sel = [int(x) for x in kc_rank.head(400).index]
        print(f"  KC elegidas: {len(self.kc_sel)}", flush=True)

        # 3) Las 11 PN más conectadas con esas KC
        kc_set_sel = set(self.kc_sel)
        mask = conex["pre"].isin(pn_set_all) & conex["post"].isin(kc_set_sel)
        conx_pn_kc = conex[mask]
        print(f"  PN->(KC seleccionadas) encontradas: {len(conx_pn_kc)}", flush=True)
        pn_rank = conx_pn_kc.groupby("pre")["syn_count"].sum().sort_values(ascending=False)
        self.pn_sel = [int(x) for x in pn_rank.head(11).index]
        print(f"  PN elegidas: {len(self.pn_sel)}", flush=True)

        # 4) DAN: una al azar
        self.dan_sel = [int(x) for x in rng.choice(dan_ids, size=1, replace=False)]

        self.apuesta_mbon = {
            "player": self.mbon_sel[0],
            "banker": self.mbon_sel[1],
            "tie":    self.mbon_sel[2],
        }

        # Guardamos la conex con tipos buenos para la red
        self._conex = conex
        self._construir_red()

    def _construir_red(self):
        print("  entrando a _construir_red...", flush=True)
        b2.start_scope()
        eqs = 'dv/dt = (I - v) / (10*ms) : 1\nI : 1'

        self.pn   = b2.NeuronGroup(len(self.pn_sel),   eqs, threshold='v>1', reset='v=0', method='euler')
        self.kc   = b2.NeuronGroup(len(self.kc_sel),   eqs, threshold='v>1', reset='v=0', method='euler')
        self.mbon = b2.NeuronGroup(len(self.mbon_sel), eqs, threshold='v>1', reset='v=0', method='euler')
        print("  grupos creados", flush=True)

        idx_pn   = {r: i for i, r in enumerate(self.pn_sel)}
        idx_kc   = {r: i for i, r in enumerate(self.kc_sel)}
        idx_mbon = {r: i for i, r in enumerate(self.mbon_sel)}

        pn_set   = set(self.pn_sel)
        kc_set   = set(self.kc_sel)
        mbon_set = set(self.mbon_sel)

        conex = self._conex

        # PN -> KC
        pn_kc = conex[conex["pre"].isin(pn_set) & conex["post"].isin(kc_set)]
        print(f"  PN->KC reales: {len(pn_kc)}", flush=True)
        self.S_pn_kc = b2.Synapses(self.pn, self.kc, 'w : 1', on_pre='v_post += w')
        if len(pn_kc) == 0:
            self.S_pn_kc.connect(p=0.3)
            self.S_pn_kc.w = 'rand() * 0.15'
        else:
            pre_i  = [idx_pn[int(r)]  for r in pn_kc["pre"]]
            post_i = [idx_kc[int(r)]  for r in pn_kc["post"]]
            self.S_pn_kc.connect(i=pre_i, j=post_i)
            self.S_pn_kc.w = np.clip(np.log1p(pn_kc["syn_count"].values) * 0.05, 0, 0.5)
        print(f"  PN->KC listas: {len(self.S_pn_kc)}", flush=True)

        # KC -> MBON
        kc_mb = conex[conex["pre"].isin(kc_set) & conex["post"].isin(mbon_set)]
        print(f"  KC->MBON reales: {len(kc_mb)}", flush=True)
        self.S_kc_mbon = b2.Synapses(self.kc, self.mbon, 'w : 1', on_pre='v_post += w')
        if len(kc_mb) == 0:
            self.S_kc_mbon.connect(p=0.3)
            self.S_kc_mbon.w = 'rand() * 0.15'
            self._real_kc_mbon = False
        else:
            pre_i  = [idx_kc[int(r)]   for r in kc_mb["pre"]]
            post_i = [idx_mbon[int(r)] for r in kc_mb["post"]]
            self.S_kc_mbon.connect(i=pre_i, j=post_i)
            self.S_kc_mbon.w = np.clip(np.log1p(kc_mb["syn_count"].values) * 0.05, 0, 0.5)
            self._real_kc_mbon = True
        print(f"  KC->MBON listas: {len(self.S_kc_mbon)}", flush=True)

        self.m_kc   = b2.SpikeMonitor(self.kc)
        self.m_mbon = b2.SpikeMonitor(self.mbon)
        self.net = b2.Network(b2.collect())
        self.ultima_act_kc = np.zeros(len(self.kc_sel))

        self.S_player = self.S_kc_mbon
        self.S_banker = self.S_kc_mbon
        self.S_tie    = self.S_kc_mbon
        print("  red lista", flush=True)

    def decidir(self, sensores):
        self.pn.I = sensores * 2.0
        c0_kc = np.array(self.m_kc.count[:]).copy()
        c0_mb = np.array(self.m_mbon.count[:]).copy()
        self.net.run(30*ms)
        act_kc = np.array(self.m_kc.count[:]) - c0_kc
        act_mb = np.array(self.m_mbon.count[:]) - c0_mb
        self.ultima_act_kc = act_kc.astype(float)
        if act_mb.sum() == 0:
            return np.random.choice(["player", "banker", "tie"])
        orden = ["player", "banker", "tie"]
        return orden[int(np.argmax(act_mb))]

    def dopamina(self, apuesta, cantidad):
        orden = ["player", "banker", "tie"]
        mbon_idx = orden.index(apuesta)
        traza = self.ultima_act_kc
        if traza.max() > 0:
            traza = traza / traza.max()
        pre  = np.array(self.S_kc_mbon.i)
        post = np.array(self.S_kc_mbon.j)
        w = np.array(self.S_kc_mbon.w)
        for k in range(len(w)):
            if post[k] == mbon_idx:
                w[k] += cantidad * traza[pre[k]]
        self.S_kc_mbon.w = np.clip(w, 0, 1)


if __name__ == "__main__":
    print("Construyendo mosca real...", flush=True)
    m = MoscaReal()
    print(f"\nSinapsis PN->KC: {len(m.S_pn_kc)}")
    print(f"Sinapsis KC->MBON: {len(m.S_kc_mbon)}")
    print("\nProbando 3 decisiones:")
    for i in range(3):
        s = np.random.rand(11)
        print(f"  Decisión {i+1}: {m.decidir(s)}")
    print("\nTodo OK.")