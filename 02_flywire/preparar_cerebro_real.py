"""
Prepara el cerebro COMPLETO para la simulación.
Guarda cerebro_full.pkl con:
  - kenyon_ids, mbon_ids, dan_ids, pn_ids:  listas de root_id de cada tipo
  - coords:        {root_id: (x, y, z)}
  - subtipos:      {root_id: "KCg-m"/"MBON10"/"PAM01"/...}
  - clases:        {root_id: "Kenyon_Cell"/"MBON"/"DAN"/"ALPN"}
  - conexiones:    DataFrame pre, post, syn_count (entre cualquier pareja de estos tipos)
"""

import pandas as pd
import numpy as np
import pickle
import os
import re
from collections import Counter

DATOS = "datos_flywire"

print("Cargando clasificaciones...")
clasif = pd.read_csv(f"{DATOS}/classification.csv.gz", low_memory=False)

CLASES_INTERES = ["Kenyon_Cell", "MBON", "DAN", "ALPN"]
sub = clasif[clasif["class"].isin(CLASES_INTERES)].copy()
print(f"  Neuronas de interés: {len(sub)}")
for cls, n in sub["class"].value_counts().items():
    print(f"    {cls}: {n}")

ids_todos = set(sub["root_id"].astype(np.int64).tolist())
clases = dict(zip(sub["root_id"].astype(np.int64), sub["class"].astype(str)))

# -------- Subtipos --------
print("\nCargando subtipos finos...")
ct = pd.read_csv(f"{DATOS}/consolidated_cell_types.csv.gz", low_memory=False)
ct_sel = ct[ct["root_id"].astype(np.int64).isin(ids_todos)]
subtipos = dict(zip(ct_sel["root_id"].astype(np.int64),
                    ct_sel["primary_type"].astype(str)))
print(f"  Con subtipo: {len(subtipos)}")

# -------- Coordenadas --------
print("\nCargando coordenadas...")
coords_df = pd.read_csv(f"{DATOS}/coordinates.csv.gz", low_memory=False)
coords_df = coords_df[coords_df["root_id"].astype(np.int64).isin(ids_todos)]

def parse_pos(s):
    nums = re.findall(r"-?\d+", str(s))
    if len(nums) < 3:
        return None
    return (float(nums[0]), float(nums[1]), float(nums[2]))

coords = {}
for _, row in coords_df.iterrows():
    p = parse_pos(row["position"])
    if p is not None:
        rid = int(row["root_id"])
        if rid not in coords:
            coords[rid] = p
print(f"  Con coordenadas: {len(coords)}")

# -------- Conexiones --------
print("\nCargando conexiones (puede tardar 1-2 min)...")
conex = pd.read_csv(
    f"{DATOS}/connections_princeton.csv.gz",
    low_memory=False,
    dtype={"pre_root_id": np.int64, "post_root_id": np.int64},
)
print(f"  Total: {len(conex):,}")

mask = (conex["pre_root_id"].isin(ids_todos) &
        conex["post_root_id"].isin(ids_todos))
conex_sel = conex.loc[mask, ["pre_root_id", "post_root_id", "syn_count"]].rename(
    columns={"pre_root_id": "pre", "post_root_id": "post"}
).reset_index(drop=True)
print(f"  Entre neuronas de interés: {len(conex_sel):,}")

# Desglose por par de tipos
print("\n  Conexiones por par (pre -> post):")
pares = []
for _, row in conex_sel.sample(min(len(conex_sel), 50000)).iterrows():
    cpre = clases.get(int(row["pre"]), "?")
    cpost = clases.get(int(row["post"]), "?")
    pares.append(f"{cpre}->{cpost}")
for par, n in Counter(pares).most_common():
    print(f"    {par}: {n} (muestra)")

# -------- Guardar --------
datos = {
    "kenyon_ids": sorted([r for r, c in clases.items() if c == "Kenyon_Cell"]),
    "mbon_ids":   sorted([r for r, c in clases.items() if c == "MBON"]),
    "dan_ids":    sorted([r for r, c in clases.items() if c == "DAN"]),
    "pn_ids":     sorted([r for r, c in clases.items() if c == "ALPN"]),
    "coords":     coords,
    "subtipos":   subtipos,
    "clases":     clases,
    "conexiones": conex_sel,
}

with open("cerebro_full.pkl", "wb") as f:
    pickle.dump(datos, f)

tam = os.path.getsize("cerebro_full.pkl") / 1024 / 1024
print(f"\nGuardado cerebro_full.pkl ({tam:.1f} MB)")
print(f"  Kenyon: {len(datos['kenyon_ids'])}")
print(f"  MBON:   {len(datos['mbon_ids'])}")
print(f"  DAN:    {len(datos['dan_ids'])}")
print(f"  PN:     {len(datos['pn_ids'])}")
print(f"  Conexiones: {len(datos['conexiones']):,}")