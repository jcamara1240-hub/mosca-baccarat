import pandas as pd

print("Cargando anotaciones (puede tardar)...")
df = pd.read_csv("datos_flywire/Supplemental_file1_neuron_annotations.tsv",
                 sep="\t", low_memory=False)

print(f"Total neuronas: {len(df):,}")
print(f"Columnas: {list(df.columns)[:20]}")
print()
print("Neurotransmisores (top_nt):")
print(df["top_nt"].value_counts(dropna=False).head(15))
print()
print("Con coordenadas de soma:")
print(df[["soma_x", "soma_y", "soma_z"]].notna().all(axis=1).sum())