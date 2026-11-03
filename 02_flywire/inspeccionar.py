import pandas as pd

clasif = pd.read_csv("datos_flywire/classification.csv.gz", low_memory=False)
print("COLUMNAS:", list(clasif.columns))
print("\nValores únicos de super_class:")
print(clasif["super_class"].value_counts().head(20))
print("\nValores de class que contienen 'enyon':")
print(clasif[clasif["class"].astype(str).str.contains("enyon", case=False, na=False)]["class"].value_counts())
print("\nEjemplo primera fila:")
print(clasif.head(3))

print("\n--- COORDENADAS ---")
coords = pd.read_csv("datos_flywire/coordinates.csv.gz", low_memory=False, nrows=3)
print(coords)

print("\n--- CELL TYPES ---")
ct = pd.read_csv("datos_flywire/consolidated_cell_types.csv.gz", low_memory=False, nrows=3)
print(ct.columns.tolist())
print(ct.head(3))