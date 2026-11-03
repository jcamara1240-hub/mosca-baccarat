import pandas as pd

clasif = pd.read_csv("datos_flywire/classification.csv.gz", low_memory=False)

print("Columnas:", list(clasif.columns))
print()

# Super classes
print("=== SUPER_CLASS (valores únicos) ===")
print(clasif["super_class"].value_counts())
print()

# Class
print("=== CLASS más frecuentes ===")
print(clasif["class"].value_counts().head(30))
print()

# Buscar MBONs
print("=== Filas con 'MBON' en alguna columna ===")
for col in ["class", "sub_class", "cell_type"]:
    if col in clasif.columns:
        hits = clasif[clasif[col].astype(str).str.contains("MBON", case=False, na=False)]
        if len(hits) > 0:
            print(f"  en '{col}': {len(hits)} filas")
            print(hits[col].value_counts().head(5))
print()

# Buscar DANs / dopaminergic
print("=== Filas con 'DAN' o 'dopamin' ===")
for col in ["class", "sub_class", "cell_type"]:
    if col in clasif.columns:
        hits = clasif[clasif[col].astype(str).str.contains("DAN|dopamin", case=False, na=False, regex=True)]
        if len(hits) > 0:
            print(f"  en '{col}': {len(hits)} filas")
            print(hits[col].value_counts().head(5))
print()

# Buscar PNs
print("=== Filas con 'projection' ===")
for col in ["class", "sub_class", "cell_type"]:
    if col in clasif.columns:
        hits = clasif[clasif[col].astype(str).str.contains("projection", case=False, na=False)]
        if len(hits) > 0:
            print(f"  en '{col}': {len(hits)} filas")
            print(hits[col].value_counts().head(5))

# Y por si acaso, consolidated_cell_types
print("\n=== CONSOLIDATED_CELL_TYPES tipos con MBON/DAN/PN ===")
ct = pd.read_csv("datos_flywire/consolidated_cell_types.csv.gz", low_memory=False)
for key in ["MBON", "DAN", "PAM", "PPL"]:
    hits = ct[ct["primary_type"].astype(str).str.contains(key, case=False, na=False)]
    if len(hits) > 0:
        print(f"  '{key}': {len(hits)} filas, primeros tipos:")
        print(hits["primary_type"].value_counts().head(5))