# Datos de FlyWire

Esta carpeta está vacía en el repositorio. Los scripts buscan aquí los datos del conectoma, que hay que descargar a mano (es gratis).

## Qué descargar

1. Entra en [codex.flywire.ai](https://codex.flywire.ai) e inicia sesión con una cuenta de Google.
2. En el menú, abre **Download Data**, marca la casilla de condiciones de uso y descarga, para la versión **FAFB v783**:
   - `Classification / Hierarchical Annotations` → `classification.csv.gz`
   - `Cell Types` → `consolidated_cell_types.csv.gz`
   - `Marked Neuron Coordinates` → `coordinates.csv.gz`
   - `Connections (Filtered)` → `connections_princeton.csv.gz` (unos 68 MB)

   Con estos cuatro funcionan las carpetas `01` y `02`.

3. Para las carpetas `03` y `04` hacen falta además los archivos del modelo de Shiu et al. 2024, que acompañan a ese artículo:
   - `Supplemental_file1_neuron_annotations.tsv`
   - `Connectivity_783.parquet` (unos 100 MB)
   - `Completeness_783.csv`

Al final la carpeta debería quedar así:

```
datos_flywire/
  classification.csv.gz
  connections_princeton.csv.gz
  consolidated_cell_types.csv.gz
  coordinates.csv.gz
  Supplemental_file1_neuron_annotations.tsv
  Connectivity_783.parquet
  Completeness_783.csv
```

## Una nota sobre las coordenadas

FlyWire guarda las posiciones X e Y en píxeles de 4 nm y la Z en cortes de 40 nm, porque así se escaneó el cerebro. Si las pintas tal cual, el cerebro sale plano como un folio (me pasó). Los scripts ya hacen la conversión a micras.

## Citar

Si usas estos datos, cita a Dorkenwald et al. 2024 y Schlegel et al. 2024 (FlyWire) y, para el modelo, a Shiu et al. 2024.
