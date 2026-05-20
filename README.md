# AFM Preprocessing Pipeline — BiFeO₃

Módulo de preprocesamiento para series de imágenes AFM de ferritas de bismuto (BiFeO₃).
Se inserta entre la conversión `.000 → .npy/.png` (AFM_ToolKit) y el recorte
de ROI con YOLO (AFM_ROI_Toolkit).

**Tesis:** *Estudio de la Evolución de los Dominios Ferroeléctricos al Switching
Usando Deep Learning para Aplicación de Memorias de Estado Sólido*
**Instituto Tecnológico de Querétaro — Maestría en Ciencias en Ingeniería**
**Autor:** Miguel Angel Castro Medina

---

## Contenido del módulo

```
AFM_Preprocessing/
├── afm_preprocess.py      # Script principal — ejecutar este
├── afm_diagnostics.py     # Módulo de diagnóstico visual — debe estar en la misma carpeta
├── requirements.txt       # Dependencias Python
└── README.md              # Este archivo
```

---

## Estructura de archivos de entrada

El script acepta los archivos `.npy` y `.png` en carpetas separadas:

```
npy/
  bifeo_training_21_Canal_1.npy
  bifeo_training_21_Canal_2.npy
  bifeo_training_21_Canal_3.npy
  ...
  bifeo_training_N_Canal_3.npy

png/                              ← opcional, mejora el diagnóstico visual
  bifeo_training_21_Canal_1.png
  bifeo_training_21_Canal_2.png
  ...
```

- El índice N puede empezar en cualquier número y no necesita ser consecutivo
- Los `.npy` son **obligatorios** — contienen los datos físicos en unidades reales
- Los `.png` son **opcionales** — si se proveen, el diagnóstico los usa como referencia visual

---

## Estructura de salida

```
Results/
├── png_procesado/
│   ├── prep/              # PNGs preprocesados — homólogo directo de las imágenes originales
│   │     bifeo_training_N_Canal_1_prep.png
│   │     bifeo_training_N_Canal_2_prep.png
│   │     bifeo_training_N_Canal_3_prep.png
│   ├── mask/              # Máscaras binarias Otsu de dominios ↑/↓ (Canal 3)
│   │     bifeo_training_N_Canal_3_mask.png
│   └── diff/              # Diferencias |f_n − f_{n-1}| de Canal 2
│         bifeo_training_N_Canal_2_diff.png
│
├── npy_procesado/
│   ├── prep/              # NPYs preprocesados
│   │     bifeo_training_N_Canal_1_prep.npy
│   │     bifeo_training_N_Canal_2_prep.npy
│   │     bifeo_training_N_Canal_3_prep.npy
│   ├── mask/              # Máscaras binarias Otsu (Canal 3)
│   │     bifeo_training_N_Canal_3_mask.npy
│   └── diff/              # Diferencias entre frames consecutivos (Canal 2)
│         bifeo_training_N_Canal_2_diff.npy
│
└── diagnostics/           # Generado solo si diagnostics=True
      index.html           # Reporte navegable — abrir en el navegador
      diag_etapas_C2_frame*.png
      diag_plane_C2_frame*.png
      diag_line_C2_frame*.png
      diag_drift_series.png
      diag_clahe_C2_frame*.png
      diag_stats_C1.png
      diag_stats_C2.png
      diag_stats_C3.png
      diag_diff_evolution_C2.png
```

### Descripción de cada tipo de archivo de salida

| Archivo | Descripción | Uso en el proyecto |
|---|---|---|
| `_prep.npy / .png` | Imagen preprocesada — homólogo directo de la original | Input YOLO + modelos predictivos |
| `_mask.npy / .png` | Máscara binaria Otsu: 255 = dominio ↑, 0 = dominio ↓ | Ground truth del modelo de segmentación |
| `_diff.npy / .png` | \|frame_N − frame_{N-1}\| de C2, calculado pre-normalización | Input de dinámica de los modelos predictivos |

> **Nota sobre `_diff`:** se calcula sobre los datos post-mediana y **antes** de la
> normalización z-score para preservar la dinámica física real del switching.
> Valores altos indican zonas donde ocurrió inversión de dominio entre ciclos consecutivos.

---

## Instalación

### Dependencias

El pipeline usa cuatro librerías de terceros. Todo lo demás (`pathlib`, `logging`, `numpy.fft`, etc.) es parte de la biblioteca estándar de Python.

| Paquete | Uso |
|---|---|
| `numpy` | Operaciones matriciales, carga/guardado de `.npy` |
| `scipy` | Filtro gaussiano, filtro mediana |
| `opencv-python` | Umbralización Otsu, lectura/escritura de PNG |
| `matplotlib` | Colormaps para renderizado de PNGs y figuras de diagnóstico |

### Entorno virtual (recomendado)

Un entorno virtual aísla las dependencias de este módulo del resto de la instalación de Python, evitando conflictos con AFM_ToolKit o AFM_ROI_Toolkit.

**Windows (PowerShell):**
```powershell
python -m venv venv_afm
.\venv_afm\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Windows (Command Prompt):**
```cmd
python -m venv venv_afm
venv_afm\Scripts\activate.bat
pip install -r requirements.txt
```

**Linux / macOS:**
```bash
python3 -m venv venv_afm
source venv_afm/bin/activate
pip install -r requirements.txt
```

**Verificar instalación:**
```bash
python -c "import numpy, scipy, cv2, matplotlib; print('OK')"
```

---

## Configuración

El script se configura editando el bloque `CONFIG` al inicio de `afm_preprocess.py`.
No requiere argumentos de terminal.

```python
CONFIG = {
    # ── Rutas ────────────────────────────────────────────────────────────────
    "npy_dir":    "./npy",          # carpeta con .npy originales (requerido)
    "png_dir":    "./png",          # carpeta con .png originales (opcional, None para omitir)
    "output_dir": "./Results",      # carpeta de salida

    # ── Identificación de la serie ───────────────────────────────────────────
    "prefix":   "training",         # bifeo_<prefix>_N_Canal_C
    "channels": [1, 2, 3],          # canales a procesar

    # ── Parámetros de preprocesamiento ───────────────────────────────────────
    "plane_order":        1,        # 1 = plano simple, 2 = paraboloide (bow)
    "drift_ref_channel":  2,        # canal usado para calcular el drift
    "gauss_sigma":        1.2,      # σ filtro gaussiano para C1 y C3
    "median_kernel":      3,        # kernel filtro mediana para C2
    "norm_plow":          2,        # percentil inferior normalización C1/C3
    "norm_phigh":         98,       # percentil superior normalización C1/C3

    # ── CLAHE (desactivado por defecto) ──────────────────────────────────────
    # Desactivado: amplifica conexiones artificiales entre dominios en C2.
    # Para reactivar: descomentar el bloque en run_pipeline() y ajustar clip a 0.5–1.0
    "clahe_clip":  0.5,
    "clahe_tile":  8,

    # ── Opciones de salida ───────────────────────────────────────────────────
    "save_diff":   True,            # guardar |f_n - f_{n-1}| de C2

    # ── Diagnóstico ──────────────────────────────────────────────────────────
    "diagnostics": True,            # generar reporte HTML navegable
    "diag_frame":  None,            # None = frame central automático
    "diag_channel": 2,              # canal analizado en detalle
}
```

### Parámetros clave a ajustar

| Parámetro | Cuándo ajustar |
|---|---|
| `plane_order = 2` | Si la topografía tiene curvatura visible tipo "cuenco" (bow) |
| `gauss_sigma = 0.8` | Si las imágenes tienen granos muy pequeños (alta resolución) |
| `norm_plow = 5, norm_phigh = 95` | Si C1 o C3 siguen saturados visualmente tras el preprocesamiento |
| `drift_ref_channel = 1` | Si C2 tiene demasiado ruido para estimar drift de forma estable |

### Referencia rápida de parámetros

| Parámetro | Default | Rango típico | Descripción |
|---|---|---|---|
| `plane_order` | 1 | 1–2 | Grado del polinomio de corrección de plano |
| `drift_ref_channel` | 2 | 1–3 | Canal de referencia para estimación de drift |
| `gauss_sigma` | 1.2 | 0.8–1.5 | Sigma del filtro gaussiano para C1 y C3 |
| `median_kernel` | 3 | 3–5 | Tamaño del kernel mediana para C2 |
| `norm_plow` | 2 | 1–5 | Percentil inferior de normalización C1/C3 |
| `norm_phigh` | 98 | 95–99 | Percentil superior de normalización C1/C3 |
| `clahe_clip` | 0.5 | 0.5–1.5 | Clip limit CLAHE (solo si se reactiva) |
| `clahe_tile` | 8 | 4–16 | Tamaño de tile CLAHE |

---

## Ejecución

```powershell
# Activar entorno virtual
.\venv_afm\Scripts\Activate.ps1

# Ejecutar
python afm_preprocess.py
```

---

## Pipeline de preprocesamiento — etapas

Todas las etapas operan sobre los datos `.npy` en unidades físicas reales,
no sobre los PNG ya normalizados, para evitar artefactos de cuantización.

| # | Etapa | Canales | Justificación |
|---|---|---|---|
| ① | Corrección de plano (poly fit orden 1 ó 2) | 1, 2, 3 | Elimina tilt e inclinación del sustrato |
| ② | Corrección línea a línea (mediana) | 1, 2, 3 | Corrige offsets entre líneas de barrido del scanner |
| ③ | Corrección de drift (phase correlation) | 1, 2, 3 | Alinea frames de la serie temporal; shifts calculados con canal de referencia y aplicados a todos |
| ④a | Filtro Gaussiano σ=1.2 | 1, 3 | Reduce ruido de alta frecuencia preservando bordes de granos |
| ④b | Filtro mediana 3×3 | 2 | Elimina spikes impulsivos de la señal de amplitud PFM sin difuminar paredes de dominio |
| ⑤ | Diff \|f_N − f_{N-1}\| pre-normalización | 2 | Captura dinámica real de switching antes de transformar la distribución |
| ⑥a | Normalización percentil p2–p98 global | 1, 3 | Más conservadora que min-max; ignora outliers extremos; contraste comparable entre frames |
| ⑥b | Normalización z-score global | 2 | Inputs estables para redes neuronales; preserva distribución relativa entre frames |
| ⑦ | Umbralización Otsu | 3 | Segmentación binaria de dominios ↑/↓ para ground truth del modelo de segmentación |

### Colormaps por canal

| Canal | Colormap | Tipo | Justificación |
|---|---|---|---|
| C1 — Topografía | `afmhot` | Secuencial | La topografía tiene jerarquía natural: alto = claro, bajo = oscuro |
| C2 — Amplitud PFM | `RdBu_r` | Divergente | La amplitud tiene un cero físico real; rojo = alta respuesta, azul = baja, blanco = pared de dominio |
| C3 — Fase PFM | `twilight` | Cíclico | La fase es un dato angular; colormap cíclico evita bordes artificiales entre −180° y +180° |

### Por qué CLAHE está desactivado

El CLAHE fue evaluado sobre imágenes de Canal 2 y se determinó que amplifica conexiones artificiales entre dominios que físicamente están separados. La estructura discreta de los dominios de amplitud PFM se preserva mejor sin él. Si en el futuro se necesita realce local para visualización o para YOLO en imágenes de bajo contraste, se puede reactivar el bloque comentado en `run_pipeline()` usando `clahe_clip` entre 0.5 y 1.0, **después** de la normalización z-score y **solo** para los PNGs de visualización, no para los NPYs del modelo.

---

## Diagnóstico visual

Cuando `diagnostics: True`, genera `Results/diagnostics/index.html` — abrir en el navegador para navegar todas las figuras.

| Figura | Contenido |
|---|---|
| `diag_etapas_C{n}_frame*.png` | Imagen + histograma en cada etapa del pipeline |
| `diag_plane_C{n}_frame*.png` | Crudo / imagen del microscopio / fondo ajustado / corregido / perfil de línea |
| `diag_line_C{n}_frame*.png` | Antes/después corrección línea a línea + mediana por línea |
| `diag_drift_series.png` | Drift acumulado dy/dx vs frame + trayectoria 2D |
| `diag_clahe_C2_frame*.png` | Antes/después CLAHE + histogramas + CDF (solo si CLAHE activo) |
| `diag_stats_C{n}.png` | Evolución de μ, σ, min, max por frame — detecta frames atípicos |
| `diag_diff_evolution_C2.png` | Cambio medio/máximo entre frames consecutivos en C2 — picos = eventos de switching |

> Si `png_dir` está configurado, las figuras del diagnóstico incluyen una columna
> adicional con borde naranja mostrando la imagen original del microscopio como
> referencia visual directa.

---

## Flujo de trabajo recomendado

### Fase de calibración (primera ejecución)

```python
# En CONFIG:
"diagnostics": True,
"diag_frame":  None,    # frame central automático
"save_diff":   False,   # más rápido sin diff
```

Abrir `Results/diagnostics/index.html` y verificar:

- **`diag_plane`** — el fondo ajustado elimina la inclinación correctamente
- **`diag_line`** — la mediana por línea queda plana después de la corrección
- **`diag_drift`** — el drift acumulado es razonable (< 10% del ancho de imagen)
- **`diag_stats`** — no hay caídas bruscas en min/max que indiquen frames atípicos
- **`diag_clahe`** — si se reactivó, verificar que no fusiona dominios artificialmente

### Fase de producción

```python
# En CONFIG:
"diagnostics": False,   # omite diagnóstico para mayor velocidad
"save_diff":   True,    # genera archivos diff para los modelos predictivos
```

---

## Compatibilidad con PNG para YOLO

Los PNGs generados por el script se guardan al **tamaño original del array**
(256×128 px para el setup de este proyecto) usando OpenCV directamente, sin
escalado por matplotlib. Esto garantiza que las coordenadas de detección de YOLO
corresponden directamente a píxeles reales, evitando el desplazamiento de
coordenadas que ocurre cuando matplotlib escala las imágenes con `figsize` y `dpi`.

---

## Integración con el pipeline completo

```
AFM_ToolKit
  .000  ──►  npy/  (datos físicos en unidades reales)
         ──►  png/  (imágenes del microscopio, opcional)
                │
                ▼
      afm_preprocess.py          ← este módulo
                │
      ┌─────────┴──────────┐
      │                    │
  png_procesado/       npy_procesado/
    prep/                prep/       ──► AFM_ROI_Toolkit (YOLO + recortes)
    mask/                mask/       ──► ground truth modelo segmentación
    diff/                diff/       ──► input dinámica modelos predictivos
                │
                ▼
      AFM_ROI_Toolkit
        YOLO entrenado con Canal 1 prep
        Recortes multicanal (C1, C2, C3) en PNG y NPY — 80×80 px
                │
                ▼
      modelo_predictivo/
        Modelo_Segmentacion.py    ──► predice C3_mask[N+1]
        Modelo_Regresion_C2.py    ──► predice C2_prep[N+1]
        Modelo_Regresion_C3.py    ──► predice C3_prep[N+1]
```

---

## Licencia

Proyecto académico — Instituto Tecnológico de Querétaro. Todos los derechos reservados.