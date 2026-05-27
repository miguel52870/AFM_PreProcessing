"""
AFM Preprocessing Pipeline — BiFeO3 Ferroelectric Domains
==========================================================
Tesis: "Estudio de la Evolución de los Dominios Ferroeléctricos al Switching
        Usando Deep Learning para Aplicación de Memorias de Estado Sólido"

Compatibilidad con flujos existentes:
  - AFM_ToolKit    : https://github.com/miguel52870/AFM_ToolKit
  - AFM_ROI_Toolkit: https://github.com/miguel52870/AFM_ROI_Toolkit

Convención de nombres de archivo:
  bifeo_<prefix>_<N>_Canal_<C>.npy   (datos físicos — requerido)
  bifeo_<prefix>_<N>_Canal_<C>.png   (imagen original del microscopio — opcional)
  donde N = índice de frame (cualquier entero >= 1), C = 1 | 2 | 3

Salida: exactamente dos carpetas dentro de --output_dir:
  png_procesado/   -> un PNG por frame/canal preprocesado
  npy_procesado/   -> un NPY por frame/canal preprocesado

─────────────────────────────────────────────────────────────────────────────
USO MÍNIMO
─────────────────────────────────────────────────────────────────────────────
  python afm_preprocess.py \
      --npy_dir ./npy \
      --output_dir ./preprocesado

─────────────────────────────────────────────────────────────────────────────
USO COMPLETO
─────────────────────────────────────────────────────────────────────────────
  python afm_preprocess.py \
      --npy_dir    ./npy \
      --png_dir    ./png \
      --output_dir ./preprocesado \
      --prefix     training \
      --channels   1 2 3 \
      --plane_order      1 \
      --drift_ref_channel 2 \
      --gauss_sigma  1.2 \
      --median_kernel  3 \
      --clahe_clip   2.0 \
      --clahe_tile     8 \
      --save_diff \
      --diagnostics \
      --diag_frame   5 \
      --diag_channel 2
"""

import argparse
import logging
import re
import sys
from collections import OrderedDict
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter, median_filter

# ══════════════════════════════════════════════════════════════════════════════
#     CONFIGURACIÓN DIRECTA
#     Modifica este bloque para ajustar el pipeline sin usar la terminal.
#     Cuando hayas terminado de ajustar, ejecuta simplemente:
#         python afm_preprocess.py
# ══════════════════════════════════════════════════════════════════════════════

CONFIG = {
    # ── Rutas ────────────────────────────────────────────────────────────────
    # Carpeta con los archivos .npy originales (requerido)
    "npy_dir":    "C:/Users/migue/Desktop/imagenes_AFM/canales_separados_gwy/canales_separados_npy",
    # Carpeta con los .png originales del microscopio (opcional).
    # Si no tienes PNGs o no los quieres usar, pon None:
    #   "png_dir": None
    "png_dir":    "C:/Users/migue/Desktop/imagenes_AFM/canales_separados_gwy/canales_separados_png",
    # Carpeta donde se crearán png_procesado/ y npy_procesado/
    "output_dir": "C:/Users/migue/Desktop/AFM_Preprocessing/Results",
    # ── Identificación de la serie ───────────────────────────────────────────z
    # Palabra entre "bifeo_" y el número en el nombre del archivo.
    # Ejemplo: "bifeo_training_21_Canal_1.npy"  →  prefix = "training"
    "prefix":   "training",
    # Canales a procesar: cualquier combinación de [1, 2, 3]
    # 1 = superficie, 2 = amplitud PFM, 3 = fase PFM
    "channels": [1, 2, 3],
    # ── Parámetros de preprocesamiento ───────────────────────────────────────
    # Grado del polinomio de corrección de plano:
    #   1 = plano simple (tilt)
    #   2 = paraboloide (tilt + curvatura/bow)
    "plane_order": 1,
    # Canal usado para calcular el drift de la serie.
    # Usa el que tenga mejor contraste estructural en tu muestra.
    "drift_ref_channel": 2,
    # Sigma del filtro gaussiano para C1 y C3.
    # Más alto = más suavizado. Rango típico: 0.8 – 1.5
    "gauss_sigma": 1.2,
    # Tamaño del kernel del filtro mediana para C2 (debe ser impar).
    # 3 es suficiente para la mayoría de los casos.
    "median_kernel": 3,
    # Percentiles para la normalización de C1 y C3.
    # p2-p98 es conservador e ignora outliers extremos.
    # Reduce norm_plow / aumenta norm_phigh para incluir más rango.
    # Aumenta norm_plow / reduce norm_phigh para recortar más agresivamente.
    "norm_plow":  2,
    "norm_phigh": 98,
    # CLAHE desactivado (era demasiado agresivo encadenado con z-score).
    # Estos valores se conservan por si decides reactivarlo manualmente.
    "clahe_clip": 0.1,
    "clahe_tile": 8,
    # ── Opciones de salida ───────────────────────────────────────────────────
    # True  = guardar imágenes diferencia |f_n - f_{n-1}| de C2
    # False = no guardarlas
    "save_diff": True,
    # ── Diagnóstico ──────────────────────────────────────────────────────────
    # True  = generar figuras comparativas y reporte HTML en diagnostics/
    # False = omitir diagnóstico (más rápido)
    "diagnostics": True,
    # Índice del frame representativo para el diagnóstico.
    # None = usar el frame central de la serie automáticamente.
    "diag_frame": 10,
    # Canal analizado en detalle en el diagnóstico (1, 2 o 3).
    "diag_channel": 2,
}

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("afm_preprocess")

CMAPS = {1: "afmhot", 2: "RdBu_r", 3: "twilight"}
# C1 (superficie): gray — la topografía no necesita falso color
# C2 (amplitud):   RdBu_r — divergente, centrado en cero, muestra dominios
# C3 (fase):       gray — evita saturación visual del colormap PiYG


# ──────────────────────────────────────────────────────────────────────────────
# Importación diferida del módulo de diagnóstico
# ──────────────────────────────────────────────────────────────────────────────
def _load_diagnostics():
    try:
        import afm_diagnostics as diag
        return diag
    except ImportError:
        log.error("afm_diagnostics.py no encontrado junto a este script.")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# 1.  DESCUBRIMIENTO DE ARCHIVOS
# ══════════════════════════════════════════════════════════════════════════════

def _scan_dir(directory: Path, prefix: str, channels: list, ext: str) -> dict:
    """Escanea `directory`. Devuelve { frame_idx: { canal: Path } }"""
    pattern = re.compile(
        r"^bifeo_" + re.escape(prefix) + r"_(\d+)_Canal_(\d+)\." + re.escape(ext) + r"$",
        re.IGNORECASE,
    )
    found = {}
    for f in sorted(directory.iterdir()):
        m = pattern.match(f.name)
        if not m:
            continue
        idx   = int(m.group(1))
        canal = int(m.group(2))
        if canal not in channels:
            continue
        found.setdefault(idx, {})[canal] = f
    return found


def discover_series(npy_dir, png_dir, prefix, channels):
    """Devuelve (npy_index, png_index) — dicts { frame_idx: { canal: Path } }"""
    npy_index = _scan_dir(npy_dir, prefix, channels, "npy")
    if not npy_index:
        log.error(f"No se encontraron .npy con patron 'bifeo_{prefix}_N_Canal_C.npy' en {npy_dir}")
        sys.exit(1)

    png_index = {}
    if png_dir is not None:
        png_index = _scan_dir(png_dir, prefix, channels, "png")
        if not png_index:
            log.warning(f"--png_dir dado pero no se encontraron PNGs en {png_dir}. Se ignora.")

    frame_ids = sorted(npy_index.keys())
    log.info(
        f"Serie: {len(frame_ids)} frames (indices {frame_ids[0]}-{frame_ids[-1]}), "
        f"canales {channels}"
    )
    if png_index:
        n_png = sum(len(v) for v in png_index.values())
        log.info(f"  PNGs originales disponibles: {n_png} archivos")
    return npy_index, png_index


# ══════════════════════════════════════════════════════════════════════════════
# 2.  LECTURA / ESCRITURA
# ══════════════════════════════════════════════════════════════════════════════

def load_npy(path):
    data = np.load(path).astype(np.float64)
    if data.ndim == 3:
        data = data[:, :, 0]
    return data


def load_png_gray(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"No se pudo abrir {path}")
    return img.astype(np.float64) / 255.0


def _render_png(data, path, cmap):
    """
    Renderiza array float a PNG al tamaño original del array (sin escalar).
    Preserva la resolución exacta del dato AFM para que YOLO trabaje
    en el mismo espacio de coordenadas que las imagenes originales.
    Usa percentiles 1-99 para el rango: evita que outliers aplasten el contraste.
    """
    lo = np.percentile(data, 1)
    hi = np.percentile(data, 99)
    norm = np.clip((data - lo) / (hi - lo + 1e-12), 0, 1)
    # Aplicar colormap y convertir a uint8 sin ningún escalado
    colored = plt.get_cmap(cmap)(norm)                     # RGBA float [0,1]
    img_rgb = (colored[:, :, :3] * 255).astype(np.uint8)  # RGB uint8
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)     # BGR para OpenCV
    cv2.imwrite(str(path), img_bgr)


def save_results(processed, src_stems,
                 png_prep, npy_prep,
                 png_diff, npy_diff,
                 save_diff):
    """
    Guarda todos los frames en las subcarpetas de salida:
      prep/ -> imagenes preprocesadas (una por frame y canal)
      diff/ -> diferencias entre frames consecutivos de C2 (si save_diff=True)
    Devuelve (diff_means, diff_maxs) para C2 (usados en diagnostico).
    """
    diff_means = []
    diff_maxs  = []

    for canal, frames in processed.items():
        cmap = CMAPS.get(canal, "gray")
        prev = None
        for frame, stem in zip(frames, src_stems[canal]):
            if frame is None or stem is None:
                prev = None
                continue

            # prep/
            np.save(npy_prep / f"{stem}_prep.npy", frame.astype(np.float32))
            _render_png(frame, png_prep / f"{stem}_prep.png", cmap)

            # diff/ — calculado antes de z-score en el pipeline principal.
            # Aquí solo acumulamos estadísticas para el diagnóstico.
            if canal == 2 and save_diff and prev is not None:
                diff = np.abs(frame - prev)
                diff_means.append(float(diff.mean()))
                diff_maxs.append(float(diff.max()))
            if canal == 2:
                prev = frame

    return diff_means, diff_maxs


# ══════════════════════════════════════════════════════════════════════════════
# 3.  ETAPAS DE PREPROCESAMIENTO
# ══════════════════════════════════════════════════════════════════════════════

def plane_correction(data, order=1):
    """Devuelve (corregido, fondo)."""
    h, w = data.shape
    x = np.linspace(0, 1, w)
    y = np.linspace(0, 1, h)
    X, Y = np.meshgrid(x, y)
    if order == 1:
        A = np.column_stack([np.ones(h * w), X.ravel(), Y.ravel()])
    else:
        A = np.column_stack([
            np.ones(h * w), X.ravel(), Y.ravel(),
            (X**2).ravel(), (X * Y).ravel(), (Y**2).ravel(),
        ])
    coeffs, _, _, _ = np.linalg.lstsq(A, data.ravel(), rcond=None)
    bg = (A @ coeffs).reshape(h, w)
    return data - bg, bg


def line_correction(data):
    return data - np.median(data, axis=1, keepdims=True)


def _phase_shift(ref, target):
    from numpy.fft import fft2, ifft2, fftshift
    R = fft2(ref)
    T = fft2(target)
    cross = R * np.conj(T)
    pc = np.real(ifft2(cross / (np.abs(cross) + 1e-12)))
    pc = fftshift(pc)
    peak = np.unravel_index(np.argmax(pc), pc.shape)
    cy, cx = np.array(pc.shape) // 2
    return int(peak[0] - cy), int(peak[1] - cx)


def apply_shift(data, dy, dx):
    s = np.roll(data, (dy, dx), axis=(0, 1))
    fill = np.median(data)
    if dy > 0:  s[:dy, :]  = fill
    elif dy < 0: s[dy:, :] = fill
    if dx > 0:  s[:, :dx]  = fill
    elif dx < 0: s[:, dx:] = fill
    return s


def compute_series_shifts(frames):
    shifts = [(0, 0)]
    cum_dy, cum_dx = 0, 0
    ref = frames[0]
    for target in frames[1:]:
        dy, dx = _phase_shift(ref, target)
        cum_dy += dy
        cum_dx += dx
        shifts.append((cum_dy, cum_dx))
        ref = target
    return shifts


def filter_c1_c3(data, sigma=1.2):
    return gaussian_filter(data, sigma=sigma)


def filter_c2_median(data, kernel=3):
    return median_filter(data, size=kernel)


def apply_clahe(data, clip_limit=2.0, tile_grid=8):
    d_min, d_max = data.min(), data.max()
    u16 = ((data - d_min) / (d_max - d_min + 1e-12) * 65535).astype(np.uint16)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    return clahe.apply(u16).astype(np.float64) / 65535.0


def normalize_percentile_global(frames, p_low=2, p_high=98):
    """
    Normalización por percentiles globales de la serie.
    Más conservadora que min-max: ignora outliers extremos.
    p_low=2, p_high=98 por defecto — ajustable en CONFIG.
    """
    all_vals = np.concatenate([f.ravel() for f in frames])
    lo = np.percentile(all_vals, p_low)
    hi = np.percentile(all_vals, p_high)
    d  = hi - lo + 1e-12
    return [np.clip((f - lo) / d, 0, 1) for f in frames]


def normalize_zscore_global(frames):
    all_vals = np.concatenate([f.ravel() for f in frames])
    mu    = all_vals.mean()
    sigma = all_vals.std() + 1e-12
    return [(f - mu) / sigma for f in frames]


def otsu_domain_mask(data):
    u8 = ((data - data.min()) / (data.max() - data.min() + 1e-12) * 255).astype(np.uint8)
    _, mask = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask.astype(np.uint8)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  PIPELINE PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(args):
    npy_dir    = Path(args.npy_dir)
    png_dir    = Path(args.png_dir) if args.png_dir else None
    output_dir = Path(args.output_dir)

    # Carpetas de salida — una por tipo de archivo
    png_prep = output_dir / "png_procesado" / "prep"
    npy_prep = output_dir / "npy_procesado" / "prep"
    png_mask = output_dir / "png_procesado" / "mask"
    npy_mask = output_dir / "npy_procesado" / "mask"
    png_diff = output_dir / "png_procesado" / "diff"
    npy_diff = output_dir / "npy_procesado" / "diff"

    for d in [png_prep, npy_prep, png_mask, npy_mask, png_diff, npy_diff]:
        d.mkdir(parents=True, exist_ok=True)

    # ── 1. Descubrir archivos ────────────────────────────────────────────────
    npy_index, png_index = discover_series(npy_dir, png_dir, args.prefix, args.channels)
    frame_ids = sorted(npy_index.keys())

    # ── 2. Cargar datos ──────────────────────────────────────────────────────
    raw           = {c: [] for c in args.channels}
    src_stems     = {c: [] for c in args.channels}
    png_originals = {c: [] for c in args.channels}

    for idx in frame_ids:
        for c in args.channels:
            npy_path = npy_index.get(idx, {}).get(c)
            if npy_path is None:
                log.warning(f"Frame {idx} Canal {c}: .npy no encontrado, se omite.")
                raw[c].append(None)
                src_stems[c].append(None)
                png_originals[c].append(None)
                continue
            raw[c].append(load_npy(npy_path))
            src_stems[c].append(npy_path.stem)

            png_path = png_index.get(idx, {}).get(c) if png_index else None
            if png_path is not None:
                try:
                    png_originals[c].append(load_png_gray(png_path))
                except IOError as e:
                    log.warning(str(e))
                    png_originals[c].append(None)
            else:
                png_originals[c].append(None)

    log.info(f"Cargados {len(frame_ids)} frames x {len(args.channels)} canales.")

    # ── 3. Corrección de plano ───────────────────────────────────────────────
    log.info("(1) Correccion de plano...")
    for c in args.channels:
        corrected = []
        for f in raw[c]:
            if f is None:
                corrected.append(None)
            else:
                corr, _ = plane_correction(f, order=args.plane_order)
                corrected.append(corr)
        raw[c] = corrected

    # ── 4. Corrección línea a línea ─────────────────────────────────────────
    log.info("(2) Correccion linea a linea...")
    for c in args.channels:
        raw[c] = [line_correction(f) if f is not None else None for f in raw[c]]

    # ── 5. Drift ─────────────────────────────────────────────────────────────
    log.info(f"(3) Drift con Canal {args.drift_ref_channel}...")
    ref_frames = [f for f in raw[args.drift_ref_channel] if f is not None]
    shifts = compute_series_shifts(ref_frames)

    shift_map = []
    it = iter(shifts)
    for f in raw[args.drift_ref_channel]:
        shift_map.append(next(it) if f is not None else (0, 0))

    for c in args.channels:
        raw[c] = [
            apply_shift(f, dy, dx) if f is not None else None
            for f, (dy, dx) in zip(raw[c], shift_map)
        ]
    log.info(f"  Drift maximo acumulado: {max(abs(dy)+abs(dx) for dy,dx in shifts)} px")

    # ── 6. Filtros + normalización ───────────────────────────────────────────
    processed = {c: list(raw[c]) for c in args.channels}

    for c in [1, 3]:
        if c not in args.channels:
            continue
        log.info(f"(4) Filtro gaussiano C{c} (sigma={args.gauss_sigma})...")
        processed[c] = [
            filter_c1_c3(f, sigma=args.gauss_sigma) if f is not None else None
            for f in processed[c]
        ]

    log.info(f"(5) Normalizacion por percentiles p{args.norm_plow}-p{args.norm_phigh} global C1/C3...")
    for c in [1, 3]:
        if c not in args.channels:
            continue
        valid  = [f for f in processed[c] if f is not None]
        normed = normalize_percentile_global(valid, p_low=args.norm_plow, p_high=args.norm_phigh)
        it2    = iter(normed)
        processed[c] = [next(it2) if f is not None else None for f in processed[c]]

    if 3 in args.channels:
        log.info("(6) Mascaras Otsu C3...")
        for frame, stem in zip(processed[3], src_stems[3]):
            if frame is None or stem is None:
                continue
            mask = otsu_domain_mask(frame)
            np.save(npy_mask / f"{stem}_mask.npy", mask)
            _render_png(mask.astype(np.float64), png_mask / f"{stem}_mask.png", "gray")

    post_filter_c2 = []
    post_clahe_c2  = []   # reservado para diagnostico, CLAHE desactivado

    if 2 in args.channels:
        log.info(f"(4) Filtro mediana C2 (kernel={args.median_kernel})...")
        processed[2] = [
            filter_c2_median(f, kernel=args.median_kernel) if f is not None else None
            for f in processed[2]
        ]
        post_filter_c2 = list(processed[2])

        # CLAHE desactivado: era demasiado agresivo encadenado con z-score.
        # Se conserva apply_clahe() disponible pero no se aplica en el pipeline.
        # Si deseas reactivarlo, descomenta las siguientes líneas y ajusta clahe_clip a 1.0:
        #processed[2] = [
        #    apply_clahe(f, clip_limit=args.clahe_clip, tile_grid=args.clahe_tile)
        #    if f is not None else None
        #    for f in processed[2]
        #]
        #post_clahe_c2 = list(processed[2])

        # Diff calculado ANTES de z-score para preservar la dinámica real de switching.
        # Los valores de diff corresponden a unidades físicas relativas, no normalizadas.
        if args.save_diff:
            log.info("(5) Calculando diferencias entre frames C2 (pre z-score)...")
            c2_valid = [(f, s) for f, s in zip(processed[2], src_stems[2])
                        if f is not None and s is not None]
            for i in range(1, len(c2_valid)):
                prev_f, _      = c2_valid[i - 1]
                curr_f, curr_s = c2_valid[i]
                diff = np.abs(curr_f - prev_f)
                np.save(npy_diff / f"{curr_s}_diff.npy", diff.astype(np.float32))
                _render_png(diff, png_diff / f"{curr_s}_diff.png", "hot")

        log.info("(6) Normalizacion z-score global C2...")
        valid  = [f for f in processed[2] if f is not None]
        normed = normalize_zscore_global(valid)
        it3    = iter(normed)
        processed[2] = [next(it3) if f is not None else None for f in processed[2]]

    # ── 7. Guardar ───────────────────────────────────────────────────────────
    log.info("Guardando resultados...")
    diff_means, diff_maxs = save_results(
        processed, src_stems,
        png_prep, npy_prep,
        png_diff, npy_diff,
        args.save_diff,
    )

    # ── 8. Diagnóstico ───────────────────────────────────────────────────────
    if args.diagnostics:
        diag = _load_diagnostics()
        if diag is None:
            log.warning("Diagnostico omitido: modulo no disponible.")
        else:
            diag_fi_idx = args.diag_frame if args.diag_frame is not None else frame_ids[len(frame_ids) // 2]
            try:
                fi = frame_ids.index(diag_fi_idx)
            except ValueError:
                fi = len(frame_ids) // 2
                diag_fi_idx = frame_ids[fi]
                log.warning(f"--diag_frame no encontrado, usando frame central: {diag_fi_idx}")

            dc = args.diag_channel if args.diag_channel in args.channels else args.channels[0]

            # Reconstruir snapshots etapa a etapa para el frame representativo
            raw_snap = load_npy(npy_index[diag_fi_idx][dc])
            snap_corr, snap_bg = plane_correction(raw_snap, order=args.plane_order)
            snap_line  = line_correction(snap_corr)
            snap_drift = apply_shift(snap_line, *shifts[fi])

            if dc in [1, 3]:
                snap_filter = filter_c1_c3(snap_drift, sigma=args.gauss_sigma)
                snap_clahe  = None
            else:
                snap_filter = filter_c2_median(snap_drift, kernel=args.median_kernel)
                snap_clahe  = apply_clahe(snap_filter,
                                          clip_limit=args.clahe_clip,
                                          tile_grid=args.clahe_tile)

            snap_final  = processed[dc][fi]
            orig_png    = png_originals[dc][fi]   # None si no se paso --png_dir
            has_orig    = orig_png is not None

            # Construir snapshots: si hay PNG original va primero como columna extra
            stage_snapshots = OrderedDict()
            if has_orig:
                stage_snapshots["Original\n(microscopio)"] = orig_png
            stage_snapshots["Crudo (npy)"]   = raw_snap
            stage_snapshots["Post-plano"]     = snap_corr
            stage_snapshots["Post-linea"]     = snap_line
            stage_snapshots["Post-drift"]     = snap_drift
            stage_snapshots["Post-filtro"]    = snap_filter
            stage_snapshots["Final"]          = snap_final

            series_stats = {}
            for c in args.channels:
                valid = [f for f in processed[c] if f is not None]
                series_stats[c] = {
                    "means": [float(f.mean()) for f in valid],
                    "stds":  [float(f.std())  for f in valid],
                    "mins":  [float(f.min())  for f in valid],
                    "maxs":  [float(f.max())  for f in valid],
                }

            diag.run_diagnostics(
                frame_ids=frame_ids,
                channels=args.channels,
                prefix=args.prefix,
                diag_frame_idx=diag_fi_idx,
                diag_channel=dc,
                stage_snapshots=stage_snapshots,
                raw_frame=raw_snap,
                original_png=orig_png,
                plane_background=snap_bg,
                after_plane=snap_corr,
                after_line=snap_line,
                after_drift=snap_drift,
                after_filter=snap_filter,
                after_clahe=snap_clahe,
                after_norm=snap_final,
                series_stats=series_stats,
                shifts=shifts,
                diff_means=diff_means if diff_means else None,
                diff_maxs=diff_maxs  if diff_maxs  else None,
                output_dir=output_dir,
                has_original_png=has_orig,
            )

    log.info("Pipeline completo.")
    log.info(f"  -> {output_dir.resolve()}")
    log.info(f"     png_procesado/prep/  : PNGs preprocesados")
    log.info(f"     png_procesado/mask/  : Mascaras Otsu C3 (PNG)")
    log.info(f"     png_procesado/diff/  : Diferencias C2 (PNG)")
    log.info(f"     npy_procesado/prep/  : NPYs preprocesados")
    log.info(f"     npy_procesado/mask/  : Mascaras Otsu C3 (NPY)")
    log.info(f"     npy_procesado/diff/  : Diferencias C2 (NPY)")


# ══════════════════════════════════════════════════════════════════════════════
# 5.  ARGUMENTOS CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Pipeline de preprocesamiento AFM — BiFeO3",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--npy_dir",    required=True,
                   help="Carpeta con los .npy originales (requerido)")
    p.add_argument("--png_dir",    default=None,
                   help="Carpeta con los .png originales del microscopio (opcional). "
                        "Mejora las comparativas visuales del diagnostico.")
    p.add_argument("--output_dir", required=True,
                   help="Carpeta de salida. Genera png_procesado/ y npy_procesado/ dentro.")
    p.add_argument("--prefix",     default="training",
                   help="Prefijo del nombre de archivo (bifeo_<PREFIX>_N_Canal_C)")
    p.add_argument("--channels",   nargs="+", type=int, default=[1, 2, 3], choices=[1, 2, 3])

    p.add_argument("--plane_order",       type=int,   default=1,   choices=[1, 2])
    p.add_argument("--drift_ref_channel", type=int,   default=2,   choices=[1, 2, 3])
    p.add_argument("--gauss_sigma",       type=float, default=1.2)
    p.add_argument("--median_kernel",     type=int,   default=3)
    p.add_argument("--norm_plow",         type=int,   default=2,
                   help="Percentil inferior para normalización de C1/C3 (default: 2)")
    p.add_argument("--norm_phigh",        type=int,   default=98,
                   help="Percentil superior para normalización de C1/C3 (default: 98)")
    p.add_argument("--clahe_clip",        type=float, default=1.0,
                   help="Clip limit CLAHE — desactivado por defecto, solo si se reactiva")
    p.add_argument("--clahe_tile",        type=int,   default=8)

    p.add_argument("--save_diff",   action="store_true",
                   help="Guardar imagenes diferencia |f_n - f_{n-1}| de C2")
    p.add_argument("--diagnostics", action="store_true",
                   help="Generar figuras de diagnostico y reporte HTML")
    p.add_argument("--diag_frame",  type=int, default=None,
                   help="Indice del frame representativo para diagnostico (default: central)")
    p.add_argument("--diag_channel", type=int, default=2, choices=[1, 2, 3])

    return p.parse_args()

# ══════════════════════════════════════════════════════════════════════════════
# 7.  ENTRYPOINT  (no modificar)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    # Convertir CONFIG en un namespace idéntico al que produce argparse,
    # para que run_pipeline() funcione igual en ambos modos.
    args = argparse.Namespace(**CONFIG)
    run_pipeline(args)