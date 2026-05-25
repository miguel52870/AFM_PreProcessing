"""
AFM Preprocessing — Módulo de Diagnóstico Visual
=================================================
Genera figuras comparativas antes/después de cada etapa del pipeline,
histogramas, mapa de drift y reporte HTML navegable.

Se invoca desde afm_preprocess.py con --diagnostics.
Todas las salidas van a:  <output_dir>/diagnostics/
El reporte navegable es: <output_dir>/diagnostics/index.html
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

log = logging.getLogger("afm_diagnostics")

CMAPS = {1: "afmhot", 2: "RdBu_r", 3: "PiYG"}
CANAL_LABELS = {
    1: "Canal 1 — Superficie (topografia)",
    2: "Canal 2 — Amplitud PFM",
    3: "Canal 3 — Fase PFM",
}


# ══════════════════════════════════════════════════════════════════════════════
# Utilidades internas
# ══════════════════════════════════════════════════════════════════════════════

def _norm(data: np.ndarray) -> np.ndarray:
    """Normaliza a [0,1] usando percentiles 1-99 para evitar aplastamiento."""
    lo = np.percentile(data, 1)
    hi = np.percentile(data, 99)
    return np.clip((data - lo) / (hi - lo + 1e-12), 0, 1)


def _imshow(ax, data, cmap, title, colorbar=False):
    im = ax.imshow(_norm(data), cmap=cmap, vmin=0, vmax=1, aspect="equal")
    ax.set_title(title, fontsize=8, pad=3)
    ax.axis("off")
    if colorbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return im


def _hist(ax, data, color, title, bins=70):
    vals = data.ravel()
    ax.hist(vals, bins=bins, color=color, alpha=0.75, edgecolor="none")
    ax.axvline(np.median(vals), color="k",       lw=0.9, ls="--",
               label=f"med={np.median(vals):.3f}")
    ax.axvline(np.mean(vals),   color="crimson", lw=0.9, ls=":",
               label=f"mu={np.mean(vals):.3f}")
    ax.set_title(title, fontsize=8, pad=3)
    ax.set_ylabel("Frecuencia", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.legend(fontsize=6)


def _save(fig, path, dpi=150):
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info(f"  -> {path.name}")


# ══════════════════════════════════════════════════════════════════════════════
# 1. Comparativa de etapas (columna por etapa)
# ══════════════════════════════════════════════════════════════════════════════

def diag_stage_comparison(stage_snapshots, canal, output_dir,
                          frame_label, has_original_png):
    """
    Layout 4+3:
      Fila 0: imágenes  etapas 0-3   (4 columnas)
      Fila 1: histos    etapas 0-3
      Fila 2: imágenes  etapas 4-6   (3 columnas + 1 vacía)
      Fila 3: histos    etapas 4-6   (3 columnas + 1 vacía)

    Si hay PNG original lo muestra con borde naranja en la primera celda.
    """
    if not stage_snapshots:
        print("  [diag_etapas] stage_snapshots vacío, se omite.")
        return

    cmap   = CMAPS.get(canal, "gray")
    labels = list(stage_snapshots.keys())
    arrays = list(stage_snapshots.values())
    n      = len(labels)   # normalmente 7 (con PNG) o 6 (sin PNG)

    COLS = 4               # columnas por bloque
    n1   = min(COLS, n)    # etapas en el primer bloque
    n2   = n - n1          # etapas en el segundo bloque

    colors = plt.cm.tab10(np.linspace(0, 0.8, n))

    # 4 filas x 4 columnas; las celdas sobrantes del bloque 2 quedan ocultas
    fig, axes = plt.subplots(4, COLS, figsize=(4.2 * COLS, 9))

    def _draw_col(row_img, row_hist, col_ax, idx, label, data, color):
        """Dibuja imagen + histograma en la posición indicada."""
        ax_img  = axes[row_img,  col_ax]
        ax_hist = axes[row_hist, col_ax]
        _imshow(ax_img, data, cmap, label)
        # Borde naranja para el PNG original
        if idx == 0 and has_original_png:
            for spine in ax_img.spines.values():
                spine.set_edgecolor("#e07b00")
                spine.set_linewidth(2.5)
                spine.set_visible(True)
            ax_img.set_title(label, fontsize=8, pad=3,
                             color="#e07b00", fontweight="bold")
        _hist(ax_hist, data, color=color,
              title=f"Histograma\n{label.split(chr(10))[0]}")

    # Bloque 1: filas 0-1, columnas 0 a n1-1
    for i in range(n1):
        _draw_col(0, 1, i, i, labels[i], arrays[i], colors[i])

    # Bloque 2: filas 2-3, columnas 0 a n2-1
    for j in range(n2):
        _draw_col(2, 3, j, n1 + j, labels[n1 + j], arrays[n1 + j], colors[n1 + j])

    # Ocultar celdas sobrantes del bloque 2
    for j in range(n2, COLS):
        axes[2, j].set_visible(False)
        axes[3, j].set_visible(False)

    # Leyenda borde naranja
    if has_original_png:
        fig.text(0.01, 0.99,
                 "Columna naranja = imagen original del microscopio",
                 fontsize=7, color="#e07b00", va="top")

    fig.suptitle(
        f"Comparativa de etapas — {CANAL_LABELS.get(canal, f'Canal {canal}')} — {frame_label}",
        fontsize=10, fontweight="bold", y=1.005,
    )
    fig.tight_layout()
    try:
        _save(fig, output_dir / f"diag_etapas_C{canal}_{frame_label}.png")
        print(f"  [diag_etapas] guardado: diag_etapas_C{canal}_{frame_label}.png")
    except Exception as e:
        print(f"  [diag_etapas] ERROR al guardar: {e}")
    finally:
        plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# 2. Corrección de plano
# ══════════════════════════════════════════════════════════════════════════════

def diag_plane_correction(raw_npy, original_png, corrected, background,
                          canal, output_dir, frame_label):
    """
    Si hay PNG original: muestra 5 columnas:
      Original (microscopio) | Crudo npy | Fondo ajustado | Corregido | Perfil
    Si no hay PNG original: 4 columnas (sin la primera).
    """
    cmap = CMAPS.get(canal, "gray")
    has_orig = original_png is not None
    ncols = 5 if has_orig else 4

    fig = plt.figure(figsize=(3.5 * ncols, 2.5))
    gs  = gridspec.GridSpec(1, ncols, figure=fig, wspace=0.3)
    axes = [fig.add_subplot(gs[i]) for i in range(ncols)]

    col = 0
    if has_orig:
        _imshow(axes[col], original_png, cmap, "Original\n(microscopio)", colorbar=False)
        for spine in axes[col].spines.values():
            spine.set_edgecolor("#e07b00")
            spine.set_linewidth(2.5)
            spine.set_visible(True)
        axes[col].set_title("Original\n(microscopio)", fontsize=8, color="#e07b00", fontweight="bold")
        col += 1

    _imshow(axes[col], raw_npy,     cmap,        "Crudo (npy)",     colorbar=True); col += 1
    _imshow(axes[col], background,  "RdYlBu_r",  "Fondo ajustado",  colorbar=True); col += 1
    _imshow(axes[col], corrected,   cmap,        "Corregido",        colorbar=True); col += 1

    # Perfil horizontal central
    cy = raw_npy.shape[0] // 2
    x  = np.arange(raw_npy.shape[1])
    ax = axes[col]
    ax.plot(x, raw_npy[cy],   lw=0.9, label="Crudo npy",  color="steelblue")
    ax.plot(x, corrected[cy], lw=0.9, label="Corregido",  color="tomato")
    if has_orig:
        # Escalar PNG original al rango del npy para comparar forma
        orig_scaled = original_png[cy] * (raw_npy.max() - raw_npy.min()) + raw_npy.min()
        ax.plot(x, orig_scaled, lw=0.8, label="Original\n(microscopio)",
                color="#e07b00", ls="--")
    ax.set_title("Perfil linea central", fontsize=8)
    ax.set_xlabel("Pixel X", fontsize=7)
    ax.legend(fontsize=6)
    ax.tick_params(labelsize=6)

    if has_orig:
        fig.text(0.01, 0.99, "Columna naranja = imagen original del microscopio",
                 fontsize=7, color="#e07b00", va="top")

    fig.suptitle(
        f"Correccion de plano — {CANAL_LABELS.get(canal, f'Canal {canal}')} — {frame_label}",
        fontsize=10, fontweight="bold",
    )
    _save(fig, output_dir / f"diag_plane_C{canal}_{frame_label}.png")


# ══════════════════════════════════════════════════════════════════════════════
# 3. Corrección línea a línea
# ══════════════════════════════════════════════════════════════════════════════

def diag_line_correction(before, after, canal, output_dir, frame_label):
    cmap = CMAPS.get(canal, "gray")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    _imshow(axes[0], before, cmap, "Antes")
    _imshow(axes[1], after,  cmap, "Despues")

    med_before = np.median(before, axis=1)
    med_after  = np.median(after,  axis=1)
    y = np.arange(len(med_before))
    axes[2].plot(med_before, y, lw=0.9, label="Antes",   color="steelblue")
    axes[2].plot(med_after,  y, lw=0.9, label="Despues", color="tomato")
    axes[2].invert_yaxis()
    axes[2].set_title("Mediana por linea\n(offset de scan)", fontsize=8)
    axes[2].set_xlabel("Valor mediana", fontsize=7)
    axes[2].set_ylabel("Linea (px)", fontsize=7)
    axes[2].legend(fontsize=7)
    axes[2].tick_params(labelsize=6)

    fig.suptitle(
        f"Correccion linea a linea — {CANAL_LABELS.get(canal, f'Canal {canal}')} — {frame_label}",
        fontsize=10, fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, output_dir / f"diag_line_C{canal}_{frame_label}.png")


# ══════════════════════════════════════════════════════════════════════════════
# 4. Drift de la serie
# ══════════════════════════════════════════════════════════════════════════════

def diag_drift(frame_ids, shifts, output_dir):
    dys = [s[0] for s in shifts]
    dxs = [s[1] for s in shifts]
    ids = list(frame_ids[:len(shifts)])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(ids, dys, "o-", lw=1, ms=3, label="Drift Y (filas)", color="steelblue")
    ax1.plot(ids, dxs, "s-", lw=1, ms=3, label="Drift X (cols)",  color="tomato")
    ax1.axhline(0, color="k", lw=0.5, ls="--")
    ax1.set_xlabel("Indice de frame", fontsize=9)
    ax1.set_ylabel("Desplazamiento acumulado (px)", fontsize=9)
    ax1.set_title("Drift acumulado por frame", fontsize=10)
    ax1.legend(fontsize=8)
    ax1.tick_params(labelsize=7)

    sc = ax2.scatter(dxs, dys, c=ids, cmap="viridis", s=20, zorder=3)
    ax2.plot(dxs, dys, lw=0.6, color="gray", zorder=2)
    ax2.scatter([dxs[0]],  [dys[0]],  marker="^", s=70, color="green", zorder=4, label="Frame inicial")
    ax2.scatter([dxs[-1]], [dys[-1]], marker="v", s=70, color="red",   zorder=4, label="Frame final")
    plt.colorbar(sc, ax=ax2, label="Indice de frame")
    ax2.set_xlabel("Drift X (px)", fontsize=9)
    ax2.set_ylabel("Drift Y (px)", fontsize=9)
    ax2.set_title("Trayectoria 2D de drift", fontsize=10)
    ax2.legend(fontsize=8)
    ax2.tick_params(labelsize=7)
    ax2.axhline(0, color="k", lw=0.4, ls=":")
    ax2.axvline(0, color="k", lw=0.4, ls=":")
    r = max(max(abs(v) for v in dys + dxs), 1) * 1.15
    ax2.set_xlim(-r, r)
    ax2.set_ylim(-r, r)
    ax2.set_aspect("equal")

    fig.suptitle("Analisis de drift — serie completa", fontsize=10, fontweight="bold")
    fig.tight_layout()
    _save(fig, output_dir / "diag_drift_series.png")


# ══════════════════════════════════════════════════════════════════════════════
# 5. CLAHE (Canal 2)
# ══════════════════════════════════════════════════════════════════════════════

def diag_clahe(before, after, original_png, output_dir, frame_label):
    """
    Si hay PNG original: 3 columnas de imagen + 2 histogramas.
    Si no hay: 2 columnas + 2 histogramas.
    """
    has_orig = original_png is not None
    ncols_img = 3 if has_orig else 2

    fig = plt.figure(figsize=(5 * ncols_img, 7))
    gs  = gridspec.GridSpec(2, ncols_img, figure=fig, hspace=0.35, wspace=0.3)

    col = 0
    if has_orig:
        ax = fig.add_subplot(gs[0, col])
        _imshow(ax, original_png, "RdBu_r", "Original\n(microscopio)")
        for spine in ax.spines.values():
            spine.set_edgecolor("#e07b00"); spine.set_linewidth(2.5); spine.set_visible(True)
        ax.set_title("Original\n(microscopio)", fontsize=8, color="#e07b00", fontweight="bold")
        col += 1

    ax_before = fig.add_subplot(gs[0, col]); col += 1
    ax_after  = fig.add_subplot(gs[0, col])
    _imshow(ax_before, before, "RdBu_r", "Antes de CLAHE")
    _imshow(ax_after,  after,  "RdBu_r", "Despues de CLAHE")

    bins = np.linspace(0, 1, 80)

    def _hist_cdf(ax, data, color_h, color_c, title):
        norm_d = _norm(data)
        ax.hist(norm_d.ravel(), bins=bins, color=color_h, alpha=0.75, edgecolor="none")
        ax.set_title(title, fontsize=8)
        ax.tick_params(labelsize=6)
        ax2 = ax.twinx()
        cdf = np.cumsum(np.histogram(norm_d.ravel(), bins=bins)[0])
        cdf = cdf / cdf[-1]
        ax2.plot(bins[:-1], cdf, color=color_c, lw=1, ls="--", label="CDF")
        ax2.set_ylim(0, 1.05)
        ax2.set_ylabel("CDF", fontsize=6)
        ax2.tick_params(labelsize=5)

    if has_orig:
        ax_h_orig   = fig.add_subplot(gs[1, 0])
        ax_h_before = fig.add_subplot(gs[1, 1])
        ax_h_after  = fig.add_subplot(gs[1, 2])
        _hist_cdf(ax_h_orig,   original_png, "#e07b00", "saddlebrown", "Histograma\nOriginal (microscopio)")
        _hist_cdf(ax_h_before, before,       "steelblue", "navy",      "Histograma\nAntes de CLAHE")
        _hist_cdf(ax_h_after,  after,        "tomato",    "darkred",   "Histograma\nDespues de CLAHE")
        fig.text(0.01, 0.99, "Columna naranja = imagen original del microscopio",
                 fontsize=7, color="#e07b00", va="top")
    else:
        ax_h_before = fig.add_subplot(gs[1, 0])
        ax_h_after  = fig.add_subplot(gs[1, 1])
        _hist_cdf(ax_h_before, before, "steelblue", "navy",    "Histograma\nAntes de CLAHE")
        _hist_cdf(ax_h_after,  after,  "tomato",    "darkred", "Histograma\nDespues de CLAHE")

    fig.suptitle(
        f"CLAHE — Canal 2 (Amplitud PFM) — {frame_label}",
        fontsize=10, fontweight="bold",
    )
    _save(fig, output_dir / f"diag_clahe_C2_{frame_label}.png")


# ══════════════════════════════════════════════════════════════════════════════
# 6. Estadísticas de serie
# ══════════════════════════════════════════════════════════════════════════════

def diag_normalization_stats(frame_ids, means, stds, mins, maxs,
                             canal, output_dir):
    ids = list(frame_ids[:len(means)])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.fill_between(ids,
                     [m - s for m, s in zip(means, stds)],
                     [m + s for m, s in zip(means, stds)],
                     alpha=0.25, color="steelblue", label="±sigma")
    ax1.plot(ids, means, "o-", lw=1, ms=3, color="steelblue", label="media")
    ax1.set_title(f"Media ± sigma por frame (C{canal})", fontsize=9)
    ax1.set_xlabel("Indice de frame", fontsize=8)
    ax1.legend(fontsize=8)
    ax1.tick_params(labelsize=7)

    ax2.plot(ids, maxs, "^-", lw=1, ms=3, color="tomato",    label="max")
    ax2.plot(ids, mins, "v-", lw=1, ms=3, color="steelblue", label="min")
    ax2.fill_between(ids, mins, maxs, alpha=0.12, color="gray")
    ax2.set_title(f"Rango [min, max] por frame (C{canal})", fontsize=9)
    ax2.set_xlabel("Indice de frame", fontsize=8)
    ax2.legend(fontsize=8)
    ax2.tick_params(labelsize=7)

    fig.suptitle(f"Evolucion estadistica — Canal {canal}", fontsize=10, fontweight="bold")
    fig.tight_layout()
    _save(fig, output_dir / f"diag_stats_C{canal}.png")


# ══════════════════════════════════════════════════════════════════════════════
# 7. Evolución de diferencias C2
# ══════════════════════════════════════════════════════════════════════════════

def diag_diff_evolution(frame_ids, diff_means, diff_maxs, output_dir):
    ids = list(frame_ids[1:len(diff_means) + 1])
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(ids, diff_means, color="steelblue", alpha=0.7, label="Cambio medio |delta|")
    ax.plot(ids, diff_maxs, "o-", lw=1, ms=3, color="tomato", label="Cambio maximo |delta|")
    ax.set_xlabel("Frame (n)", fontsize=9)
    ax.set_ylabel("|f_n - f_{n-1}|", fontsize=9)
    ax.set_title("Evolucion del cambio entre frames — Canal 2\n"
                 "Picos = posibles eventos de switching", fontsize=9)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    _save(fig, output_dir / "diag_diff_evolution_C2.png")


# ══════════════════════════════════════════════════════════════════════════════
# 8. Reporte HTML navegable
# ══════════════════════════════════════════════════════════════════════════════

def generate_html_report(diag_dir, prefix, frame_label, channels,
                         shifts, frame_ids, has_original_png):
    images = sorted(diag_dir.glob("diag_*.png"))
    if not images:
        return

    sections = {
        "Comparativa de etapas": [],
        "Correccion de plano":   [],
        "Correccion linea a linea": [],
        "Drift": [],
        "CLAHE (Canal 2)": [],
        "Estadisticas de serie": [],
        "Evolucion de diferencias": [],
        "Otros": [],
    }
    kw_map = {
        "etapas": "Comparativa de etapas",
        "plane":  "Correccion de plano",
        "line":   "Correccion linea a linea",
        "drift":  "Drift",
        "clahe":  "CLAHE (Canal 2)",
        "stats":  "Estadisticas de serie",
        "diff":   "Evolucion de diferencias",
    }
    for img in images:
        placed = False
        for kw, sec in kw_map.items():
            if kw in img.name.lower():
                sections[sec].append(img)
                placed = True
                break
        if not placed:
            sections["Otros"].append(img)

    max_drift = max((abs(dy) + abs(dx)) for dy, dx in shifts) if shifts else 0

    has_png_note = (
        "<p style='background:#fff3e0;border-left:4px solid #e07b00;"
        "padding:8px 12px;border-radius:0 6px 6px 0;font-size:.85rem;'>"
        "<b style='color:#e07b00'>Columna naranja</b> en las comparativas = "
        "imagen original del microscopio (del PNG de AFM_ToolKit). "
        "Las demas columnas se generan desde los datos .npy.</p>"
        if has_original_png else ""
    )

    lines = [
        "<!DOCTYPE html><html lang='es'><head>",
        "<meta charset='UTF-8'>",
        "<meta name='viewport' content='width=device-width,initial-scale=1'>",
        "<title>Diagnostico AFM — BiFeO3</title>",
        "<style>",
        "body{font-family:system-ui,sans-serif;max-width:1200px;margin:0 auto;",
        "     padding:2rem;background:#f7f7f5;color:#222;}",
        "h1{font-size:1.4rem;border-bottom:2px solid #444;padding-bottom:.4rem;}",
        "h2{font-size:1.05rem;margin-top:2rem;color:#333;border-left:4px solid #888;",
        "   padding-left:.6rem;}",
        ".meta{background:#fff;border:1px solid #ddd;border-radius:8px;",
        "      padding:1rem 1.2rem;margin-bottom:1.5rem;font-size:.9rem;}",
        ".meta table{border-collapse:collapse;width:100%;}",
        ".meta td{padding:5px 14px;}",
        ".meta tr:nth-child(even){background:#f3f3f1;}",
        ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(500px,1fr));",
        "      gap:1rem;margin-top:.8rem;}",
        ".card{background:#fff;border:1px solid #ddd;border-radius:8px;padding:.8rem;}",
        ".card img{width:100%;border-radius:4px;cursor:zoom-in;}",
        ".card p{font-size:.75rem;color:#666;margin:.3rem 0 0;text-align:center;}",
        "footer{margin-top:3rem;font-size:.75rem;color:#aaa;text-align:center;}",
        "</style></head><body>",
        f"<h1>Reporte de diagnostico AFM — BiFeO3<br>",
        f"<small>Prefijo: <code>{prefix}</code> &nbsp;·&nbsp; "
        f"Frame analizado: <code>{frame_label}</code> &nbsp;·&nbsp; "
        f"Canales: {channels}</small></h1>",
        has_png_note,
        "<div class='meta'><table>",
        f"<tr><td><b>Frames en la serie</b></td><td>{len(frame_ids)}</td></tr>",
        f"<tr><td><b>Rango de indices</b></td><td>{frame_ids[0]} – {frame_ids[-1]}</td></tr>",
        f"<tr><td><b>Drift maximo acumulado</b></td><td>{max_drift} px</td></tr>",
        f"<tr><td><b>Canales procesados</b></td><td>{', '.join(str(c) for c in channels)}</td></tr>",
        f"<tr><td><b>PNG original disponible</b></td><td>{'Si' if has_original_png else 'No — comparativas usan npy crudo'}</td></tr>",
        "</table></div>",
    ]

    for sec_name, imgs in sections.items():
        if not imgs:
            continue
        lines.append(f"<h2>{sec_name}</h2><div class='grid'>")
        for img in imgs:
            caption = img.stem.replace("_", " ")
            lines.append(
                f"<div class='card'>"
                f"<img src='{img.name}' alt='{caption}' loading='lazy'>"
                f"<p>{caption}</p></div>"
            )
        lines.append("</div>")

    lines += [
        "<footer>Generado por afm_diagnostics.py &nbsp;·&nbsp; "
        "Tesis: Evolucion de dominios ferroeléctricos con Deep Learning</footer>",
        "</body></html>",
    ]

    out = diag_dir / "index.html"
    out.write_text("\n".join(lines), encoding="utf-8")
    log.info(f"  -> index.html")


# ══════════════════════════════════════════════════════════════════════════════
# 9. Punto de entrada principal
# ══════════════════════════════════════════════════════════════════════════════

def run_diagnostics(
    *,
    frame_ids,
    channels,
    prefix,
    diag_frame_idx,
    diag_channel,
    stage_snapshots,
    raw_frame,
    original_png,          # None si no se paso --png_dir
    plane_background,
    after_plane,
    after_line,
    after_drift,
    after_filter,
    after_clahe,
    after_norm,
    series_stats,
    shifts,
    diff_means,
    diff_maxs,
    output_dir,
    has_original_png,
):
    diag_dir = Path(output_dir) / "diagnostics"
    diag_dir.mkdir(exist_ok=True)
    log.info(f"Generando diagnosticos en {diag_dir} ...")

    frame_label = f"frame{diag_frame_idx:03d}"
    dc = diag_channel

    # 1. Comparativa de etapas
    diag_stage_comparison(
        stage_snapshots, dc, diag_dir, frame_label, has_original_png
    )

    # 2. Corrección de plano
    diag_plane_correction(
        raw_npy=raw_frame,
        original_png=original_png,
        corrected=after_plane,
        background=plane_background,
        canal=dc,
        output_dir=diag_dir,
        frame_label=frame_label,
    )

    # 3. Corrección línea a línea
    diag_line_correction(
        before=after_plane,
        after=after_line,
        canal=dc,
        output_dir=diag_dir,
        frame_label=frame_label,
    )

    # 4. Drift
    diag_drift(frame_ids=frame_ids, shifts=shifts, output_dir=diag_dir)

    # 5. CLAHE (solo C2)
    if dc == 2 and after_clahe is not None:
        diag_clahe(
            before=after_filter,
            after=after_clahe,
            original_png=original_png,
            output_dir=diag_dir,
            frame_label=frame_label,
        )

    # 6. Estadísticas de serie por canal
    for c, stats in series_stats.items():
        valid_ids = frame_ids[:len(stats["means"])]
        diag_normalization_stats(
            frame_ids=valid_ids,
            means=stats["means"],
            stds=stats["stds"],
            mins=stats["mins"],
            maxs=stats["maxs"],
            canal=c,
            output_dir=diag_dir,
        )

    # 7. Evolución de diferencias C2
    if diff_means and diff_maxs:
        diag_diff_evolution(
            frame_ids=frame_ids,
            diff_means=diff_means,
            diff_maxs=diff_maxs,
            output_dir=diag_dir,
        )

    # 8. Reporte HTML
    generate_html_report(
        diag_dir=diag_dir,
        prefix=prefix,
        frame_label=frame_label,
        channels=channels,
        shifts=shifts,
        frame_ids=frame_ids,
        has_original_png=has_original_png,
    )

    log.info(f"Diagnosticos completos -> {diag_dir / 'index.html'}")