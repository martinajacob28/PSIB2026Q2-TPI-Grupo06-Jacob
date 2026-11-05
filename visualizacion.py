"""
visualizacion.py
================

Figuras del trabajo. Cada función recibe el diccionario que devuelve
``cobb_pipeline.procesar_radiografia`` (y, cuando corresponde, la referencia)
y dibuja una etapa del procesamiento. No modifica ningún resultado.
"""

import numpy as np
import matplotlib.pyplot as plt

import cobb_pipeline as cp

COLORES_CURVAS = ["cyan", "magenta", "lime", "orange"]
COLORES_CALIDAD = {"Alta": "lime", "Media": "orange", "Baja": "red"}


def _global(r, puntos):
    x_min, y_min = r["roi_coords"][:2]
    return puntos + np.array([x_min, y_min])


def comparar_preprocesamientos(img, titulo=""):
    """Original, Gaussiano 5×5, CLAHE y Gaussiano + CLAHE."""
    import cv2
    gauss = cv2.GaussianBlur(img, (5, 5), sigmaX=0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    paneles = {
        "Original": img,
        "Gaussiano 5×5": gauss,
        "CLAHE": clahe.apply(img),
        "Gaussiano + CLAHE": clahe.apply(gauss),
    }
    fig, axes = plt.subplots(1, 4, figsize=(16, 9))
    for ax, (t, im) in zip(axes, paneles.items()):
        ax.imshow(im, cmap="gray", vmin=0, vmax=255)
        ax.set_title(t)
        ax.axis("off")
    fig.suptitle(titulo, fontsize=13)
    plt.tight_layout()
    return fig


def proyeccion_vertical(img, titulo=""):
    """Estimación inicial de la posición horizontal de la columna."""
    from scipy.ndimage import gaussian_filter1d
    proc = cp.preprocesar_imagen(img)
    alto, ancho = proc.shape
    x0, x1 = int(ancho * 0.20), int(ancho * 0.80)
    y0, y1 = int(alto * 0.15), int(alto * 0.80)
    perfil = gaussian_filter1d(proc[y0:y1, x0:x1].mean(axis=0), sigma=5)
    xc = x0 + int(np.argmax(perfil))

    fig, axes = plt.subplots(1, 2, figsize=(11, 7))
    axes[0].imshow(proc, cmap="gray", vmin=0, vmax=255)
    axes[0].add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                    edgecolor="cyan", facecolor="none", lw=1.5))
    axes[0].axvline(xc, color="orange", ls="--", label="Posición inicial estimada")
    axes[0].set_title(titulo)
    axes[0].legend(fontsize=8)
    axes[0].axis("off")
    axes[1].plot(np.arange(x0, x1), perfil)
    axes[1].axvline(xc, color="orange", ls="--")
    axes[1].set_xlabel("Coordenada horizontal [px]")
    axes[1].set_ylabel("Intensidad media suavizada")
    axes[1].set_title("Proyección vertical en la región de búsqueda")
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    return fig


def roi_y_realce(r, titulo=""):
    """ROI, fondo estimado, matriz ósea, Top-Hat e imagen base."""
    x_min, y_min, x_max, y_max = r["roi_coords"]
    fig, axes = plt.subplots(1, 5, figsize=(18, 8))
    axes[0].imshow(r["imagen"], cmap="gray", vmin=0, vmax=255)
    axes[0].add_patch(plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                    edgecolor="cyan", facecolor="none", lw=1.5))
    axes[0].axvline(r["columna_pico"], color="orange", ls="--", lw=1)
    axes[0].set_title("Radiografía y ROI")
    axes[1].imshow(r["fondo"], cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("Fondo estimado (apertura)")
    axes[2].imshow(r["matriz_osea"], cmap="magma")
    axes[2].set_title("Matriz ósea")
    axes[3].imshow(r["tophat"], cmap="magma")
    axes[3].set_title("Realce Top-Hat")
    axes[4].imshow(r["base_osea"], cmap="magma")
    axes[4].set_title("Imagen base (CLAHE + Top-Hat)")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(titulo)
    plt.tight_layout()
    return fig


def niveles_y_eje(r, titulo=""):
    """Perfil vertical, 17 niveles y eje polinómico."""
    base = r["base_osea"]
    xp0, xp1 = r["banda_perfil"]
    f0, f1 = r["filas_validas"]
    picos, ajust = r["picos_vertebrales"], r["ajustado_por_perfil"]
    perfil = r["perfil_corregido"]
    cc, val = r["centros_candidatos"], r["candidatos_validos"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 10),
                             gridspec_kw={"width_ratios": [1, 0.8, 1]})
    axes[0].imshow(base, cmap="magma")
    for i, (fila, a) in enumerate(zip(picos, ajust), start=1):
        axes[0].plot([xp0, xp1], [fila, fila], color="yellow" if a else "orange",
                     ls="-" if a else "--", lw=0.9)
        axes[0].text(xp1 + 5, fila, f"N{i}", color="yellow", fontsize=8, va="center")
    axes[0].set_title("Secuencia de 17 niveles")
    axes[0].axis("off")

    filas = np.arange(base.shape[0])
    axes[1].plot(perfil, filas, color="steelblue")
    axes[1].scatter(perfil[picos[ajust]], picos[ajust], color="gold", s=25,
                    label="Ajustados a un máximo")
    axes[1].scatter(perfil[picos[~ajust]], picos[~ajust], color="orange",
                    marker="x", s=35, label="Posición regularizada")
    axes[1].set_ylim(f1, f0)
    axes[1].set_xlabel("Intensidad ósea corregida")
    axes[1].set_ylabel("Fila dentro de la ROI")
    axes[1].set_title("Perfil vertical")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    axes[2].imshow(base, cmap="magma")
    axes[2].scatter(cc[val, 0], cc[val, 1], color="orange", s=25,
                    label="Candidatos conservados")
    axes[2].scatter(cc[~val, 0], cc[~val, 1], color="red", marker="x", s=35,
                    label="Candidatos descartados")
    axes[2].plot(r["x_eje"], r["y_eje"], color="lime", lw=2, label="Ajuste grado 4")
    axes[2].set_title("Eje estimado")
    axes[2].legend(fontsize=8)
    axes[2].axis("off")
    fig.suptitle(titulo)
    plt.tight_layout()
    return fig


def cajas(r, titulo=""):
    """Cajas orientadas sobre la ROI y la radiografía."""
    if "cajas" not in r:
        r["cajas"] = cp.construir_cajas(r["base_osea"], r["picos_vertebrales"],
                                        r["perfil_corregido"],
                                        r["coeficientes_eje"], r["roi_coords"][:2])
    fig, axes = plt.subplots(1, 2, figsize=(10, 11))
    axes[0].imshow(r["base_osea"], cmap="magma")
    axes[1].imshow(r["imagen"], cmap="gray", vmin=0, vmax=255)
    colores = plt.cm.hsv(np.linspace(0, 0.85, len(r["cajas"])))
    for caja, color in zip(r["cajas"], colores):
        for ax, key in ((axes[0], "esquinas_roi"), (axes[1], "esquinas_globales")):
            e = caja[key]
            ax.add_patch(plt.Polygon(e, closed=True, fill=False, edgecolor=color, lw=1.5))
            ax.text(e[:, 0].min() - 6, e[:, 1].mean(), f"C{caja['candidato']}",
                    color="yellow", fontsize=8, ha="right")
    axes[0].plot(r["x_eje"], r["y_eje"], color="lime", lw=1.5)
    axes[1].plot(r["x_eje"] + r["roi_coords"][0], r["y_eje"] + r["roi_coords"][1],
                 color="lime", lw=1.5)
    axes[0].set_title("Cajas en la ROI")
    axes[1].set_title("Cajas en la radiografía")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(titulo)
    plt.tight_layout()
    return fig


def platillos_calidad(r, titulo=""):
    """Platillos detectados coloreados según el control de calidad."""
    fig, ax = plt.subplots(figsize=(7, 13))
    ax.imshow(r["imagen"], cmap="gray")
    x_min, y_min = r["roi_coords"][:2]
    for i, cal in enumerate(r["tabla_calidad"]["Calidad"]):
        color = COLORES_CALIDAD[cal]
        for recta in (r["platillos_superiores"][i], r["platillos_inferiores"][i]):
            g = _global(r, recta)
            ax.plot(g[:, 0], g[:, 1], color=color, lw=2)
        xt = np.polyval(r["coeficientes_eje"], r["picos_vertebrales"][i]) + x_min - 60
        ax.text(xt, r["picos_vertebrales"][i] + y_min, f"C{i + 1}: {cal}",
                color=color, fontsize=8, ha="right")
    ax.set_title(f"Control de calidad de los platillos\n"
                 f"Verde: alta · Naranja: media · Rojo: baja\n{titulo}")
    ax.axis("off")
    plt.tight_layout()
    return fig


def perfil_inclinaciones(r, titulo=""):
    """Perfil angular medido/suavizado y vértebras límite."""
    idx = np.arange(len(r["angulos_vertebrales"]))
    fig, axes = plt.subplots(1, 2, figsize=(11, 13))
    axes[0].imshow(r["imagen"], cmap="gray")
    for n, (s, i) in enumerate(r["curvas"]):
        c = COLORES_CURVAS[n % 4]
        gs = _global(r, r["platillos_superiores"][s])
        gi = _global(r, r["platillos_inferiores"][i])
        axes[0].plot(gs[:, 0], gs[:, 1], color=c, lw=4,
                     label=f"Curva {n + 1}: C{s + 1}–C{i + 1}")
        axes[0].plot(gi[:, 0], gi[:, 1], color=c, lw=4)
    axes[0].set_title("Vértebras límite propuestas")
    axes[0].legend(fontsize=8)
    axes[0].axis("off")

    axes[1].plot(r["angulos_vertebrales"], idx, "o--", color="lightgray",
                 label="Ángulo medido")
    axes[1].plot(r["angulos_suavizados"], idx, "o-", color="tab:blue",
                 label="Perfil suavizado")
    axes[1].axvline(0, color="black", ls=":", lw=1)
    for n, (s, i) in enumerate(r["curvas"]):
        axes[1].scatter([r["angulos_suavizados"][s], r["angulos_suavizados"][i]],
                        [s, i], color=COLORES_CURVAS[n % 4], s=80, zorder=5)
    axes[1].set_yticks(idx)
    axes[1].set_yticklabels([f"C{i + 1}" for i in idx])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Inclinación [°]")
    axes[1].set_title("Perfil de inclinaciones")
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    fig.suptitle(titulo)
    plt.tight_layout()
    return fig


def cobb(r, titulo="", ax=None, referencia=None):
    """Medición final del ángulo de Cobb sobre la radiografía. Si se pasa la
    referencia, se superponen sus platillos límite en línea punteada."""
    propia = ax is None
    if propia:
        fig, ax = plt.subplots(figsize=(7, 13))
    else:
        fig = ax.figure
    img = r["imagen"]
    ax.imshow(img, cmap="gray")
    x_min, y_min = r["roi_coords"][:2]
    ancho = img.shape[1]
    for n, ((s, i), res) in enumerate(zip(r["curvas"], r["resultados"])):
        c = COLORES_CURVAS[n % 4]
        for recta in (r["platillos_superiores"][s], r["platillos_inferiores"][i]):
            g = _global(r, recta)
            ax.plot(g[:, 0], g[:, 1], color=c, lw=4)
            m = (g[1, 1] - g[0, 1]) / (g[1, 0] - g[0, 0] + 1e-8)
            xe = np.array([0.05 * ancho, 0.95 * ancho])
            ax.plot(xe, g[0, 1] + m * (xe - g[0, 0]), "--", color=c, lw=1)
        yt = (np.mean(r["platillos_superiores"][s][:, 1])
              + np.mean(r["platillos_inferiores"][i][:, 1])) / 2 + y_min
        ax.text(x_min + 15, yt,
                f"Curva {n + 1} (C{s + 1}–C{i + 1}): {res['cobb']:.1f}°\n{res['gravedad']}",
                color=c, fontsize=9,
                bbox={"facecolor": "black", "alpha": 0.65, "edgecolor": c})
    if referencia is not None:
        v = referencia["vertebras"]
        for n, (s, i) in enumerate(referencia["curvas"]):
            ax.plot(v[s, :2, 0], v[s, :2, 1], color="white", lw=2, ls=":")
            ax.plot(v[i, 2:, 0], v[i, 2:, 1], color="white", lw=2, ls=":")
    ax.set_title(titulo or "Medición automática del ángulo de Cobb")
    ax.axis("off")
    if propia:
        plt.tight_layout()
    return fig


def referencia(img, ref, v_sup=None, v_inf=None, titulo=""):
    """Anotaciones de referencia y perfil de inclinaciones de platillos."""
    v = ref["vertebras"]
    niveles = np.arange(1, 18)
    fig, axes = plt.subplots(1, 2, figsize=(11, 12),
                             gridspec_kw={"width_ratios": [1, 0.8]})
    axes[0].imshow(img, cmap="gray", vmin=0, vmax=255)
    for k, vert in enumerate(v, start=1):
        axes[0].plot(vert[:2, 0], vert[:2, 1], color="cyan", lw=1.3,
                     label="Platillo superior" if k == 1 else None)
        axes[0].plot(vert[2:, 0], vert[2:, 1], color="orange", lw=1.3,
                     label="Platillo inferior" if k == 1 else None)
        axes[0].text(vert[:, 0].max() + 0.012 * img.shape[1], vert[:, 1].mean(),
                     f"V{k}", color="yellow", fontsize=7)
    axes[0].set_title("Anotaciones de referencia")
    axes[0].legend(fontsize=8, loc="lower right")
    axes[0].axis("off")
    axes[1].plot(ref["angulos_superiores"], niveles, "o-", color="tab:cyan",
                 label="Superior")
    axes[1].plot(ref["angulos_inferiores"], niveles, "o-", color="tab:orange",
                 label="Inferior")
    axes[1].axvline(0, color="gray", ls=":", lw=1)
    axes[1].set_yticks(niveles)
    axes[1].set_yticklabels([f"V{k}" for k in niveles])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Inclinación [°] (convención de imagen)")
    axes[1].set_title("Inclinación de los platillos")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.suptitle(titulo)
    plt.tight_layout()
    return fig


def centros_vs_referencia(r, ref, titulo=""):
    """Centros automáticos frente a los centros de referencia."""
    x_min, y_min = r["roi_coords"][:2]
    picos = r["picos_vertebrales"]
    auto = np.column_stack([np.polyval(r["coeficientes_eje"], picos) + x_min,
                            picos + y_min])
    refc = ref["vertebras"].mean(axis=1)
    fig, ax = plt.subplots(figsize=(7, 13))
    ax.imshow(r["imagen"], cmap="gray", vmin=0, vmax=255)
    for a, b in zip(auto, refc):
        ax.plot([a[0], b[0]], [a[1], b[1]], color="yellow", lw=0.8, alpha=0.7)
    ax.scatter(auto[:, 0], auto[:, 1], color="orange", s=30, label="Centro automático")
    ax.scatter(refc[:, 0], refc[:, 1], color="lime", marker="x", s=40,
               label="Centro de referencia")
    ax.set_title(f"Localización de centros\n{titulo}")
    ax.legend(fontsize=8)
    ax.axis("off")
    plt.tight_layout()
    return fig, auto, refc


def dispersion_cobb(comparaciones, titulo=""):
    """Cobb automático vs. referencia y gráfico de Bland-Altman."""
    a = comparaciones["cobb_automatico"].to_numpy(float)
    b = comparaciones["cobb_referencia"].to_numpy(float)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    lim = max(a.max(), b.max()) * 1.05
    for conj, mk in (("Calibración", "s"), ("Validación", "o")):
        m = (comparaciones["conjunto"] == conj).to_numpy()
        axes[0].scatter(b[m], a[m], s=22, alpha=0.7, marker=mk, label=conj)
    axes[0].plot([0, lim], [0, lim], "k--", lw=1, label="Identidad")
    axes[0].fill_between([0, lim], [-5, lim - 5], [5, lim + 5], color="gray",
                         alpha=0.15, label="±5°")
    for u in (10, 25, 40):
        axes[0].axvline(u, color="gray", ls=":", lw=0.8)
        axes[0].axhline(u, color="gray", ls=":", lw=0.8)
    axes[0].set_xlim(0, lim)
    axes[0].set_ylim(0, lim)
    axes[0].set_xlabel("Cobb de referencia [°]")
    axes[0].set_ylabel("Cobb automático [°]")
    axes[0].set_title("Concordancia")
    axes[0].legend(fontsize=8)

    media, dif = (a + b) / 2, a - b
    sesgo, de = dif.mean(), dif.std(ddof=1)
    axes[1].scatter(media, dif, s=18, alpha=0.6)
    for y, t in ((sesgo, f"Sesgo {sesgo:.1f}°"),
                 (sesgo + 1.96 * de, f"+1,96 DE {sesgo + 1.96 * de:.1f}°"),
                 (sesgo - 1.96 * de, f"−1,96 DE {sesgo - 1.96 * de:.1f}°")):
        axes[1].axhline(y, color="tab:red" if "Sesgo" in t else "gray", ls="--", lw=1)
        axes[1].text(media.max(), y, t, ha="right", va="bottom", fontsize=8)
    axes[1].set_xlabel("Media de ambos métodos [°]")
    axes[1].set_ylabel("Automático − referencia [°]")
    axes[1].set_title("Bland-Altman")
    fig.suptitle(titulo)
    plt.tight_layout()
    return fig


def matriz_confusion(matriz, titulo="Gravedad: referencia vs. automática"):
    etiquetas = ["<10°", "Leve", "Moderada", "Severa"]
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(matriz.values, cmap="Blues")
    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):
            v = matriz.values[i, j]
            ax.text(j, i, int(v), ha="center", va="center",
                    color="white" if v > matriz.values.max() / 2 else "black")
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(etiquetas)
    ax.set_yticklabels(etiquetas)
    ax.set_xlabel("Automática")
    ax.set_ylabel("Referencia")
    ax.set_title(titulo)
    fig.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    return fig
