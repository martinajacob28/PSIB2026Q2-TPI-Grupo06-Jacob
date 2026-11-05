"""
cobb_pipeline.py
================

Pipeline clásico (sin aprendizaje automático) para estimar el ángulo de Cobb
en radiografías de columna completa (AP) del dataset AASCE/MICCAI 2019.

Etapas (cada una es una función independiente):

1. ``extraer_roi_hueso``      – ROI de la columna y sustracción de fondo.
2. ``realzar_estructuras``    – CLAHE + Top-Hat sobre la matriz ósea.
3. ``detectar_niveles``       – 17 niveles vertebrales regularizados.
4. ``ajustar_eje``            – centros candidatos y eje polinómico (grado 4).
5. ``construir_cajas``        – cajas orientadas por vértebra (visualización).
6. ``ajustar_platillos``      – ajuste restringido de platillos por gradiente.
7. ``control_calidad``        – confianza de cada platillo y coherencia geométrica.
8. ``seleccionar_curvas``     – perfil angular y vértebras límite.
9. ``calcular_cobbs``         – ángulo de Cobb y gravedad (Hong et al., 2017).

``procesar_radiografia`` encadena todas las etapas. Las anotaciones de
referencia se leen recién en la evaluación (``cargar_referencia``) y nunca
intervienen en el procesamiento automático.
"""

from __future__ import annotations

import io
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, savgol_filter

NUMERO_NIVELES = 17

URL_DATASET = (
    "https://dl.dropboxusercontent.com/scl/fi/"
    "nvfqc9t92i0e4rh4f0gz2/Archive.zip"
    "?rlkey=ohwud8jbkv5joozysg7mym0lt&dl=1"
)

BASE_REFERENCIAS = (
    "https://raw.githubusercontent.com/"
    "sudaoer/AASCE2019-test98-annotations/"
    "08b3c9cd799ac787ec6a537aace6560afcaa6d0e/"
    "dataset/MICCAI_98_Annotations/"
)

ORDEN_GRAVEDAD = [
    "Inferior al umbral de 10°",
    "Leve",
    "Moderada",
    "Severa",
]


# ---------------------------------------------------------------------------
# 0. Preprocesamiento exploratorio
# ---------------------------------------------------------------------------

def preprocesar_imagen(img_gray, ksize=5, clip_limit=2.0, tile_grid=(8, 8)):
    """Filtro Gaussiano 5×5 seguido de CLAHE (etapa exploratoria)."""
    suavizada = cv2.GaussianBlur(img_gray, (ksize, ksize), sigmaX=0)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    return clahe.apply(suavizada)


# ---------------------------------------------------------------------------
# 1. ROI de la columna y matriz ósea
# ---------------------------------------------------------------------------

def extraer_roi_hueso(img_gray):
    """
    Estima la posición horizontal de la columna, recorta una ROI amplia y
    resta el fondo de baja frecuencia (apertura morfológica).

    Devuelve (roi_columna, fondo_macro, matriz_osea, (x_min, y_min, x_max,
    y_max), columna_pico).
    """
    alto, ancho = img_gray.shape

    imagen_mediana = cv2.medianBlur(img_gray, 9)

    x_busqueda_0 = int(0.20 * ancho)
    x_busqueda_1 = int(0.80 * ancho)
    zona_central = imagen_mediana[:, x_busqueda_0:x_busqueda_1]

    suma_columnas = zona_central.sum(axis=0)
    suma_suavizada = gaussian_filter1d(suma_columnas.astype(float), sigma=5)
    columna_pico = x_busqueda_0 + int(np.argmax(suma_suavizada))

    ancho_roi = int(0.45 * ancho)
    x_min = max(0, columna_pico - ancho_roi // 2)
    x_max = min(ancho, columna_pico + ancho_roi // 2)
    y_min = int(0.10 * alto)
    y_max = int(0.90 * alto)

    roi_columna = imagen_mediana[y_min:y_max, x_min:x_max]

    ancho_kernel = max(3, int(0.08 * ancho))
    alto_kernel = max(3, int(0.04 * alto))
    kernel_fondo = cv2.getStructuringElement(
        cv2.MORPH_RECT, (ancho_kernel, alto_kernel)
    )
    fondo_macro = cv2.morphologyEx(roi_columna, cv2.MORPH_OPEN, kernel_fondo)
    matriz_osea = cv2.subtract(roi_columna, fondo_macro)

    return (
        roi_columna,
        fondo_macro,
        matriz_osea,
        (x_min, y_min, x_max, y_max),
        columna_pico,
    )


# ---------------------------------------------------------------------------
# 2. Realce de estructuras óseas
# ---------------------------------------------------------------------------

def realzar_estructuras(solo_hueso):
    """CLAHE (clip 3.0) + Top-Hat elíptico, combinados 0.55 / 0.45."""
    alto_roi, ancho_roi = solo_hueso.shape

    clahe_oseo = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    matriz_clahe = clahe_oseo.apply(solo_hueso)

    ancho_tophat = max(3, ancho_roi // 4)
    alto_tophat = max(3, alto_roi // 14)
    kernel_tophat = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (ancho_tophat, alto_tophat)
    )
    realce_tophat = cv2.morphologyEx(matriz_clahe, cv2.MORPH_TOPHAT, kernel_tophat)

    base_osea = cv2.addWeighted(matriz_clahe, 0.55, realce_tophat, 0.45, 0)

    return matriz_clahe, realce_tophat, base_osea


# ---------------------------------------------------------------------------
# 3. Detección regularizada de 17 niveles
# ---------------------------------------------------------------------------

def detectar_niveles(base_osea, numero_niveles=NUMERO_NIVELES,
                     relacion_separacion=1.70):
    """
    Perfil vertical de intensidad ósea + secuencia anatómica esperada de 17
    niveles cuya separación crece linealmente hacia la región lumbar. Cada
    nivel se desplaza al máximo local más cercano sólo si cae dentro de una
    ventana del 20 % de la separación local.
    """
    alto_base, ancho_base = base_osea.shape

    x_perfil_0 = int(0.20 * ancho_base)
    x_perfil_1 = int(0.80 * ancho_base)
    perfil_vertebral = base_osea[:, x_perfil_0:x_perfil_1].mean(axis=1)

    perfil_suave = gaussian_filter1d(
        perfil_vertebral, sigma=max(3, 0.003 * alto_base)
    )
    tendencia_perfil = gaussian_filter1d(
        perfil_suave, sigma=max(5, 0.035 * alto_base)
    )
    perfil_corregido = perfil_suave - tendencia_perfil

    fila_valida_0 = int(0.13 * alto_base)
    fila_valida_1 = int(0.88 * alto_base)
    zona_valida = perfil_corregido[fila_valida_0:fila_valida_1]

    variabilidad = max(
        1e-6,
        1.4826 * np.median(np.abs(zona_valida - np.median(zona_valida))),
    )
    picos_locales, propiedades = find_peaks(
        zona_valida,
        distance=max(8, int(0.010 * alto_base)),
        prominence=0.08 * variabilidad,
    )
    picos_disponibles = fila_valida_0 + picos_locales
    prominencias = propiedades["prominences"]

    fila_inicial = 0.16 * alto_base
    fila_final = 0.852 * alto_base
    primera_separacion = (
        2 * (fila_final - fila_inicial)
        / ((numero_niveles - 1) * (1 + relacion_separacion))
    )
    separaciones_esperadas = np.linspace(
        primera_separacion,
        relacion_separacion * primera_separacion,
        numero_niveles - 1,
    )
    niveles_esperados = np.concatenate([
        [fila_inicial],
        fila_inicial + np.cumsum(separaciones_esperadas),
    ])

    niveles_detectados = []
    ajustado_por_perfil = []

    for i, nivel_esperado in enumerate(niveles_esperados):
        if i == 0:
            separacion_local = separaciones_esperadas[0]
        elif i == numero_niveles - 1:
            separacion_local = separaciones_esperadas[-1]
        else:
            separacion_local = np.mean([
                separaciones_esperadas[i - 1],
                separaciones_esperadas[i],
            ])

        radio_ajuste = 0.20 * separacion_local
        indices_cercanos = np.where(
            np.abs(picos_disponibles - nivel_esperado) <= radio_ajuste
        )[0]

        if len(indices_cercanos):
            distancias = np.abs(picos_disponibles[indices_cercanos] - nivel_esperado)
            prominencias_locales = prominencias[indices_cercanos]
            prominencias_normalizadas = (
                prominencias_locales / (prominencias_locales.max() + 1e-9)
            )
            puntuacion = (
                -distancias / radio_ajuste + 0.15 * prominencias_normalizadas
            )
            mejor_indice = indices_cercanos[np.argmax(puntuacion)]
            nivel_final = float(picos_disponibles[mejor_indice])
            ajustado = True
        else:
            nivel_final = float(nivel_esperado)
            ajustado = False

        niveles_detectados.append(nivel_final)
        ajustado_por_perfil.append(ajustado)

    return {
        "picos_vertebrales": np.rint(niveles_detectados).astype(int),
        "ajustado_por_perfil": np.asarray(ajustado_por_perfil, dtype=bool),
        "perfil_corregido": perfil_corregido,
        "banda_perfil": (x_perfil_0, x_perfil_1),
        "filas_validas": (fila_valida_0, fila_valida_1),
    }


# ---------------------------------------------------------------------------
# 4. Centros candidatos y eje polinómico
# ---------------------------------------------------------------------------

def ajustar_eje(base_osea, picos_vertebrales, grado=4):
    """
    Busca un centro horizontal por nivel (intensidad alta penalizando saltos
    respecto del nivel anterior), ajusta x = p(y) de grado 4, descarta
    candidatos con residuo > 2.5·MAD y vuelve a ajustar.
    """
    alto_base, ancho_base = base_osea.shape
    centros_candidatos = []
    x_anterior = ancho_base / 2

    if len(picos_vertebrales) > 1:
        separacion_mediana = np.median(np.diff(picos_vertebrales))
    else:
        separacion_mediana = 0.05 * alto_base

    for fila in picos_vertebrales:
        medio_alto = max(10, int(0.25 * separacion_mediana))
        y0_local = max(0, int(fila - medio_alto))
        y1_local = min(alto_base, int(fila + medio_alto))

        radio_horizontal = int(0.15 * ancho_base)
        x0_local = max(0, int(x_anterior - radio_horizontal))
        x1_local = min(ancho_base, int(x_anterior + radio_horizontal))

        recorte_local = base_osea[y0_local:y1_local, x0_local:x1_local]
        perfil_horizontal = gaussian_filter1d(
            recorte_local.mean(axis=0), sigma=max(1, 0.012 * ancho_base)
        )
        coordenadas_x = np.arange(x0_local, x1_local)
        amplitud = max(np.ptp(perfil_horizontal), 1e-6)
        distancia_previa = (
            np.abs(coordenadas_x - x_anterior) / max(radio_horizontal, 1)
        )
        puntuacion = perfil_horizontal - 0.55 * amplitud * distancia_previa

        x_candidato = float(coordenadas_x[np.argmax(puntuacion)])
        centros_candidatos.append([x_candidato, float(fila)])
        x_anterior = x_candidato

    centros_candidatos = np.asarray(centros_candidatos, dtype=float)
    if len(centros_candidatos) < 5:
        raise ValueError("No hay suficientes candidatos para ajustar el eje.")

    y_centros = centros_candidatos[:, 1]
    x_centros = centros_candidatos[:, 0]

    coeficientes_eje = np.polyfit(y_centros, x_centros, deg=grado)
    residuos = x_centros - np.polyval(coeficientes_eje, y_centros)
    mediana_residuos = np.median(residuos)
    mad_residuos = max(
        1.0, 1.4826 * np.median(np.abs(residuos - mediana_residuos))
    )
    candidatos_validos = np.abs(residuos - mediana_residuos) <= 2.5 * mad_residuos

    if candidatos_validos.sum() >= 5:
        coeficientes_eje = np.polyfit(
            y_centros[candidatos_validos],
            x_centros[candidatos_validos],
            deg=grado,
        )

    y_eje = np.arange(int(y_centros.min()), int(y_centros.max()) + 1)
    x_eje = np.polyval(coeficientes_eje, y_eje)

    return {
        "coeficientes_eje": coeficientes_eje,
        "centros_candidatos": centros_candidatos,
        "candidatos_validos": candidatos_validos,
        "x_eje": x_eje,
        "y_eje": y_eje,
    }


# ---------------------------------------------------------------------------
# 5. Cajas vertebrales orientadas (apoyo visual)
# ---------------------------------------------------------------------------

def construir_cajas(base_osea, picos_vertebrales, perfil_corregido,
                    coeficientes_eje, origen_roi=(0, 0)):
    """Cajas de altura uniforme (mediana) y ancho 35 % de la ROI,
    orientadas según la pendiente local del eje."""
    alto_base, ancho_base = base_osea.shape
    x_min, y_min = origen_roi

    if len(picos_vertebrales) < 2:
        raise ValueError("Se necesitan al menos dos candidatos.")

    fronteras = [max(0, int(picos_vertebrales[0]
                            - (picos_vertebrales[1] - picos_vertebrales[0]) / 2))]
    for i in range(len(picos_vertebrales) - 1):
        inicio = int(picos_vertebrales[i])
        fin = int(picos_vertebrales[i + 1])
        fronteras.append(inicio + int(np.argmin(perfil_corregido[inicio:fin])))
    fronteras.append(min(alto_base - 1, int(
        picos_vertebrales[-1] + (picos_vertebrales[-1] - picos_vertebrales[-2]) / 2
    )))
    fronteras = np.asarray(fronteras, dtype=int)

    alto_uniforme = float(np.median(np.diff(fronteras)))
    ancho_uniforme = float(0.35 * ancho_base)

    cajas = []
    for i in range(len(picos_vertebrales)):
        sup, inf = int(fronteras[i]), int(fronteras[i + 1])
        centro_y = (sup + inf) / 2
        centro_x = float(np.polyval(coeficientes_eje, centro_y))

        y_s = max(0, centro_y - alto_uniforme / 2)
        y_i = min(alto_base - 1, centro_y + alto_uniforme / 2)
        dx = float(np.polyval(coeficientes_eje, y_i)) - float(
            np.polyval(coeficientes_eje, y_s))
        dy = y_i - y_s
        angulo_caja = np.degrees(np.arctan2(-dx, dy))
        a = np.radians(angulo_caja)

        h = np.array([np.cos(a), np.sin(a)])
        v = np.array([-np.sin(a), np.cos(a)])
        c = np.array([centro_x, centro_y])
        ma, mh = ancho_uniforme / 2, alto_uniforme / 2
        esquinas_roi = np.array([
            c - ma * h - mh * v,
            c + ma * h - mh * v,
            c + ma * h + mh * v,
            c - ma * h + mh * v,
        ])
        cajas.append({
            "candidato": i + 1,
            "centro_roi": c,
            "esquinas_roi": esquinas_roi,
            "esquinas_globales": esquinas_roi + np.array([x_min, y_min]),
            "angulo_inicial": float(angulo_caja),
            "ancho": ancho_uniforme,
            "alto": alto_uniforme,
            "banda": (sup, inf),
        })
    return cajas


# ---------------------------------------------------------------------------
# 6. Ajuste restringido de platillos
# ---------------------------------------------------------------------------

class MapaGradiente:
    """Gradientes Sobel de la imagen base y respuesta de borde sobre rectas."""

    def __init__(self, base_osea):
        imagen_ajuste = cv2.normalize(
            base_osea, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)
        imagen_suave = cv2.GaussianBlur(imagen_ajuste, (5, 5), 0)
        self.gx = cv2.Sobel(imagen_suave.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        self.gy = cv2.Sobel(imagen_suave.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        self.alto, self.ancho = imagen_ajuste.shape

    def respuesta_recta(self, x_centro, y_recta, angulo, semiancho):
        """Media (recortada al p90) de la componente del gradiente
        perpendicular a la recta candidata."""
        x = np.linspace(x_centro - semiancho, x_centro + semiancho, 100)
        y = y_recta + np.tan(np.radians(angulo)) * (x - x_centro)

        validos = (
            (x >= 0) & (x < self.ancho - 1) & (y >= 0) & (y < self.alto - 1)
        )
        x = x[validos].astype(np.float32)
        y = y[validos].astype(np.float32)
        if len(x) < 30:
            return 0.0

        mapa_x, mapa_y = x.reshape(1, -1), y.reshape(1, -1)
        gx = cv2.remap(self.gx, mapa_x, mapa_y, cv2.INTER_LINEAR).ravel()
        gy = cv2.remap(self.gy, mapa_x, mapa_y, cv2.INTER_LINEAR).ravel()

        theta = np.radians(angulo)
        respuesta = np.abs(-np.sin(theta) * gx + np.cos(theta) * gy)
        respuesta = np.clip(respuesta, 0, np.percentile(respuesta, 90))
        return float(np.mean(respuesta))

    def ajustar_platillo(self, x_centro, y_esperado, angulo_inicial,
                         separacion, semiancho, peso_angulo):
        """Búsqueda en grilla (13 posiciones × 25 ángulos, ±10 % de la
        separación y ±12°) con penalización de desplazamiento y giro."""
        posiciones = np.linspace(
            y_esperado - 0.10 * separacion, y_esperado + 0.10 * separacion, 13
        )
        angulos = np.linspace(angulo_inicial - 12, angulo_inicial + 12, 25)

        candidatos = np.asarray([
            [self.respuesta_recta(x_centro, yc, ang, semiancho), yc, ang]
            for yc in posiciones for ang in angulos
        ], dtype=float)

        respuestas = candidatos[:, 0]
        normalizadas = (respuestas - respuestas.min()) / (np.ptp(respuestas) + 1e-8)
        pen_y = np.abs(candidatos[:, 1] - y_esperado) / (0.10 * separacion)
        pen_ang = np.abs(candidatos[:, 2] - angulo_inicial) / 12
        puntuacion = normalizadas - 0.22 * pen_y - peso_angulo * pen_ang

        mejor = int(np.argmax(puntuacion))
        y_final, angulo_final = candidatos[mejor, 1], candidatos[mejor, 2]

        x_linea = np.array([x_centro - semiancho, x_centro + semiancho])
        y_linea = y_final + np.tan(np.radians(angulo_final)) * (x_linea - x_centro)
        return np.column_stack([x_linea, y_linea]), float(angulo_final)


def separacion_local(picos_vertebrales, i):
    """Separación vertical característica alrededor del nivel i."""
    if i == 0:
        return picos_vertebrales[1] - picos_vertebrales[0]
    if i == len(picos_vertebrales) - 1:
        return picos_vertebrales[-1] - picos_vertebrales[-2]
    return (picos_vertebrales[i + 1] - picos_vertebrales[i - 1]) / 2


def ajustar_platillos(mapa, picos_vertebrales, coeficientes_eje):
    """Platillo superior e inferior de cada nivel (coordenadas de la ROI).
    Los ángulos están en convención de imagen (y hacia abajo)."""
    n = len(picos_vertebrales)
    sup_rectas, inf_rectas, sup_ang, inf_ang = [], [], [], []

    for i, y_centro in enumerate(picos_vertebrales):
        separacion = separacion_local(picos_vertebrales, i)
        x_centro = float(np.polyval(coeficientes_eje, y_centro))

        # Inclinación local a partir de los niveles vecinos.
        if i == 0:
            i1, i2 = 0, min(2, n - 1)
        elif i == n - 1:
            i1, i2 = max(0, n - 3), n - 1
        else:
            i1, i2 = i - 1, i + 1
        y1, y2 = float(picos_vertebrales[i1]), float(picos_vertebrales[i2])
        x1 = float(np.polyval(coeficientes_eje, y1))
        x2 = float(np.polyval(coeficientes_eje, y2))
        dx_dy = (x2 - x1) / (y2 - y1 + 1e-8)

        # El platillo se espera perpendicular al eje local.
        angulo_inicial = float(np.clip(np.degrees(np.arctan(-dx_dy)), -25, 25))

        # Más restricción en la región superior (costillas y clavículas).
        peso_angulo = 0.25 if i < 7 else 0.12

        semiancho = min(0.15 * mapa.ancho, 0.70 * separacion)

        superior, a_sup = mapa.ajustar_platillo(
            x_centro, y_centro - 0.23 * separacion, angulo_inicial,
            separacion, semiancho, peso_angulo)
        inferior, a_inf = mapa.ajustar_platillo(
            x_centro, y_centro + 0.23 * separacion, angulo_inicial,
            separacion, semiancho, peso_angulo)

        sup_rectas.append(superior)
        inf_rectas.append(inferior)
        sup_ang.append(a_sup)
        inf_ang.append(a_inf)

    return {
        "platillos_superiores": sup_rectas,
        "platillos_inferiores": inf_rectas,
        "angulos_superiores": np.asarray(sup_ang, dtype=float),
        "angulos_inferiores": np.asarray(inf_ang, dtype=float),
    }


# ---------------------------------------------------------------------------
# 7. Control de calidad
# ---------------------------------------------------------------------------

def confianza_de_recta(mapa, recta, angulo, x_centro, separacion):
    """Puntaje robusto (mediana/MAD) de la recta frente a 16 paralelas
    vecinas desplazadas hasta ±18 % de la separación."""
    y_recta = np.mean(recta[:, 1])
    semiancho = abs(recta[1, 0] - recta[0, 0]) / 2
    respuesta_detectada = mapa.respuesta_recta(x_centro, y_recta, angulo, semiancho)

    vecinas = []
    for d in np.linspace(-0.18 * separacion, 0.18 * separacion, 19):
        if abs(d) < 0.035 * separacion:
            continue
        vecinas.append(mapa.respuesta_recta(x_centro, y_recta + d, angulo, semiancho))
    vecinas = np.asarray(vecinas)

    mediana = np.median(vecinas)
    mad = np.median(np.abs(vecinas - mediana))
    return float((respuesta_detectada - mediana) / (1.4826 * mad + 1e-8))


def control_calidad(mapa, picos_vertebrales, coeficientes_eje, platillos):
    """Clasifica cada nivel en calidad Alta / Media / Baja."""
    filas, validos = [], []
    for i in range(len(picos_vertebrales)):
        sep = separacion_local(picos_vertebrales, i)
        x_centro = float(np.polyval(coeficientes_eje, picos_vertebrales[i]))
        sup = platillos["platillos_superiores"][i]
        inf = platillos["platillos_inferiores"][i]
        a_sup = platillos["angulos_superiores"][i]
        a_inf = platillos["angulos_inferiores"][i]

        c_sup = confianza_de_recta(mapa, sup, a_sup, x_centro, sep)
        c_inf = confianza_de_recta(mapa, inf, a_inf, x_centro, sep)

        altura_relativa = (np.mean(inf[:, 1]) - np.mean(sup[:, 1])) / sep
        diferencia_angular = abs(a_sup - a_inf)
        c_min = min(c_sup, c_inf)
        geometria_valida = (0.28 <= altura_relativa <= 0.70
                            and diferencia_angular <= 16)

        if c_min >= 1.0 and geometria_valida:
            calidad, valido = "Alta", True
        elif c_min >= 0.0 and geometria_valida:
            calidad, valido = "Media", True
        else:
            calidad, valido = "Baja", False

        validos.append(valido)
        filas.append({
            "Nivel": f"C{i + 1}",
            "Confianza superior": c_sup,
            "Confianza inferior": c_inf,
            "Altura relativa": altura_relativa,
            "Diferencia angular (°)": diferencia_angular,
            "Calidad": calidad,
        })
    return pd.DataFrame(filas), np.asarray(validos, dtype=bool)


# ---------------------------------------------------------------------------
# 8. Perfil angular y vértebras límite
# ---------------------------------------------------------------------------

def angulo_representativo(angulos_superiores, angulos_inferiores):
    """Media circular (orientaciones de rectas, período 180°)."""
    res = []
    for s, i in zip(angulos_superiores, angulos_inferiores):
        r = np.radians([s, i])
        res.append(0.5 * np.degrees(np.arctan2(np.mean(np.sin(2 * r)),
                                               np.mean(np.cos(2 * r)))))
    return np.asarray(res, dtype=float)


def _suavizar_y_extremos(angulos):
    n = len(angulos)
    if n >= 5:
        suav = savgol_filter(angulos, window_length=5, polyorder=2, mode="interp")
    else:
        suav = angulos.copy()
    prominencia = max(1.5, 0.12 * np.ptp(suav))
    maximos, _ = find_peaks(suav, prominence=prominencia, distance=2)
    minimos, _ = find_peaks(-suav, prominence=prominencia, distance=2)
    extremos = np.unique(np.concatenate([[0], maximos, minimos, [n - 1]]))
    return suav, extremos


def seleccionar_curvas(angulos_superiores, angulos_inferiores, validos,
                       cambio_angular_minimo=5.0):
    """
    Vértebras límite del resultado automático: los niveles rechazados por el
    control de calidad se interpolan; los límites se trasladan al nivel
    aceptado más próximo.
    """
    angulos = angulo_representativo(angulos_superiores, angulos_inferiores)
    validos = np.asarray(validos, dtype=bool).copy()
    n = len(angulos)

    if validos.sum() < 4:
        validos[:] = True

    indices = np.arange(n)
    indices_validos = indices[validos]
    interp = np.interp(indices, indices_validos, angulos[validos])
    suav, extremos = _suavizar_y_extremos(interp)

    def mas_cercano(k):
        return int(indices_validos[np.argmin(np.abs(indices_validos - k))])

    curvas = []
    for e1, e2 in zip(extremos[:-1], extremos[1:]):
        a1, a2 = suav[e1], suav[e2]
        if a1 * a2 <= 0 and abs(a2 - a1) >= cambio_angular_minimo:
            s, i = mas_cercano(e1), mas_cercano(e2)
            if s > i:
                s, i = i, s
            if s != i and (s, i) not in curvas:
                curvas.append((s, i))

    if not curvas:
        imax = mas_cercano(int(np.argmax(suav)))
        imin = mas_cercano(int(np.argmin(suav)))
        s, i = min(imax, imin), max(imax, imin)
        if s != i:
            curvas.append((s, i))

    return {
        "curvas": curvas,
        "angulos_vertebrales": angulos,
        "angulos_suavizados": suav,
        "validos_usados": validos,
    }


def seleccionar_curvas_desde_angulos(angulos_superiores, angulos_inferiores,
                                     cambio_angular_minimo=5.0):
    """Mismo criterio de extremos, aplicado a la referencia (sin control de
    calidad ni interpolación)."""
    angulos = angulo_representativo(angulos_superiores, angulos_inferiores)
    suav, extremos = _suavizar_y_extremos(angulos)

    curvas = []
    for e1, e2 in zip(extremos[:-1], extremos[1:]):
        a1, a2 = suav[e1], suav[e2]
        if a1 * a2 <= 0 and abs(a2 - a1) >= cambio_angular_minimo:
            curvas.append((int(e1), int(e2)))

    if not curvas:
        imax, imin = int(np.argmax(suav)), int(np.argmin(suav))
        s, i = min(imax, imin), max(imax, imin)
        if s != i:
            curvas.append((s, i))
    return curvas


# ---------------------------------------------------------------------------
# 9. Ángulo de Cobb y gravedad
# ---------------------------------------------------------------------------

def diferencia_aguda(angulo_1, angulo_2):
    """Menor ángulo entre dos rectas: min(|θs−θi|, 180°−|θs−θi|)."""
    d = abs(float(angulo_1) - float(angulo_2)) % 180
    return min(d, 180 - d)


def clasificar_gravedad(cobb):
    """Umbrales de Hong et al. (2017)."""
    if not np.isfinite(cobb):
        return "No disponible"
    if cobb < 10:
        return "Inferior al umbral de 10°"
    if cobb < 25:
        return "Leve"
    if cobb < 40:
        return "Moderada"
    return "Severa"


def calcular_cobbs(curvas, angulos_superiores, angulos_inferiores):
    """Cobb = platillo superior de la vértebra límite superior vs.
    platillo inferior de la vértebra límite inferior."""
    resultados = []
    for numero, (s, i) in enumerate(curvas, start=1):
        cobb = diferencia_aguda(angulos_superiores[s], angulos_inferiores[i])
        resultados.append({
            "curva": numero,
            "limite_superior": s + 1,
            "limite_inferior": i + 1,
            "angulo_superior": float(angulos_superiores[s]),
            "angulo_inferior": float(angulos_inferiores[i]),
            "cobb": cobb,
            "gravedad": clasificar_gravedad(cobb),
        })
    return resultados


# ---------------------------------------------------------------------------
# Pipeline completo
# ---------------------------------------------------------------------------

def procesar_radiografia(img_gray, con_cajas=False):
    """Ejecuta todas las etapas sobre una radiografía en escala de grises y
    devuelve un diccionario con los resultados intermedios y finales."""
    roi, fondo, matriz_osea, roi_coords, columna_pico = extraer_roi_hueso(img_gray)
    matriz_clahe, tophat, base_osea = realzar_estructuras(matriz_osea)

    niveles = detectar_niveles(base_osea)
    picos = niveles["picos_vertebrales"]
    eje = ajustar_eje(base_osea, picos)

    mapa = MapaGradiente(base_osea)
    platillos = ajustar_platillos(mapa, picos, eje["coeficientes_eje"])
    tabla_calidad, validos = control_calidad(
        mapa, picos, eje["coeficientes_eje"], platillos)

    seleccion = seleccionar_curvas(
        platillos["angulos_superiores"], platillos["angulos_inferiores"], validos)
    resultados = calcular_cobbs(
        seleccion["curvas"],
        platillos["angulos_superiores"],
        platillos["angulos_inferiores"])

    salida = {
        "imagen": img_gray,
        "roi": roi,
        "fondo": fondo,
        "matriz_osea": matriz_osea,
        "matriz_clahe": matriz_clahe,
        "tophat": tophat,
        "base_osea": base_osea,
        "roi_coords": roi_coords,
        "columna_pico": columna_pico,
        **niveles,
        **eje,
        **platillos,
        "tabla_calidad": tabla_calidad,
        "platillos_validos": validos,
        **seleccion,
        "resultados": resultados,
    }
    if con_cajas:
        salida["cajas"] = construir_cajas(
            base_osea, picos, niveles["perfil_corregido"],
            eje["coeficientes_eje"], roi_coords[:2])
    return salida


def curva_principal(resultados):
    """Curva con mayor ángulo de Cobb (o None)."""
    if not resultados:
        return None
    return max(resultados, key=lambda r: r["cobb"])


# ---------------------------------------------------------------------------
# Referencia (sólo para evaluar)
# ---------------------------------------------------------------------------

def cargar_esquinas_referencia(nombre_imagen, carpeta_local=None):
    """68 esquinas (17 vértebras × 4) en coordenadas de la imagen original.
    Se leen de ``carpeta_local`` si existe; si no, del repositorio público."""
    stem = Path(nombre_imagen).stem
    if carpeta_local is not None:
        ruta = Path(carpeta_local) / f"{stem}_corner.pts"
        if ruta.exists():
            esquinas = np.loadtxt(ruta)
            return esquinas.reshape(17, 4, 2)
    with urllib.request.urlopen(BASE_REFERENCIAS + stem + "_corner.pts",
                                timeout=30) as r:
        esquinas = np.loadtxt(io.StringIO(r.read().decode("utf-8")))
    if esquinas.shape != (68, 2):
        raise ValueError("La referencia no contiene los 68 puntos esperados.")
    return esquinas.reshape(17, 4, 2)


def angulos_referencia(vertebras):
    """Inclinación de platillos de referencia en convención de imagen."""
    vs = vertebras[:, 1] - vertebras[:, 0]
    vi = vertebras[:, 3] - vertebras[:, 2]
    return (np.degrees(np.arctan2(vs[:, 1], vs[:, 0])),
            np.degrees(np.arctan2(vi[:, 1], vi[:, 0])))


def obtener_referencia(nombre_imagen, carpeta_local=None):
    vertebras = cargar_esquinas_referencia(nombre_imagen, carpeta_local)
    a_sup, a_inf = angulos_referencia(vertebras)
    curvas = seleccionar_curvas_desde_angulos(a_sup, a_inf)
    return {
        "vertebras": vertebras,
        "angulos_superiores": a_sup,
        "angulos_inferiores": a_inf,
        "curvas": curvas,
        "resultados": calcular_cobbs(curvas, a_sup, a_inf),
    }


# ---------------------------------------------------------------------------
# Emparejamiento de curvas
# ---------------------------------------------------------------------------

def iou_intervalos(par_1, par_2):
    """IoU entre dos intervalos de niveles vertebrales (inclusive)."""
    (a1, b1), (a2, b2) = par_1, par_2
    inter = max(0, min(b1, b2) - max(a1, a2) + 1)
    union = max(b1, b2) - min(a1, a2) + 1
    return inter / union


def emparejar_curvas(curvas_automaticas, curvas_referencia):
    """Emparejamiento voraz por IoU decreciente (IoU > 0)."""
    candidatos = [
        (iou_intervalos(ca, cr), ia, ir)
        for ia, ca in enumerate(curvas_automaticas)
        for ir, cr in enumerate(curvas_referencia)
    ]
    usados_auto, usados_ref, parejas = set(), set(), []
    for sup, ia, ir in sorted(candidatos, reverse=True):
        if sup <= 0 or ia in usados_auto or ir in usados_ref:
            continue
        parejas.append((ia, ir, sup))
        usados_auto.add(ia)
        usados_ref.add(ir)
    return parejas, usados_auto, usados_ref


# ---------------------------------------------------------------------------
# Evaluación sobre un conjunto de imágenes
# ---------------------------------------------------------------------------

def evaluar_imagen(ruta_imagen, carpeta_referencias=None):
    """Procesa una radiografía (sin referencia) y luego la compara."""
    img = cv2.imread(str(ruta_imagen), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("No se pudo leer la radiografía.")

    auto = procesar_radiografia(img)
    ref = obtener_referencia(Path(ruta_imagen).name, carpeta_referencias)
    parejas, usados_auto, usados_ref = emparejar_curvas(auto["curvas"], ref["curvas"])

    nombre = Path(ruta_imagen).name
    resumen = {
        "archivo": nombre,
        "curvas_automaticas": len(auto["curvas"]),
        "curvas_referencia": len(ref["curvas"]),
        "niveles_confirmados": int(auto["ajustado_por_perfil"].sum()),
        "platillos_validos": int(auto["platillos_validos"].sum()),
        "curvas_emparejadas": len(parejas),
        "estado": "OK",
    }

    filas = []
    for ia, ir, sup in parejas:
        ra, rr = auto["resultados"][ia], ref["resultados"][ir]
        filas.append({
            "archivo": nombre,
            "curva_automatica": ia + 1,
            "limite_superior_auto": ra["limite_superior"],
            "limite_inferior_auto": ra["limite_inferior"],
            "cobb_automatico": ra["cobb"],
            "gravedad_automatica": ra["gravedad"],
            "curva_referencia": ir + 1,
            "limite_superior_ref": rr["limite_superior"],
            "limite_inferior_ref": rr["limite_inferior"],
            "cobb_referencia": rr["cobb"],
            "gravedad_referencia": rr["gravedad"],
            "error_absoluto": abs(ra["cobb"] - rr["cobb"]),
            "superposicion_limites": sup,
            "coincide_gravedad": ra["gravedad"] == rr["gravedad"],
            "estado_curva": "Emparejada",
        })
    for ia, ra in enumerate(auto["resultados"]):
        if ia not in usados_auto:
            filas.append({
                "archivo": nombre,
                "curva_automatica": ia + 1,
                "limite_superior_auto": ra["limite_superior"],
                "limite_inferior_auto": ra["limite_inferior"],
                "cobb_automatico": ra["cobb"],
                "gravedad_automatica": ra["gravedad"],
                "estado_curva": "Automática sin referencia emparejada",
            })
    for ir, rr in enumerate(ref["resultados"]):
        if ir not in usados_ref:
            filas.append({
                "archivo": nombre,
                "curva_referencia": ir + 1,
                "limite_superior_ref": rr["limite_superior"],
                "limite_inferior_ref": rr["limite_inferior"],
                "cobb_referencia": rr["cobb"],
                "gravedad_referencia": rr["gravedad"],
                "estado_curva": "Referencia no detectada",
            })

    # Error de localización de centros (px) frente a la referencia.
    x_min, y_min = auto["roi_coords"][:2]
    picos = auto["picos_vertebrales"]
    centros_auto = np.column_stack([
        np.polyval(auto["coeficientes_eje"], picos) + x_min, picos + y_min])
    centros_ref = ref["vertebras"].mean(axis=1)
    err = centros_auto - centros_ref
    resumen["error_centro_x_px"] = float(np.mean(np.abs(err[:, 0])))
    resumen["error_centro_y_px"] = float(np.mean(np.abs(err[:, 1])))
    resumen["error_centro_px"] = float(np.mean(np.hypot(err[:, 0], err[:, 1])))
    resumen["alto_px"], resumen["ancho_px"] = img.shape

    return resumen, filas


def metricas(comparaciones):
    """MAE, mediana, % dentro de ±5° y concordancia de gravedad."""
    e = comparaciones["error_absoluto"]
    return {
        "Curvas comparadas": len(comparaciones),
        "MAE Cobb (°)": e.mean(),
        "Mediana error (°)": e.median(),
        "Dentro de ±5° (%)": e.le(5).mean() * 100,
        "Gravedad correcta (%)": comparaciones["coincide_gravedad"].mean() * 100,
    }


def separar_calibracion(nombres, n_calibracion=20, semilla=2026):
    """Partición reproducible calibración / validación."""
    nombres = sorted(nombres)
    rng = np.random.default_rng(semilla)
    elegidas = rng.choice(len(nombres), size=n_calibracion, replace=False)
    return sorted(nombres[k] for k in elegidas)
