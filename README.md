# PSIB2026Q2-TPI-Grupo6-Jacob

**Caracterización automática de la escoliosis mediante análisis geométrico de radiografías de columna**

Trabajo Práctico de Imágenes · 16.63 Procesamiento de Señales e Imágenes Biomédicas (ITBA) · 2026 Q2
Grupo 06 · Martina Jacob (62649)

Pipeline en Python que estima el **ángulo de Cobb** en radiografías AP de columna completa y clasifica la gravedad de cada curva, usando únicamente procesamiento clásico de imágenes (sin aprendizaje automático). Se evalúa sobre las 98 radiografías de prueba del desafío AASCE/MICCAI 2019.

| Gravedad (Hong et al., 2017) | Ángulo de Cobb |
|---|---|
| Inferior al umbral | < 10° |
| Leve | 10° ≤ Cobb < 25° |
| Moderada | 25° ≤ Cobb < 40° |
| Severa | ≥ 40° |

## Contenido del repositorio

| Archivo | Contenido |
|---|---|
| `PSIB2026Q2_TPI_Grupo06_Jacob.ipynb` | Notebook principal: exploración del dataset, pipeline paso a paso sobre una radiografía, interfaz interactiva, evaluación de las 98 imágenes y conclusiones. |
| `cobb_pipeline.py` | Todas las etapas del procesamiento como funciones (ROI, realce, niveles vertebrales, eje, platillos, control de calidad, vértebras límite, Cobb), la lectura de las anotaciones de referencia, el emparejamiento de curvas y las métricas. |
| `visualizacion.py` | Funciones que generan las figuras de cada etapa y del análisis de resultados. |
| `resultados/` | CSV generados por el notebook (métricas, resultados por curva y por imagen, matriz de confusión). |
| `exploratorio/` | Versión de trabajo original del notebook (Colab), previa a la reorganización. |
| `requirements.txt` | Dependencias. |

## Instalación

Requiere Python ≥ 3.10.

```bash
git clone https://github.com/martinajacob28/PSIB2026Q2-TPI-Grupo06-Jacob.git
cd PSIB2026Q2-TPI-Grupo06-Jacob
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Ejecución

### Google Colab (recomendado)

1. Abrir `PSIB2026Q2_TPI_Grupo06_Jacob.ipynb` en Colab (*Archivo → Abrir notebook → GitHub* y pegar la URL del repositorio).
2. *Entorno de ejecución → Ejecutar todas*. La primera celda clona el repositorio para tener los módulos `.py`, y la sección 2 descarga y descomprime el dataset (≈ 80 MB) automáticamente.

### Local

```bash
jupyter notebook PSIB2026Q2_TPI_Grupo06_Jacob.ipynb
```

El notebook descarga el dataset en `data/`. Si ya se tiene el `Archive.zip` del desafío, basta con dejarlo en la raíz del repositorio, o indicar otra carpeta con la variable de entorno `AASCE_DATA`.

Las anotaciones de referencia (68 esquinas por imagen, en coordenadas de la imagen original) se leen del repositorio público [`sudaoer/AASCE2019-test98-annotations`](https://github.com/sudaoer/AASCE2019-test98-annotations). Para trabajar sin conexión, copiar `dataset/MICCAI_98_Annotations/*_corner.pts` a una carpeta `referencias/` en la raíz.

### Uso del módulo sin notebook

```python
import cv2
import cobb_pipeline as cp

img = cv2.imread("data/01-July-2019-1.jpg", cv2.IMREAD_GRAYSCALE)
r = cp.procesar_radiografia(img)
for curva in r["resultados"]:
    print(curva["limite_superior"], curva["limite_inferior"],
          round(curva["cobb"], 1), curva["gravedad"])
```

## Interfaz

La sección 4 del notebook tiene una interfaz con `ipywidgets`: se elige una radiografía en el menú desplegable, se marca si se quiere superponer la referencia y se presiona **Analizar**. Se muestran los platillos límite detectados, el ángulo de Cobb y la gravedad de cada curva, el perfil de inclinaciones vertebrales y la tabla comparativa con la referencia.

## Pipeline

1. **ROI y matriz ósea:** mediana 9×9, localización horizontal por suma de columnas, apertura morfológica y sustracción de fondo.
2. **Realce:** CLAHE (clip 3,0) + Top-Hat elíptico, combinados 0,55 / 0,45.
3. **Niveles vertebrales:** perfil vertical sin tendencia y secuencia regularizada de 17 niveles (separación creciente hacia la región lumbar).
4. **Eje de la columna:** centros por perfil horizontal local y polinomio de grado 4 con rechazo de *outliers* (2,5·MAD).
5. **Platillos:** ajuste restringido de rectas sobre el gradiente de Sobel (13 posiciones × 25 inclinaciones) con penalización de desplazamiento y giro.
6. **Control de calidad:** confianza robusta frente a rectas paralelas vecinas y coherencia geométrica.
7. **Vértebras límite:** perfil angular (media circular + Savitzky-Golay) y extremos consecutivos de signo opuesto.
8. **Cobb y gravedad:** menor ángulo entre el platillo superior de la vértebra límite superior y el inferior de la vértebra límite inferior.

## Historial del desarrollo

El trabajo se desarrolló inicialmente en Google Colab, de forma iterativa, en un único notebook. En él se exploraron el preprocesamiento (Gaussiano, CLAHE), la localización de la columna, la detección de niveles vertebrales, el ajuste de platillos (primero con la transformada de Hough, descartada por las falsas detecciones en las costillas) y el cálculo del ángulo de Cobb. La última versión de esa etapa se conserva en `exploratorio/PSIB2026Q2_TPI_Jacob_v14_exploratorio.ipynb`.

Para la entrega, ese código se reorganizó en el repositorio:

1. el pipeline se extrajo a funciones en `cobb_pipeline.py`, verificando que reprodujera exactamente los resultados del notebook de Colab;
2. las figuras se separaron en `visualizacion.py`;
3. el notebook principal se reordenó, se le agregó la interfaz interactiva y se amplió el análisis de errores;
4. se volvió a ejecutar la evaluación completa sobre las 98 radiografías (`resultados/`).

Por eso la mayor parte del historial de commits corresponde a esta etapa final de organización. El proceso exploratorio previo está documentado en el notebook de `exploratorio/`.

## Referencias

- Hong, A. et al. (2017). Surface Topography Classification Trees for Assessing Severity and Monitoring Progression in Adolescent Idiopathic Scoliosis. *Spine*, 42(13), E781–E787.
- AASCE 2019 – Accurate Automated Spinal Curvature Estimation. https://aasce19.grand-challenge.org/
- Cobb, J. R. (1948). Outline for the Study of Scoliosis. *Instructional Course Lectures*, 5, 261–275.
