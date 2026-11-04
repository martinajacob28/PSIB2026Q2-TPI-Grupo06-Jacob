# PSIB2026Q2-TPI-Jacob
# Caracterización automática de la escoliosis mediante análisis geométrico y textural de radiografías de columna vertebral.
Este repositorio contiene el desarrollo de un pipeline automatizado en Python para el análisis geométrico y morfológico de radiografías de columna vertebral, orientado al diagnóstico y caracterización de la escoliosis. El proyecto fue desarrollado en entornos de Jupyter Notebook (Google Colab).

El sistema se basa en la extraccion de caracteristicas geometricas de la columna vertebral con el fin de analizar el angulo de Cobb de cada una y de esa manera evaluar la gravedad de la enfermedad. 
La gravedad de la escoliosis se caracteriza en: 
1. Leve: 10° ≤ Cobb < 25°
2. Moderada: 	25° ≤ Cobb < 40°
3. Grave: 	Cobb ≥ 40°




## Configuracion del dataset 
El pipeline fue desarrollado utilizando Google Colab conectado a Google Drive como almacenamiento en la nube. Dado que el dataset proviene de AASCE (Accurate Automated Spinal Curvature Estimation) y fue estructurado manualmente en drive, cualquier usuario que descargue este proyecto deberá descargar el dataset y modificar las rutas. 

1. **Descargar el Dataset:** Obtener el conjunto de datos oficial desde https://aasce19.grand-challenge.org/ 
2. **Subir a Google Drive:** Cargar las carpetas de imágenes a Drive.
3. **Modificar las Rutas en el Notebook:** Al abrir el archivo `.ipynb`, se deberán modifica
