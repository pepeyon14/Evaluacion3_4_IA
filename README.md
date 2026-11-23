# ☀️ Evaluación 3 y 4 — Segmentación Fotovoltaica y Automatización en Producción (Planta Lechuzas)

Repositorio oficial para la **Evaluación 3 y Evaluación 4 de Inteligencia Artificial**, enfocado en el análisis de regímenes operativos de paneles solares fotovoltaicos (`10 - lechuzasdataset.csv`, con **105,216 registros**) mediante aprendizaje no supervisado (**K-Means**) y su automatización en producción como servicio web (**FastAPI + Random Forest**).

> 📘 **Para una explicación pedagógica paso a paso de todo el proyecto y guía de defensa oral, revisa también:** [`Desafio_Lechuzas/EXPLICACION_COMPLETA_PROYECTO.md`](Desafio_Lechuzas/EXPLICACION_COMPLETA_PROYECTO.md)

---

## 📁 Estructura del Repositorio

```text
Evaluacion3_4_IA/
├── Desafio_Lechuzas/
│   ├── EXPLICACION_COMPLETA_PROYECTO.md               # Explicación detallada paso a paso y guía para defensa oral (Ev. 4)
│   ├── Notebook_Ev3_Segmentacion_Fotovoltaica.ipynb   # Jupyter Notebook documentado con EDA, K-Means y Clasificación
│   ├── 01_clusterizacion_y_eda.py                     # Paso 1: Limpieza, EDA, Método del Codo y K-Means (k=4)
│   ├── 02_entrenar_clasificador.py                    # Paso 2: Entrenamiento RandomForest (99.56% Acc) y serialización (.pkl)
│   ├── main.py                                        # Paso 3: API REST en FastAPI para inferencia y alertas en producción
│   ├── 03_mapa_conceptual_y_presentacion.html         # Paso 4: Mapa conceptual interactivo, gráficos y simulador web
│   ├── 10 - lechuzasdataset.csv                       # Dataset original de monitoreo solar (105,216 filas x 5 columnas)
│   ├── 11 - IA -Desafío Segmentación...pdf            # Enunciado y rúbrica oficial (Ev. 3 y Ev. 4)
│   ├── models/                                        # Artefactos serializados (.pkl) listos para producción
│   │   ├── clasificador_solar_rf.pkl
│   │   ├── kmeans_solar.pkl
│   │   └── scaler_solar.pkl
│   └── graficos/
│       └── metricas_evaluacion_k.csv                  # Tabla de Inercia, Silhouette, Davies-Bouldin y Calinski-Harabasz
├── requirements.txt                                   # Librerías necesarias para ejecutar todo el proyecto
└── README.md                                          # Documentación general del repositorio
```

---

## 📊 Resultados Reales Obtenidos sobre las 105,216 Filas

### 1. Evaluación de Hiperparámetros en K-Means ($k = 2..6$)

| $k$ (Clusters) | Inercia (WCSS - Codo) | Silhouette Score | Índice Davies-Bouldin | Índice Calinski-Harabasz |
| :---: | :---: | :---: | :---: | :---: |
| **2** | `102,651.4` | `0.6910` | `0.4902` | `326,156.8` |
| **3** | `72,264.5` | `0.5794` | `0.8139` | `253,772.4` |
| **4 (Elegido)** | **`53,214.3`** | **`0.4345`** | **`0.8740`** | **`242,304.7`** |
| **5** | `44,616.7` | `0.4117` | `0.9225` | `221,811.7` |
| **6** | `37,498.4` | `0.4247` | `0.9171` | `215,126.4` |

### 2. Perfil Físico de los 4 Regímenes Fotovoltaicos Descubiertos ($k = 4$)

| Cluster | Régimen Operativo | Registros (%) | Potencia Media | Radiación Media | Temp. Ambiente | Temp. Panel |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **0** | **Operación Nocturna / Standby** | `43,012` (`40.9%`) | `-32.5 W` | `11.5 W/m²` | `6.6 °C` | `5.8 °C` |
| **1** | **Baja Radiación / Transición** | `29,999` (`28.5%`) | `-163.6 W` | `60.0 W/m²` | `15.0 °C` | `15.0 °C` |
| **2** | **Generación Media / Templada** | `14,492` (`13.8%`) | `-1,783.6 W` | `612.4 W/m²` | `21.0 °C` | `25.3 °C` |
| **3** | **Generación Pico / Alta Temp. Panel** | `17,713` (`16.8%`) | `-2,690.1 W` | `952.6 W/m²` | `32.0 °C` | `39.5 °C` |

### 3. Rendimiento del Clasificador Supervisado en Producción (`RandomForestClassifier`)
* **Conjunto de Prueba (20% estratificado = `21,044` filas)**:
  * **Accuracy**: **`99.56%` (`0.9956`)**
  * **Macro Avg F1-Score**: **`0.99`**

---

## 🛠️ Instrucciones Rápidas de Ejecución

### 1. Instalar dependencias
```powershell
python -m pip install -r requirements.txt
```

### 2. (Opcional) Re-entrenar Clusterización y Clasificador
*(Los modelos `.pkl` ya vienen entrenados en `Desafio_Lechuzas/models/`, pero puedes regenerarlos cuando quieras)*:
```powershell
python "Desafio_Lechuzas/01_clusterizacion_y_eda.py"
python "Desafio_Lechuzas/02_entrenar_clasificador.py"
```

### 3. Iniciar la API de Producción (FastAPI)
```powershell
python -m uvicorn Desafio_Lechuzas.main:app --reload --port 8000
```
Abre en tu navegador: **http://127.0.0.1:8000/docs**
