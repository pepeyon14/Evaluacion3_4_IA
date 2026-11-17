import os
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

# Configurar rutas relativas a la carpeta Desafio_Lechuzas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_INPUT = os.path.join(BASE_DIR, "10 - lechuzasdataset.csv")
CSV_OUTPUT = os.path.join(BASE_DIR, "10 - lechuzas_etiquetado.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
GRAPHS_DIR = os.path.join(BASE_DIR, "graficos")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(GRAPHS_DIR, exist_ok=True)

print("=" * 65)
print(" PASO 1: EDA, PREPROCESAMIENTO Y CLUSTERIZACIÓN FOTOVOLTAICA")
print("=" * 65)

# 1. Cargar el dataset
df = pd.read_csv(CSV_INPUT)
df.columns = df.columns.str.strip()

print(f"\n[1] Dimensiones originales: {df.shape[0]:,} filas x {df.shape[1]} columnas")
print(f"    Columnas: {list(df.columns)}")
print(f"    Valores nulos totales: {df.isnull().sum().sum()}")

# 2. Preprocesamiento y limpieza física
# 2.1 Eliminar columna 'id' (no aporta información física y distorsiona distancias)
df_clean = df.drop(columns=["id"]).copy()

# 2.2 Corregir valores negativos en 'Radiacion' (ruido nocturno del sensor piranómetro)
neg_rad_count = (df_clean["Radiacion"] < 0).sum()
print(f"\n[2] Preprocesamiento físico:")
print(f"    - Registros con Radiación < 0 W/m² (offset nocturno): {neg_rad_count:,}")
df_clean["Radiacion"] = df_clean["Radiacion"].clip(lower=0.0)
print("    - Se truncaron los valores negativos de Radiación a 0.0 W/m².")

# 2.3 Escalamiento de variables (indispensable para K-Means basado en distancia euclidiana)
FEATURE_COLS = ["Potencia", "Radiacion", "Temperatura", "Temperatura panel"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_clean[FEATURE_COLS].values)

# 3. Evaluación de hiperparámetros (k = 2 a 6)
print("\n[3] Evaluando número óptimo de clusters (k = 2..6):")
print("-" * 65)
print(f"{'k':>3} | {'Inercia (WCSS)':>15} | {'Silhouette':>11} | {'Davies-Bouldin':>15} | {'Calinski-Harabasz':>17}")
print("-" * 65)

k_values = [2, 3, 4, 5, 6]
metrics_history = []

for k in k_values:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_k = km.fit_predict(X_scaled)
    inertia = km.inertia_
    # Muestra estratificada aleatoria de 10,000 puntos para Silhouette rápido y exacto
    sil = silhouette_score(X_scaled, labels_k, sample_size=10000, random_state=42)
    db = davies_bouldin_score(X_scaled, labels_k)
    ch = calinski_harabasz_score(X_scaled, labels_k)
    metrics_history.append({
        "k": k,
        "inertia": inertia,
        "silhouette": sil,
        "davies_bouldin": db,
        "calinski_harabasz": ch,
    })
    print(f"{k:>3} | {inertia:>15,.1f} | {sil:>11.4f} | {db:>15.4f} | {ch:>17,.1f}")

print("-" * 65)

# 4. Entrenamiento del modelo K-Means seleccionado (k = 4 regímenes operativos solares)
K_OPTIMO = 4
print(f"\n[4] Entrenando K-Means definitivo con k = {K_OPTIMO} regímenes operativos...")
kmeans_raw = KMeans(n_clusters=K_OPTIMO, random_state=42, n_init=10)
raw_clusters = kmeans_raw.fit_predict(X_scaled)

# Ordenar los clusters de menor a mayor Radiación promedio para que el número
# de cluster tenga sentido físico directo (0=Noche, 1=Baja, 2=Media, 3=Pico)
cluster_means_rad = [
    df_clean.loc[raw_clusters == c, "Radiacion"].mean() for c in range(K_OPTIMO)
]
sorted_order = np.argsort(cluster_means_rad)
remap = {old_c: int(new_c) for new_c, old_c in enumerate(sorted_order)}

# Reordenar también los centroides dentro del objeto KMeans para que kmeans.predict()
# devuelva directamente los IDs ordenados (0, 1, 2, 3)
kmeans_raw.cluster_centers_ = kmeans_raw.cluster_centers_[sorted_order]
df_clean["cluster"] = np.vectorize(remap.get)(raw_clusters)

NOMBRES_REGIMENES = {
    0: "Operación Nocturna / Standby",
    1: "Baja Radiación / Transición",
    2: "Generación Media / Templada",
    3: "Generación Pico / Alta Temp. Panel",
}
df_clean["regimen_operativo"] = df_clean["cluster"].map(NOMBRES_REGIMENES)

# 5. Resumen estadístico por Cluster (Interpretación Física)
print("\n[5] Perfil físico promedio por Cluster descubierto:")
print("=" * 85)
resumen = (
    df_clean.groupby(["cluster", "regimen_operativo"])[FEATURE_COLS]
    .agg(["count", "mean"])
)
for c in range(K_OPTIMO):
    sub = df_clean[df_clean["cluster"] == c]
    pct = len(sub) / len(df_clean) * 100
    print(
        f"Cluster {c} ({NOMBRES_REGIMENES[c]}): {len(sub):,} registros ({pct:.1f}%)\n"
        f"   -> Potencia media: {sub['Potencia'].mean():.1f} W | "
        f"Radiación media: {sub['Radiacion'].mean():.1f} W/m² | "
        f"Temp. Amb: {sub['Temperatura'].mean():.1f} °C | "
        f"Temp. Panel: {sub['Temperatura panel'].mean():.1f} °C"
    )
print("=" * 85)

# 6. Guardar dataset etiquetado y modelos de clusterización
df_clean.to_csv(CSV_OUTPUT, index=False)
joblib.dump(kmeans_raw, os.path.join(MODELS_DIR, "kmeans_solar.pkl"))
joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler_solar.pkl"))
pd.DataFrame(metrics_history).to_csv(
    os.path.join(GRAPHS_DIR, "metricas_evaluacion_k.csv"), index=False
)

print(f"\n[OK] Dataset etiquetado guardado en: {CSV_OUTPUT}")
print(f"[OK] Modelos guardados en: {MODELS_DIR}")
