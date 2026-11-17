import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Configurar rutas dentro de Desafio_Lechuzas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_ETIQUETADO = os.path.join(BASE_DIR, "10 - lechuzas_etiquetado.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

print("=" * 65)
print(" PASO 2: ENTRENAMIENTO Y SERIALIZACIÓN DEL CLASIFICADOR SOLAR")
print("=" * 65)

if not os.path.exists(CSV_ETIQUETADO):
    raise FileNotFoundError(
        f"No se encontró '{CSV_ETIQUETADO}'. Ejecuta primero 01_clusterizacion_y_eda.py"
    )

# 1. Cargar el dataset ya etiquetado con los clusters descubiertos en el Paso 1
df = pd.read_csv(CSV_ETIQUETADO)

FEATURE_COLS = ["Potencia", "Radiacion", "Temperatura", "Temperatura panel"]
X = df[FEATURE_COLS]
y = df["cluster"]

# 2. Dividir en entrenamiento (80%) y prueba (20%) manteniendo proporción de clusters
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Escalar las características con StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.values)
X_test_scaled = scaler.transform(X_test.values)

# 4. Entrenar clasificador supervisado Random Forest para producción
rf_classifier = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
)
rf_classifier.fit(X_train_scaled, y_train)

# 5. Evaluar rendimiento del clasificador sobre el conjunto de prueba
y_pred = rf_classifier.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)

print(f"\n=== Reporte de Clasificación en Producción (Accuracy: {acc:.4f}) ===")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Cluster 0 (Nocturno/Standby)",
            "Cluster 1 (Baja Radiación)",
            "Cluster 2 (Generación Media)",
            "Cluster 3 (Generación Pico)",
        ],
    )
)

# 6. Serializar el clasificador y el escalador para FastAPI (main.py)
joblib.dump(rf_classifier, os.path.join(MODELS_DIR, "clasificador_solar_rf.pkl"))
joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler_solar.pkl"))

print("[OK] Clasificador y escalador serializados exitosamente en 'Desafio_Lechuzas/models/'")
