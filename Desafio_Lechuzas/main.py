import os
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
import joblib
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "clasificador_solar_rf.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler_solar.pkl")

app = FastAPI(
    title="API de Segmentación y Monitoreo Fotovoltaico - Planta Lechuzas",
    description=(
        "Servicio automatizado en producción que recibe lecturas de sensores solares "
        "y variables contextuales (clima, fase lunar) para clasificar el régimen operativo."
    ),
    version="1.0",
)

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except Exception as e:
    raise RuntimeError(
        f"Error al cargar los modelos desde '{ BASE_DIR }/models/'. "
        f"Ejecuta primero 01_clusterizacion_y_eda.py y 02_entrenar_clasificador.py. Detalle: {e}"
    )

REGIMENES_SOLARES = {
    0: {
        "nombre": "Operación Nocturna / Standby",
        "descripcion": "Sin irradiancia solar efectiva; inversor en modo espera o consumo mínimo nocturno.",
        "accion_recomendada": "Monitoreo pasivo nocturno.",
    },
    1: {
        "nombre": "Baja Radiación / Transición (Amanecer-Atardecer o Nublado)",
        "descripcion": "Irradiancia baja con inicio de generación fotovoltaica y paneles templados.",
        "accion_recomendada": "Operación normal en rampa de subida/bajada.",
    },
    2: {
        "nombre": "Generación Media / Operación Templada Óptima",
        "descripcion": "Irradiancia moderada-alta con buena eficiencia térmica en celdas.",
        "accion_recomendada": "Inyección estable a red eléctrica.",
    },
    3: {
        "nombre": "Generación Pico / Alta Carga Térmica",
        "descripcion": "Máxima irradiancia y potencia generada con incremento térmico en superficie de panel.",
        "accion_recomendada": "Vigilar delta térmico del panel para evitar pérdida de eficiencia por calor.",
    },
}


class SensorFotovoltaicoInput(BaseModel):
    potencia: float = Field(..., alias="Potencia", description="Potencia eléctrica registrada (W)")
    radiacion: float = Field(..., alias="Radiacion", description="Irradiancia solar (W/m²)")
    temperatura: float = Field(..., alias="Temperatura", description="Temperatura ambiente (°C)")
    temperatura_panel: float = Field(..., alias="Temperatura panel", description="Temperatura superficial del panel (°C)")
    # Variables asociadas adicionales sugeridas en el caso de estudio
    condicion_climatica: Optional[str] = Field(
        default="Despejado",
        description="Contexto climático opcional (Despejado, Nublado, Lluvia, Polvo en suspensión)",
    )
    fase_lunar: Optional[str] = Field(
        default="No aplica (Diurno)",
        description="Fase lunar opcional para monitoreo nocturno (Luna Nueva, Cuarto Creciente, Luna Llena)",
    )

    model_config = ConfigDict(populate_by_name=True)


@app.get("/")
def read_root():
    return {
        "status": "active",
        "planta": "Fotovoltaica Lechuzas",
        "message": "API de segmentación y clasificación fotovoltaica activa",
    }


@app.post("/predict")
def predict_regimen_solar(data: SensorFotovoltaicoInput):
    try:
        # 1. Aplicar la misma regla física de preprocesamiento: truncar radiación negativa a 0.0
        radiacion_limpia = max(0.0, data.radiacion)

        # 2. Construir vector en el orden exacto del entrenamiento:
        #    [Potencia, Radiacion, Temperatura, Temperatura panel]
        features = np.array([[
            data.potencia,
            radiacion_limpia,
            data.temperatura,
            data.temperatura_panel,
        ]])

        # 3. Escalar y predecir cluster + probabilidad
        scaled_features = scaler.transform(features)
        cluster_id = int(model.predict(scaled_features)[0])
        probabilities = model.predict_proba(scaled_features)[0]
        confidence = float(probabilities[cluster_id])

        # 4. Indicadores operativos adicionales y reglas de alerta automática
        delta_termico = round(data.temperatura_panel - data.temperatura, 2)
        potencia_generada_w = round(abs(data.potencia) if data.potencia < 0 else 0.0, 2)

        alerta = "Sin anomalías"
        if radiacion_limpia > 400 and data.potencia >= 0:
            alerta = "ALERTA CRÍTICA: Alta radiación sin generación eléctrica (posible falla en inversor/fusible)"
        elif data.temperatura_panel >= 48.0 or delta_termico >= 15.0:
            alerta = "ALERTA TÉRMICA: Sobrecalentamiento en superficie del panel (pérdida de eficiencia)"

        info_regimen = REGIMENES_SOLARES.get(cluster_id, {})

        return {
            "cluster": cluster_id,
            "regimen_operativo": info_regimen.get("nombre", f"Cluster {cluster_id}"),
            "confianza_modelo": round(confidence, 4),
            "potencia_generada_estimada_W": potencia_generada_w,
            "delta_termico_panel_C": delta_termico,
            "alerta_operativa": alerta,
            "recomendacion": info_regimen.get("accion_recomendada", ""),
            "contexto_ambiental": {
                "condicion_climatica": data.condicion_climatica,
                "fase_lunar": data.fase_lunar,
                "radiacion_procesada_Wm2": radiacion_limpia,
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
