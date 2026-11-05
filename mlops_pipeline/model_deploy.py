import joblib
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
from fastapi import FastAPI
from pydantic import BaseModel

class PredictionInput(BaseModel):
    capital_prestado: float
    plazo_meses: float
    edad_cliente: int
    salario_cliente: float
    total_otros_prestamos: float
    puntaje_datacredito: int
    cant_creditosvigentes: int
    huella_consulta: int
    saldo_mora: int
    saldo_total: float
    creditos_sectorCooperativo: int
    creditos_sectorReal: int
    promedio_ingresos_datacredito: float
    mes_prestamo: int
    ratio_deuda_ingreso: float
    tipo_laboral_Empleado: int
    tipo_laboral_Independiente: int
    tendencia_ingresos: float
    tipo_credito_4: int
    tipo_credito_9: int
    tipo_credito_10: int
    tipo_credito_Otros: int

app = FastAPI(title="API de Scoring Crediticio")

# Cargar modelo
ruta_modelo = Path(__file__).resolve().parent.parent / "modelo_seleccionado.pkl"
# Si no lo encuentra en la carpeta padre (ej. dentro de Docker), lo busca en la misma carpeta
if not ruta_modelo.exists():
    ruta_modelo = Path(__file__).resolve().parent / "modelo_seleccionado.pkl"
if not ruta_modelo.exists():
    ruta_modelo = Path("modelo_seleccionado.pkl")

try:
    model = joblib.load(ruta_modelo)
    print("Modelo cargado exitosamente.")
except Exception as e:
    raise RuntimeError(f"No se pudo cargar el modelo: {e}")

@app.get("/")
def inicio():
    return {"Mensaje": "API funcionando. Dirígete a /docs para probar."}

@app.post("/predict")
def predecir_uno(data: PredictionInput) -> Dict[str, Any]:
    df_in = pd.DataFrame([data.model_dump()])
    pred = int(model.predict(df_in)[0])
    prob = float(model.predict_proba(df_in)[0][1])
    return {"prediccion": pred, "probabilidad": prob}

@app.post("/predict_batch")
def predecir_lote(data: List[PredictionInput]) -> List[Dict[str, Any]]:
    df_in = pd.DataFrame([d.model_dump() for d in data])
    preds = model.predict(df_in).tolist()
    probs = model.predict_proba(df_in)[:, 1].tolist()
    return [{"prediccion": int(p), "probabilidad": float(pr)} for p, pr in zip(preds, probs)]
