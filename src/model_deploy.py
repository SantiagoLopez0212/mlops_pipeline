from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# Inicializar la app
app = FastAPI(title="API de Predicción de Churn")

# Cargar modelo y preprocesador
model = joblib.load("model_heuristic.pkl")
preprocessor = joblib.load("preprocessor.pkl")

# Definir el formato de entrada
class Cliente(BaseModel):
    AccountWeeks: float
    DataUsage: float
    CustServCalls: int
    DayMins: float
    DayCalls: int
    MonthlyCharge: float
    OverageFee: float
    RoamMins: float
    ContractRenewal: int
    DataPlan: int

@app.post("/predict")
def predict(clientes: list[Cliente]):
    # Convertir lista de objetos a DataFrame
    df = pd.DataFrame([c.dict() for c in clientes])

    # Aplicar preprocesamiento
    X_new = preprocessor.transform(df)

    # Realizar predicciones
    preds = model.predict(X_new)
    probs = model.predict_proba(X_new)[:, 1]

    # Formatear respuesta JSON
    resultados = [
        {
            "Cliente": i + 1,
            "Predicción": "Churn" if p == 1 else "No Churn",
            "Probabilidad": round(prob, 2)
        }
        for i, (p, prob) in enumerate(zip(preds, probs))
    ]

    return {"Resultados": resultados}
