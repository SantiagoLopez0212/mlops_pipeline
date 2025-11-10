from fastapi import FastAPI, UploadFile, File
import pandas as pd
import joblib
import io

app = FastAPI(title="API de Predicción de Churn", description="Modelo MLOps - Santiago López Gómez")

# Cargar modelo y preprocesador
model = joblib.load("model_heuristic.pkl")
preprocessor = joblib.load("preprocessor.pkl")

@app.get("/")
def home():
    return {"mensaje": "API de predicción de churn activa 🚀"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Leer CSV subido
    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

    # Preprocesar y predecir
    X_new = preprocessor.transform(df)
    y_pred = model.predict(X_new)
    df["Predicción_Churn"] = y_pred

    # Retornar primeras filas con predicción
    return {"resultados": df.head(10).to_dict(orient="records")}

