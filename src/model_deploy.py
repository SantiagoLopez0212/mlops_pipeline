import pandas as pd
import joblib

# Cargar el modelo y preprocesador entrenados
model = joblib.load("model_heuristic.pkl")
preprocessor = joblib.load("preprocessor.pkl")

# Crear un nuevo cliente para probar la predicción
nuevo_cliente = pd.DataFrame({
    "AccountWeeks": [120],
    "DataUsage": [3.5],
    "CustServCalls": [1],
    "DayMins": [240],
    "DayCalls": [110],
    "MonthlyCharge": [70],
    "OverageFee": [10],
    "RoamMins": [15],
    "ContractRenewal": [1],
    "DataPlan": [1]
})

# Aplicar preprocesador
X_new = preprocessor.transform(nuevo_cliente)

# Realizar predicción
pred = model.predict(X_new)[0]
prob = model.predict_proba(X_new)[0][1]

print(f"Predicción: {'Churn' if pred == 1 else 'No Churn'}")
print(f"Probabilidad estimada: {prob:.2f}")

