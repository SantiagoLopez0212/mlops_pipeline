"""
Descripción:
Este script entrena un modelo base de Machine Learning para predecir la probabilidad de churn en clientes de telecomunicaciones.

Incluye:
- Carga segura del dataset (compatible con entorno local y Docker)
- Pipeline de ingeniería de características desde ft_engineering.py
- División de datos en entrenamiento y prueba
- Entrenamiento del modelo base (LogisticRegression)
- Evaluación mediante métricas y matriz de confusión
- Guardado del modelo y preprocesador para despliegue futuro
"""

import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# Importar funciones personalizadas
from ft_engineering import create_feature_pipeline, split_dataset


def run_heuristic_model():


    # Cargar datos 
    csv_path = os.path.join(os.path.dirname(__file__), "..", "telecom_churn.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No se encontró el archivo CSV en la ruta: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"Dataset cargado correctamente ({df.shape[0]} registros, {df.shape[1]} columnas)")

    # Crear pipeline de características 
    preprocessor = create_feature_pipeline()

    # División de datos 
    X_train, X_test, y_train, y_test = split_dataset(df)
    print("División de datos completada correctamente.")

    # Aplicar transformaciones 
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # Entrenar modelo base 
    model = LogisticRegression(max_iter=1000, solver='liblinear', random_state=42)
    model.fit(X_train_transformed, y_train)
    print("Modelo entrenado correctamente.")

    # Evaluar modelo 
    y_pred = model.predict(X_test_transformed)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print("\nRESULTADOS DEL MODELO BASE (LOGISTIC REGRESSION) ")
    print(f"Accuracy: {acc:.4f}")
    print("\nMatriz de Confusión:\n", cm)
    print("\nReporte de Clasificación:\n", report)

    # Guardar modelo y preprocesador
    os.makedirs("../models", exist_ok=True)
    joblib.dump(model, "../models/model_baseline.pkl")
    joblib.dump(preprocessor, "../models/preprocessor.pkl")

    print("Modelo y preprocesador guardados exitosamente en la carpeta 'models/'.")

    return model, preprocessor


# Ejecución principal 
if __name__ == "__main__":
    run_heuristic_model()


    except FileNotFoundError:
        print("Error: No se encontró el archivo '../telecom_churn.csv'. Verifica la ruta.")
    except Exception as e:
        print(f" Error inesperado: {e}")
