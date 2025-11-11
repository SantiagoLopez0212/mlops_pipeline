from typing import List, Tuple
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

"""
Módulo: ft_engineering.py
Este script realiza la ingeniería de características para el proyecto de churn prediction.
Incluye pipelines de preprocesamiento, imputación y codificación de variables,
y la división de datos en conjuntos de entrenamiento y prueba.
"""

# Variables según la EDA
NUMERIC_FEATURES: List[str] = [
    "AccountWeeks", "DataUsage", "CustServCalls", "DayMins",
    "DayCalls", "MonthlyCharge", "OverageFee", "RoamMins"
]

CATEGORICAL_FEATURES: List[str] = ["ContractRenewal", "DataPlan"]
ORDINAL_FEATURES: List[str] = []
TARGET_COL: str = "Churn"

def create_feature_pipeline() -> ColumnTransformer:
    """
    Este codigo crea un pipeline de preprocesamiento que aplica:
    - Imputación por mediana a variables numéricas.
    - Codificación OneHot a variables categóricas.
    Retorna un objeto ColumnTransformer listo para entrenamiento.
    """
    numeric_pipeline = SimpleImputer(strategy="median")
    categorical_pipeline = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES)
        ],
        remainder="passthrough"
    )
    return preprocessor


def split_dataset(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
                  ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Divide el dataset en conjuntos de entrenamiento y prueba.
    Aplica estratificación por la variable objetivo 'Churn'.
    Retorna X_train, X_test, y_train, y_test.
    """
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


if __name__ == "__main__":
    try:
        df = pd.read_csv("../telecom_churn.csv")
        print("Dataset cargado correctamente")
        print(f"Dimensiones: {df.shape}")

        preprocessor = create_feature_pipeline()
        X_train, X_test, y_train, y_test = split_dataset(df)

        X_train_transformed = preprocessor.fit_transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)

        # Convertir las salidas transformadas a DataFrames
        X_train_df = pd.DataFrame(
            X_train_transformed.toarray() if hasattr(X_train_transformed, "toarray") else X_train_transformed
        )
        X_test_df = pd.DataFrame(
            X_test_transformed.toarray() if hasattr(X_test_transformed, "toarray") else X_test_transformed
        )

        print("Transformación completada correctamente.")
        print(f"Shape entrenamiento: {X_train_df.shape}")
        print(f"Shape prueba: {X_test_df.shape}")
        print("Dataset limpio y transformado listo para modelado.")

    except FileNotFoundError:
        print("Error: No se encontró el archivo '../telecom_churn.csv'. Verifica la ruta.")
    except Exception as e:
        print(f"Error inesperado: {e}")
