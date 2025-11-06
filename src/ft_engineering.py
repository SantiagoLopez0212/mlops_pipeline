from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split

# Variables usando la EDA
NUMERIC_FEATURES: List[str] = [
    "AccountWeeks", "DataUsage", "CustServCalls", "DayMins",
    "DayCalls", "MonthlyCharge", "OverageFee", "RoamMins"
]

CATEGORICAL_FEATURES: List[str] = ["ContractRenewal", "DataPlan"]
ORDINAL_FEATURES: List[str] = []  # si tuvieras alguna ordinal, se define aquí
TARGET_COL: str = "Churn"


# Pipeline de transformación para variables numéricas y categóricas
def create_feature_pipeline() -> ColumnTransformer:
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


# Función de división de los datos
def split_dataset(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Cargar el dataset
    df = pd.read_csv("../telecom_churn.csv)
    print(f"Dimensiones: {df.shape}\n")

    # Crear el pipeline
    preprocessor = create_feature_pipeline()
    print("Pipeline de características: ")

    # División de los datos
    X_train, X_test, y_train, y_test = split_dataset(df)
    print(f"Conjunto de entrenamiento: {X_train.shape}, Prueba: {X_test.shape}\n")

    # Aplicar el pipeline de transformación
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    print("Transformación completada.")
    print(f"Shape final de entrenamiento: {X_train_transformed.shape}")
    print(f"Shape final de prueba: {X_test_transformed.shape}\n")

    # Primeras filas
    print("Primeras filas del dataset original:")
    print(X_train.head(3))
    print("\nPipeline aplicado. Listo para modelar.")

"""
Este código genera los pipelines de transformación de variables y
retorna los datasets de validación y entrenamiento listos para modelar.
"""
