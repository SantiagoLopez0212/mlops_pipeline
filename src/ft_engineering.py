"""
Resultado esperado:
    - Genera los pipelines de transformación de variables.
    - Retorna los datasets de entrenamiento y validación listos para modelar.
"""

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


# Crear pipelines de los features
def create_feature_pipeline() -> ColumnTransformer:
    """
    #1. Crear un pipeline de transformación de variables numéricas y categóricas.
    
    Returns:
        ColumnTransformer: pipeline de preprocesamiento
    """
    numeric_pipeline = SimpleImputer(strategy="median")
    categorical_pipeline = OneHotEncoder(handle_unknown="ignore")

    # ColumnTransformer combina las transformaciones
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES)
        ],
        remainder="passthrough"  # conserva las columnas no transformadas
    )

    return preprocessor


# División de datos
def split_dataset(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Divide el dataset en conjuntos de entrenamiento y prueba.
    
    Args:
        df (pd.DataFrame): Dataset completo
        test_size (float): Proporción del conjunto de prueba
        random_state (int): Semilla de aleatoriedad
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test


# Función principal de prueba
if __name__ == "__main__":
    # Carga del dataset base
    df = pd.read_csv("../base_de_datos.csv")  # nota: el archivo está fuera de src

    print("Dataset cargado correctamente.")
    print(f"Dimensiones: {df.shape}")

    # Crear pipeline
    preprocessor = create_feature_pipeline()
    print("\nPipeline de features creado correctamente.")
    print(preprocessor)

    # División de datos
    X_train, X_test, y_train, y_test = split_dataset(df)
    print(f"\nTamaño entrenamiento: {X_train.shape}, prueba: {X_test.shape}")

    print("\nPrimeras filas del dataset procesado:")
    print(X_train.head())
