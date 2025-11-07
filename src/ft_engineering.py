from typing import List, Tuple
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# Variables según la EDA
NUMERIC_FEATURES: List[str] = [
    "AccountWeeks", "DataUsage", "CustServCalls", "DayMins",
    "DayCalls", "MonthlyCharge", "OverageFee", "RoamMins"
]

CATEGORICAL_FEATURES: List[str] = ["ContractRenewal", "DataPlan"]
ORDINAL_FEATURES: List[str] = []
TARGET_COL: str = "Churn"

# Pipeline de transformación
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

# División de los datos
def split_dataset(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
                  ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
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

        print("Transformación completada correctamente.")
        print(f" Shape entrenamiento: {X_train_transformed.shape}")
        print(f" Shape prueba: {X_test_transformed.shape}")

    except FileNotFoundError:
        print("Error: No se encontró el archivo '../telecom_churn.csv'. Verifica la ruta.")
    except Exception as e:
        print(f" Error inesperado: {e}")
