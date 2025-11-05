from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split

# Definición de las variables según la eda

NUMERIC_FEATURES: List[str] = [
    "AccountWeeks", "DataUsage", "CustServCalls", "DayMins",
    "DayCalls", "MonthlyCharge", "OverageFee", "RoamMins"
]
CATEGORICAL_FEATURES: List[str] = ["ContractRenewal", "DataPlan"]
ORDINAL_FEATURES: List[str] = []  # si tuvieras alguna ordinal, se define aquí
TARGET_COL = "Churn"

# Creación del preprocesador (ColumnTransformer)

def create_preprocessor(
    numeric_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
    ordinal_features: Optional[List[str]] = None,
    ordinal_categories: Optional[List[List[str]]] = None
) -> ColumnTransformer:
    from sklearn.pipeline import Pipeline

    if numeric_features is None:
        numeric_features = NUMERIC_FEATURES
    if categorical_features is None:
        categorical_features = CATEGORICAL_FEATURES
    if ordinal_features is None:
        ordinal_features = ORDINAL_FEATURES

    # Pipeline numérico
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean"))
    ])

    # Pipeline categórico
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    # Pipeline ordinal 
    transformers = [
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ]

    if len(ordinal_features) > 0:
        ordinal_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ordinal", OrdinalEncoder(categories=ordinal_categories))
        ])
        transformers.append(("ordinal", ordinal_pipeline, ordinal_features))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False
    )

    return preprocessor

# División X, y y train/test

def split_xy(df: pd.DataFrame, target: str = TARGET_COL) -> Tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[target])
    y = df[target].astype(int)
    return X, y

def make_train_test(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    X, y = split_xy(df, TARGET_COL)
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

# Carga limpia de datos

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in CATEGORICAL_FEATURES + [TARGET_COL]:
        if c in df.columns:
            df[c] = df[c].astype(int)
    return df
