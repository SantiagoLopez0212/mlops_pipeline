import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from ft_engineering import create_feature_pipeline, split_dataset


def run_heuristic_model(csv_path: str = "../telecom_churn.csv"):
    df = pd.read_csv(csv_path)

    # Crear pipeline de características
    preprocessor = create_feature_pipeline()

    # División de datos
    X_train, X_test, y_train, y_test = split_dataset(df)

    # Aplicar transformaciones
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # Entrenar modelo base
    model = LogisticRegression(max_iter=1000, solver='liblinear')
    model.fit(X_train_transformed, y_train)

    # Predicciones
    y_pred = model.predict(X_test_transformed)

    # Métricas de evaluación
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print("Modelo base (Logistic Regression)")
    print(f"Accuracy: {acc:.4f}")
    print("\nMatriz de confusión:")
    print(cm)
    print("\nReporte de clasificación:")
    print(report)

    return model, preprocessor

if __name__ == "__main__":
    run_heuristic_model()
