# Proyecto final - Machine learning Supervisado
## Predicción de Deserción de clientes (Telecom Churn)

### Comprensión del negocio
Una empresa de telecomunicaciones busca reducir la tasa de deserción de clientes (churn).
Este churn pasa cuando un cliente toma la decisión de cancelar su servicio, generando pérdidas de ingresos y mayores costos de adquisición de nuevos usuarios.

El objetivo de este proyecto sera **predecir que probabilidad hay de que un cliente abandone la compañia** tomando como base los patrones de uso, cargos mensuales y comportamiento de atención al cliente.

### Objetivos específicos
1. Analizar las variables que más influyen en la deserción de los clientes.
2. Construir un modelo supervisado capaz de predecir si un cliente se marchará ('Churn = 1').
3. Evaluar el rendimiento de varios modelos y seleccionar el mejor para el proyecto.
4. Aplicar el balancio de clases y técnicas de ensamble para mejorar la precisión.

### 📂 Dataset
**Fuente:** [Telecom Churn Dataset](https://www.kaggle.com/datasets)  
**Descripción:** Contiene información de clientes de una empresa de telecomunicaciones, incluyendo:
- `AccountWeeks`: Antigüedad del cliente.  
- `ContractRenewal`: Indicador de renovación de contrato.  
- `DataPlan` y `DataUsage`: Plan y consumo de datos.  
- `CustServCalls`: Llamadas al servicio al cliente.  
- `MonthlyCharge`, `OverageFee`, `RoamMins`: Cargos mensuales, excesos y minutos en roaming.  
- `Churn`: Variable objetivo (1 = se fue, 0 = permaneció).

### Tipo de Proyecto
- **Modelo:** Supervisado (Clasificación binaria)  
- **Variable objetivo:** `Churn`  
- **Métricas principales:** ROC-AUC, F1-score  
- **Lenguaje:** Python  
- **Herramientas:** Scikit-learn, Imbalanced-learn, XGBoost, Pandas, Seaborn  

### Flujo del Proyecto
1. Limpieza y análisis exploratorio (EDA)  
2. Preprocesamiento y feature engineering  
3. Entrenamiento de modelos supervisados  
4. Aplicación de balanceo (SMOTE)  
5. Evaluación de métricas y comparación de modelos  
6. Selección del modelo ganador (XGBoost)

### Resultados Esperados
- Identificar los factores clave que influyen en la deserción.  
- Entrenar un modelo predictivo robusto que permita detectar clientes propensos a abandonar.  
- Formular estrategias de retención basadas en los resultados del modelo.

### Autor
**Santiago López Gómez**  
Estudiante de Ingeniería de Sistemas - Universidad Católica Luis Amigó  
2025
