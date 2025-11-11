# Proyecto final - Machine learning Supervisado
## Predicción de Deserción de clientes (Telecom Churn)

## Comprensión del negocio

El proyecto busca predecir **la deserción de clientes (Churn)** en una empresa de telecomunicaciones.  
A partir de características del cliente —como su antigüedad (`AccountWeeks`), uso de datos (`DataUsage`), plan contratado (`DataPlan`), cantidad de llamadas al servicio al cliente (`CustServCalls`), entre otras—, el objetivo es determinar la probabilidad de que un cliente **abandone el servicio**.

El modelo permitirá a la empresa:
- Identificar clientes con alto riesgo de fuga.
- Optimizar estrategias de retención.
- Reducir costos de adquisición de nuevos usuarios.

**Variable objetivo:** `Churn` (1 = el cliente canceló el servicio, 0 = sigue activo).  
**Tipo de modelo:** Supervisado, clasificación binaria.

El objetivo de este proyecto sera **predecir que probabilidad hay de que un cliente abandone la compañia** tomando como base los patrones de uso, cargos mensuales y comportamiento de atención al cliente.

### Objetivos específicos
1. Analizar las variables que más influyen en la deserción de los clientes.
2. Construir un modelo supervisado capaz de predecir si un cliente se marchará ('Churn = 1').
3. Evaluar el rendimiento de varios modelos y seleccionar el mejor para el proyecto.
4. Aplicar el balancio de clases y técnicas de ensamble para mejorar la precisión.

### Dataset
**Fuente:** [Telecom Churn Dataset](https://www.kaggle.com/datasets/barun2104/telecom-churn)  
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
1. Exploración y limpieza de los datos -> Cargar_datos.iynb y comprension_eda.ipynb  
2. Transformación y feature engineering -> ft_engineering.py
3. Entrenamiento de modelos base -> heuristic_model.py  
4. Entrenamiento supervisado avanzado -> model_training.ipynb 
5. Evaluación de desempeño -> model_evaluation.ipynb  
6. Despliegue del modelo -> model_deploy.ipynb
7. Monitoreo y métricas en producción -> model_monitoring

#Activar el entorno virtual: 
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Instalación del entorno
pip install --upgrade pip setuptools wheel

# Instalar dependencias principales
pip install -r requirements.txt

# Ejecución de los Scripts en la terminal del Deepnote, recomiendo esa
1. Abrir la terminal del proyecto, en New Terminal.
2. Ejecutar el script del modelo base: python src/heuristic_model.py

  
Para el feature engineering: 
1. poner cd src y despues python ft_engineering.py


### Resultados Esperados
- Identificar los factores clave que influyen en la deserción de clienrtes.  
- Entrenar un modelo predictivo robusto de XGBoost o Random Forest.  
- Detectar los clientes con alta probabilidad de abandono.
- Formular estrategias de retención basadas en los insights del modelo.

### Salida esperada: 
RESULTADOS DEL MODELO BASE (LOGISTIC REGRESSION)
Accuracy: 0.8125

Matriz de Confusión:
[[845   75]
 [120  160]]

Reporte de Clasificación:
              precision    recall  f1-score   support
           0       0.88      0.92      0.90       920
           1       0.68      0.57      0.62       280
    accuracy                           0.84      1200

Conclusiones: Los modelos de Random Forest y XGBoost dieron los mejores resultados, con F1-score > 0.70
-Las variables con mayor importancia fueron: CustServCalls, DataPlan y Data Usage, y OverageFee.
-El modelo final permite predecir con una buena precisión los clientes que son propensos a abandonar la compañia. 

### Trabajo realizado por:
**Santiago López Gómez**  
Estudiante de Ingeniería de Sistemas.













