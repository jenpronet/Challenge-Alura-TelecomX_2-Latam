# Proyecto de Predicción de Cancelación de Clientes (Churn Analysis)

![Churn Analysis](https://raw.githubusercontent.com/jenpronet/Challenge-Alura-TelecomX_2-Latam/refs/heads/main/Portada_CHURN.png)

## Propósito del Análisis
Este proyecto tiene como objetivo principal **predecir la probabilidad de cancelación de clientes** (churn) utilizando variables relevantes de comportamiento, contrato y demográficas. El análisis busca identificar patrones clave y factores de riesgo para desarrollar estrategias de retención efectivas y reducir la tasa de abandono de clientes.

## Estructura del Proyecto
```
├── data/
│   ├── raw/                    # Datos originales sin procesar
│   │   └── TelecomX_Data.json
│   └── processed/              # Datos tratados y listos para modelado
│       └── datos_tratados.csv
├── README.md                   # Este archivo
└── requirements.txt            # Dependencias de Python
```

## Proceso de Preparación de Datos

### 1. Clasificación de Variables
**Variables Categóricas (18):**
- `gender`, `partner`, `dependents`, `phone_service`, `multiple_lines`, `internet_service`, `online_security`, `online_backup`, `device_protection`, `tech_support`, `streaming_tv`, `streaming_movies`, `contract`, `paperless_billing`, `payment_method`

**Variables Numéricas (3):**
- `tenure` (antigüedad en meses)
- `monthly_charges` (cargos mensuales)
- `total_charges` (cargos totales)

**Variable Objetivo:**
- `churn` (cancelación: Sí/No)

### 2. Transformaciones Aplicadas
| Transformación | Variables Aplicadas | Justificación |
|----------------|---------------------|---------------|
| **One-Hot Encoding** | Todas categóricas | Convertir variables categóricas a formato numérico para modelos de ML |
| **StandardScaler** | `tenure`, `monthly_charges`, `total_charges` | Normalizar variables numéricas para modelos sensibles a escala (Regresión Logística, SVM) |
| **Conversión de Tipo** | `total_charges` (object → float) | Corregir tipo de datos para análisis numérico |
| **Manejo de Valores Faltantes** | `churn` (eliminación) | Eliminar registros incompletos en variable objetivo |

### 3. División de Datos
```python
# Separación estratificada 80% entrenamiento - 20% prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)
```
- **Justificación**: Mantener proporción de churn en ambos conjuntos (26.5% churn)
- **Resultado**: 
  - Entrenamiento: 5,634 registros (80%)
  - Prueba: 1,409 registros (20%)

## Justificación de Decisiones de Modelización

### Selección de Modelos
| Modelo | Justificación | Normalización Requerida |
|--------|---------------|-------------------------|
| **XGBoost** | Alto rendimiento con datos desbalanceados, manejo automático de relaciones no lineales | No |
| **Regresión Logística** | Interpretabilidad de coeficientes, referencia para modelos lineales | Sí |
| **Random Forest** | Robustez a outliers, buen rendimiento con múltiples variables | No |
| **SVM** | Efectivo en espacios de alta dimensión, buena separación de clases | Sí |

### Manejo de Desbalance (26.5% churn)
- **Técnica**: SMOTE (Synthetic Minority Oversampling)
- **Implementación**: Solo en datos de entrenamiento
- **Razón**: Mejorar detección de clase minoritaria (churn) sin afectar datos de prueba

### Métricas de Evaluación Prioritarias
1. **Recall** (capacidad de detectar verdaderos positivos)
2. **F1-Score** (balance entre precisión y recall)
3. **AUC-ROC** (capacidad de distinguir entre clases)

*Justificación: Es más costoso no identificar un cliente que cancelará (falso negativo) que intervenir innecesariamente a uno que no cancelaría (falso positivo).*

## Análisis Exploratorio (EDA) - Insights Clave

### 1. Distribución de Churn
```python
sns.countplot(x='churn', data=df)
plt.title('Distribución de Cancelación (Churn)')
```
**Insight**: 26.5% de cancelación, indicando desbalance moderado que requiere técnicas especiales de muestreo.

### 2. Relación entre Antigüedad y Cancelación
```python
sns.boxplot(x='churn', y='tenure', data=df)
plt.title('Antigüedad vs Estado de Cancelación')
```
**Insight**: Clientes que cancelan tienen significativamente menor antigüedad (mediana: 10 meses vs 38 meses para clientes leales).

### 3. Impacto del Tipo de Contrato
```python
contract_churn = df.groupby('contract')['churn'].value_counts(normalize=True).unstack()
contract_churn.plot(kind='bar', stacked=True)
plt.title('Tasa de Cancelación por Tipo de Contrato')
```
**Insight**: 
- Contratos mes a mes: 43% tasa de cancelación
- Contratos anuales: 11% tasa de cancelación
- **Conclusión**: El tipo de contrato es el predictor más fuerte de cancelación

### 4. Correlación de Variables
```python
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Matriz de Correlación')
```
**Insights clave**:
- Alta correlación negativa entre `tenure` y `churn` (-0.35)
- Alta correlación positiva entre `monthly_charges` y `churn` (0.19)
- Fuertes correlaciones entre servicios adicionales (`online_security`, `tech_support`)

## Instrucciones de Ejecución

### Requisitos Previos
1. Instalar Python 3.8+
2. Clonar repositorio:
   ```bash
   git clone https://github.com/jenpronet/Challenge-Alura-TelecomX_2-Latam.git
   cd Challenge-Alura-TelecomX_2-Latam
   ```

### Ejecutar Cuadernos Jupyter
1. Iniciar Jupyter:
   ```bash
   jupyter notebook
   ```
2. Ejecutar en orden:
   - `TelecomX_2.ipynb`
   
### Cargar Datos Tratados
```python
import pandas as pd

# Desde los notebooks
df = pd.read_csv('../data/processed/datos_tratados.csv')

# Desde scripts Python
from src.data_processing import load_processed_data
df = load_processed_data()
```

### Entrenar Modelos
```python
# Ejemplo para XGBoost
from src.model_training import train_xgboost_model
from sklearn.model_selection import train_test_split

X = df.drop('churn', axis=1)
y = df['churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = train_xgboost_model(X_train, y_train)
```

## Resultados Clave y Conclusiones
El modelo XGBoost demostró el mejor rendimiento predictivo con **82% de recall** y **0.87 AUC-ROC**. Los principales factores predictivos identificados fueron:

1. Tipo de contrato (mes a mes)
2. Baja antigüedad (<12 meses)
3. Ausencia de servicios de protección online
4. Pagos con cheque electrónico
5. Altos cargos mensuales

**Estrategias recomendadas**:
- Programa de retención temprana para nuevos clientes
- Incentivos para conversión a contratos anuales
- Paquetes que incluyan servicios de seguridad online
- Sistema de alerta temprana con modelo predictivo
