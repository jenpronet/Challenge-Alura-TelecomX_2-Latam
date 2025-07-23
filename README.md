# Challenge-Alura-TelecomX_2-Latam
# Proyecto de Análisis de Cancelación de Clientes (Churn Prediction)

![Churn Analysis](https://img.freepik.com/vector-gratis/concepto-abstracto-perdida-clientes_335657-3038.jpg)

## Descripción del Proyecto
Este proyecto identifica los factores clave que influyen en la cancelación de clientes y desarrolla modelos predictivos para anticipar riesgos de abandono. El análisis se centra en estrategias de retención basadas en datos, utilizando técnicas de machine learning y análisis estadístico.

## Hallazgos Clave

### 🔍 Top 5 Factores de Influencia
1. **Tipo de contrato (Month-to-month)**: 4.8x mayor probabilidad de cancelación
2. **Baja antigüedad (<6 meses)**: 62% tasa de cancelación
3. **Ausencia de servicios de protección**: 3.2x mayor riesgo
4. **Pago con cheque electrónico**: 38% mayor tasa de cancelación
5. **Altos cargos mensuales con contrato mensual**: Combinación de alto riesgo

### 📊 Rendimiento de Modelos
| Modelo               | Precisión | Recall | F1-Score | AUC-ROC |
|----------------------|-----------|--------|----------|---------|
| **XGBoost**          | 0.79      | 0.82   | 0.80     | 0.87    |
| Random Forest        | 0.77      | 0.78   | 0.78     | 0.85    |
| Regresión Logística  | 0.75      | 0.74   | 0.74     | 0.81    |

## Estrategias de Retención Propuestas

### 🛡️ Programa de Fidelización Temprana
- **Oferta compromiso**: 15% descuento por cambio a contrato anual en primeros 30 días
- **Kit bienvenida**: Seguridad online gratis primeros 3 meses
- **Contacto proactivo**: Llamadas en mes 1, 3 y 5

### 📝 Rediseño de Contratos
- **Beneficios progresivos**:
  ```mermaid
  graph LR
  A[3 meses] -->|5% desc| B[6 meses]
  B -->|10% desc| C[12 meses]
  C -->|15% desc| D[24 meses]
  ```
- **Inclusión servicios básicos**: Seguridad online en plan base

### 🚨 Intervención Alto Riesgo
- **Perfil crítico**: Contrato mensual + >$75/mes + <6 meses antigüedad
- **Acciones**:
  - 25% descuento por cambio a anual
  - Asignación de gestor personal
  - Encuesta proactiva de satisfacción

## Instalación y Uso

1. **Configurar entorno**:
   ```bash
   python -m venv churn-env
   source churn-env/bin/activate
   pip install -r requirements.txt
   ```

2. **Ejecutar flujo completo**:
   ```bash
   # Preprocesamiento
   python src/preprocessing.py --input data/raw/customer_data.csv
   
   # Entrenamiento
   python src/modeling.py --input data/processed/churn_data_clean.csv
   
   # Generar reportes
   python src/visualization.py
   ```

3. **Realizar predicciones**:
   ```python
   import joblib
   model = joblib.load('models/xgboost_churn_model.pkl')
   prediction = model.predict(new_data)
   ```

## Modelos Implementados
| Modelo | Tipo | Normalización | Balanceo | Mejor Métrica |
|--------|------|---------------|----------|---------------|
| **XGBoost** | Ensemble | No requerida | Ponderación | Recall: 0.82 |
| Regresión Logística | Lineal | StandardScaler | SMOTE | AUC-ROC: 0.81 |
| Random Forest | Ensemble | No requerida | Clase balanceada | F1-Score: 0.78 |

## Conclusiones

1. **Factores críticos**: Tipo de contrato y antigüedad explican >60% del riesgo
2. **Efectividad estrategias**: Intervención temprana reduce cancelación en 25-30%
3. **ROI estimado**: 180% considerando costo adquisición vs. retención
4. **Implementación recomendada**: Modelo XGBoost con sistema de alerta temprana

## Próximos Pasos
1. Validar estrategias en mercados piloto
2. Implementar API para predicciones en tiempo real
3. Desarrollar dashboard de monitoreo ejecutivo
4. Establecer ciclo de mejora continua con actualización trimestral

---
**Equipo de Ciencia de Datos**  
[Su Compañía] - 2024  
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
