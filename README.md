# Proyecto MLOps: Predicción de Pago a Tiempo

##  1. Caso de Negocio
El objetivo de este proyecto es predecir si un cliente pagará a tiempo un préstamo (`Pago_atiempo`), optimizando la toma de decisiones en la aprobación de créditos y reduciendo el riesgo de morosidad para la entidad financiera.

##  2. Principales Hallazgos (EDA)
- **Desbalance de clases:** Existe una fuerte proporción de clientes que pagan a tiempo (~95%) frente a los que no (~5%). Se utilizó estratificación en el split para manejar esto.
- **Variables predictivas:** El `puntaje_datacredito` y `saldo_mora` demostraron ser los mejores indicadores de impago.
- **Multicolinealidad:** Se eliminaron variables redundantes como `cuota_pactada`, `saldo_principal` y `puntaje` para evitar ruido en el modelo.

##  3. Arquitectura del Pipeline
1. **Ingeniería (`ft_engineering.py`):** Limpieza de datos, tratamiento de variables categóricas, creación del atributo `ratio_deuda_ingreso`, división 80/20 estratificada y transformación vía `ColumnTransformer` (evitando Data Leakage).
2. **Entrenamiento (`model_training_evaluation.py`):** Evaluación de Regresión Logística, Árbol de Decisión y Random Forest. El modelo final se seleccionó balanceando Performance (F1, ROC-AUC), Consistencia (Gap Train-Test) y Escalabilidad (Tiempo).
3. **Monitoreo (`model_monitoring.py` + Streamlit):** Cálculo de Data Drift (PSI, KS, Chi-cuadrado, Jensen-Shannon) comparando la data histórica con datos simulados de producción.
4. **Despliegue (`model_deploy.py`):** API REST con FastAPI para predicciones individuales y por lotes, empaquetada en Docker.

---

##  4. Guía Definitiva de Instalación y Ejecución (Paso a Paso)

Para ejecutar este proyecto desde cero en una computadora nueva, abre tu terminal (Mac/Linux) o consola (Windows), asegúrate de tener instalado Python 3.9+ y Docker Desktop, y sigue este bloque de comandos en orden.

```bash
# ---------------------------------------------------------
# PASO 0: Clonar el repositorio y entrar a la carpeta
# ---------------------------------------------------------
git clone https://github.com/JulianRamirez43/MachineLearningProject.git
cd MachineLearningProject

# ---------------------------------------------------------
# PASO 1: Configurar el Entorno Virtual e Instalar Dependencias
# ---------------------------------------------------------

#  SI ESTÁS EN MAC/LINUX, ejecuta estas 3 líneas:
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip && pip install -r requirements.txt

#  SI ESTÁS EN WINDOWS, ejecuta estas 3 líneas en su lugar:
# python -m venv venv
# venv\Scripts\activate
# pip install --upgrade pip && pip install -r requirements.txt

# (Verás que aparece "(venv)" al inicio de tu línea de comandos, indicando que está activo)

# ---------------------------------------------------------
# PASO 2: Procesar datos, Entrenar modelo y Calcular Drift
# ---------------------------------------------------------
python mlops_pipeline/ft_engineering.py
python mlops_pipeline/model_training_evaluation.py
python mlops_pipeline/model_monitoring.py

# ---------------------------------------------------------
# PASO 3: Levantar el Panel de Monitoreo (Streamlit)
# ---------------------------------------------------------
# Abre una pestaña en el navegador con el dashboard interactivo. 
# Para cerrarlo y volver a la terminal, presiona Ctrl + C.
streamlit run app_monitoring.py

# ---------------------------------------------------------
# PASO 4: Desplegar la API con Docker (Servicio Web)
# ---------------------------------------------------------
# Asegúrate de que Docker Desktop esté abierto antes de correr esto
docker build -t mlops-api .
docker run -p 8000:8000 mlops-api

# Una vez corriendo, abre en tu navegador: http://127.0.0.1:8000/docs
# Para detener el contenedor, presiona Ctrl + C en la terminal.

# ---------------------------------------------------------
# PASO 5 (Opcional): Análisis de Calidad de Código (SonarQube)
# ---------------------------------------------------------
# Si deseas replicar el análisis estático del código:
pip install pysonar
pysonar --sonar-token=TU_TOKEN_AQUI --sonar-project-key=JulianRamirez43_MachineLearningProject --sonar-organization=julianramirez43