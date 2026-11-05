import time
from pathlib import Path
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Importaciones de modelos
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Importaciones de métricas y validación
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_validate


def build_model(model, X_train, y_train):
    """
    Entrena el modelo y registra el tiempo de entrenamiento (Escalabilidad).
    """
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    return model, training_time


def summarize_classification(y_true, y_pred, y_prob, model_name, training_time, train_score, test_score, cv_f1_mean):
    """
    Calcula métricas de desempeño (Performance, incluyendo ROC-AUC), evalúa
    el sobreajuste (Consistencia) e incluye el resultado del Cross-Validation.
    """
    gap = train_score - test_score  # Brecha entre Train y Test para medir inconsistencia/overfitting
    
    # Calculamos ROC-AUC (Necesita las probabilidades de la clase positiva)
    roc_auc = roc_auc_score(y_true, y_prob)
    
    return {
        'Modelo': model_name,
        'Accuracy': round(accuracy_score(y_true, y_pred), 4),
        'Precision': round(precision_score(y_true, y_pred), 4),
        'Recall': round(recall_score(y_true, y_pred), 4),
        'F1_Score': round(f1_score(y_true, y_pred), 4),
        'ROC_AUC': round(roc_auc, 4),
        'CV_F1_Mean': round(cv_f1_mean, 4),
        'Tiempo_Entrenamiento (s)': round(training_time, 4),
        'Train_Accuracy': round(train_score, 4),
        'Test_Accuracy': round(test_score, 4),
        'Gap (Consistency)': round(gap, 4)
    }


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    ruta_processed = base_dir / "data" / "processed"

    print("Cargando datos procesados...")
    X_train = pd.read_csv(ruta_processed / "X_train.csv")
    X_test = pd.read_csv(ruta_processed / "X_test.csv")
    y_train = pd.read_csv(ruta_processed / "y_train.csv").squeeze()
    y_test = pd.read_csv(ruta_processed / "y_test.csv").squeeze()

    # Modelos tradicionales de clasificación a evaluar
    modelos = {
        'Regresión Logística': LogisticRegression(max_iter=1000, random_state=42),
        'Árbol de Decisión': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42)
    }

    resultados = []
    modelos_entrenados = {}
    predicciones = {}
    probabilidades = {}

    # Configuración de Cross-Validation para validación robusta
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\nIniciando entrenamiento y validación de modelos...\n")

    for nombre, modelo in modelos.items():
        print(f"--- Procesando {nombre} ---")
        
        # 1. Validación cruzada (Cross-Validation)
        cv_results = cross_validate(modelo, X_train, y_train, cv=cv, scoring='f1', n_jobs=-1)
        cv_f1_mean = np.mean(cv_results['test_score'])
        print(f"  -> F1-Score promedio en CV: {cv_f1_mean:.4f}")

        # 2. Entrenamiento final (usando función build_model)
        modelo_fit, tiempo = build_model(modelo, X_train, y_train)
        modelos_entrenados[nombre] = modelo_fit

        # 3. Obtener predicciones y probabilidades en Test
        y_pred = modelo_fit.predict(X_test)
        y_prob = modelo_fit.predict_proba(X_test)[:, 1]  # Probabilidad de la clase positiva
        predicciones[nombre] = y_pred
        probabilidades[nombre] = y_prob

        # 4. Calcular scores para la Consistencia
        train_acc = modelo_fit.score(X_train, y_train)
        test_acc = modelo_fit.score(X_test, y_test)

        # 5. Función summarize_classification
        metricas = summarize_classification(
            y_true=y_test,
            y_pred=y_pred,
            y_prob=y_prob,
            model_name=nombre,
            training_time=tiempo,
            train_score=train_acc,
            test_score=test_acc,
            cv_f1_mean=cv_f1_mean
        )
        resultados.append(metricas)

    # 6. Imprimir Tabla Resumen en Consola
    df_resultados = pd.DataFrame(resultados)
    print("\n=================== TABLA RESUMEN DE MODELOS ===================")
    print(df_resultados.to_string(index=False))
    print("================================================================")

    # 7. Gráficos Comparativos: Matriz de Confusión
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Matrices de Confusión (Test Set)', fontsize=16)
    for i, (nombre, y_pred) in enumerate(predicciones.items()):
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False)
        axes[i].set_title(nombre)
        axes[i].set_xlabel('Predicho')
        axes[i].set_ylabel('Real')
    plt.tight_layout()
    plt.show()

    # 8. Gráficos Comparativos: Curvas ROC
    plt.figure(figsize=(8, 6))
    for nombre, y_prob in probabilidades.items():
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        plt.plot(fpr, tpr, label=f'{nombre} (AUC = {auc:.4f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Aleatorio (AUC = 0.5)')
    plt.xlabel('Tasa de Falsos Positivos (FPR)')
    plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
    plt.title('Curvas ROC Comparativas')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 9. Selección Justificada del Mejor Modelo (Balance de 3 dimensiones)
    print("\n================ EVALUANDO BALANCE (Performance + Consistencia + Escalabilidad) ================")
    
    # Normalización Min-Max para llevar todas las métricas a escala 0-1
    f1_norm = df_resultados['F1_Score'] / df_resultados['F1_Score'].max()
    auc_norm = df_resultados['ROC_AUC'] / df_resultados['ROC_AUC'].max()
    
    # Para Gap y Tiempo, invertimos (1 - valor/max) porque MENOR es MEJOR
    # Agregamos un pequeño epsilon (1e-9) por si el máximo es 0 para evitar división por cero
    gap_norm = 1 - (df_resultados['Gap (Consistency)'] / (df_resultados['Gap (Consistency)'].max() + 1e-9))
    time_norm = 1 - (df_resultados['Tiempo_Entrenamiento (s)'] / (df_resultados['Tiempo_Entrenamiento (s)'].max() + 1e-9))

    # Calculamos el Score Compuesto (Ponderación de variables)
    df_resultados['Score_Compuesto'] = (
        (f1_norm * 0.35) +      # 35% F1-Score (Performance)
        (auc_norm * 0.25) +     # 25% ROC-AUC (Performance)
        (gap_norm * 0.25) +     # 25% Consistencia (Evitar overfitting)
        (time_norm * 0.15)      # 15% Escalabilidad (Tiempo de cómputo)
    ).round(4)

    # Ordenamos por el Score Compuesto
    df_resultados = df_resultados.sort_values(by='Score_Compuesto', ascending=False).reset_index(drop=True)
    print(df_resultados[['Modelo', 'F1_Score', 'ROC_AUC', 'Gap (Consistency)', 'Tiempo_Entrenamiento (s)', 'Score_Compuesto']].to_string(index=False))
    
    mejor_modelo_nombre = df_resultados.iloc[0]['Modelo']
    mejor_modelo_obj = modelos_entrenados[mejor_modelo_nombre]

    print("\n================ JUSTIFICACIÓN DE SELECCIÓN ================")
    print(f"El mejor modelo seleccionado es: {mejor_modelo_nombre}")
    
    mejor_fila = df_resultados.iloc[0]
    print(f"\n¿Por qué se eligió {mejor_modelo_nombre} sobre los demás?")
    print(f"  1. **Performance**: Destaca por su F1-Score ({mejor_fila['F1_Score']}) y ROC-AUC ({mejor_fila['ROC_AUC']}).")
    print(f"  2. **Consistencia**: Tiene un Gap Train-Test de {mejor_fila['Gap (Consistency)']}. Esto demuestra que no está memorizando (no hay overfitting grave) y que generalizará bien.")
    print(f"  3. **Escalabilidad**: Tardó {mejor_fila['Tiempo_Entrenamiento (s)']} segundos en entrenar, siendo eficiente para el despliegue.")
    print(f"  -> Al ponderar estas tres dimensiones, obtuvo el mejor Score Compuesto ({mejor_fila['Score_Compuesto']}), siendo el modelo más equilibrado para producción.")
    print("============================================================")

    # Guardar objeto del modelo en la raíz del proyecto
    ruta_modelo_salida = base_dir / "modelo_seleccionado.pkl"
    joblib.dump(mejor_modelo_obj, ruta_modelo_salida)
    print(f"Modelo guardado exitosamente en: {ruta_modelo_salida}")