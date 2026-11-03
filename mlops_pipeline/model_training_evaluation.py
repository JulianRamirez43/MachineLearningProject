import time
from pathlib import Path
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def build_model(model, X_train, y_train):
    """
    Entrena el modelo y registra el tiempo de entrenamiento (Scalability).
    """
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    return model, training_time


def summarize_classification(y_true, y_pred, model_name, training_time, train_score, test_score):
    """
    Calcula métricas de desempeño (Performance) y evalúa el sobreajuste (Consistency).
    """
    gap = train_score - test_score  # Brecha entre Train y Test para medir inconsistencia/overfitting
    return {
        'Modelo': model_name,
        'Accuracy': round(accuracy_score(y_true, y_pred), 4),
        'Precision': round(precision_score(y_true, y_pred), 4),
        'Recall': round(recall_score(y_true, y_pred), 4),
        'F1_Score': round(f1_score(y_true, y_pred), 4),
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

    # Proceso de entrenamiento y evaluación comparativa
    for nombre, modelo in modelos.items():
        # 1. Función build_model
        modelo_fit, tiempo = build_model(modelo, X_train, y_train)
        modelos_entrenados[nombre] = modelo_fit

        # 2. Obtener predicciones en Test
        y_pred = modelo_fit.predict(X_test)

        # 3. Calcular scores para la Consistency
        train_acc = modelo_fit.score(X_train, y_train)
        test_acc = modelo_fit.score(X_test, y_test)

        # 4. Función summarize_classification
        metricas = summarize_classification(
            y_true=y_test,
            y_pred=y_pred,
            model_name=nombre,
            training_time=tiempo,
            train_score=train_acc,
            test_score=test_acc
        )
        resultados.append(metricas)

    # 5. Imprimir Tabla Resumen en Consola
    df_resultados = pd.DataFrame(resultados)
    print("\n=================== TABLA RESUMEN DE MODELOS ===================")
    print(df_resultados.to_string(index=False))
    print("================================================================")

    # 6. Gráfico Comparativo (F1-Score)
    plt.figure(figsize=(9, 5))
    barras = plt.bar(df_resultados['Modelo'], df_resultados['F1_Score'], color=['#1f77b4', '#2ca02c', '#ff7f0e'])
    plt.title('Comparación de Performance por Modelo (F1-Score)', fontsize=14)
    plt.xlabel('Modelo', fontsize=12)
    plt.ylabel('F1-Score', fontsize=12)
    plt.ylim(0, 1.1)

    for barra in barras:
        yval = barra.get_height()
        plt.text(barra.get_x() + barra.get_width()/2.0, yval + 0.02, str(yval), ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.show()

    # 7. Selección del mejor modelo (ordenado por F1-Score)
    mejor_modelo_nombre = df_resultados.sort_values(by='F1_Score', ascending=False).iloc[0]['Modelo']
    mejor_modelo_obj = modelos_entrenados[mejor_modelo_nombre]

    print(f"\nEl mejor modelo seleccionado es: {mejor_modelo_nombre}")

    # Guardar objeto del modelo en la raíz del proyecto
    ruta_modelo_salida = base_dir / "modelo_seleccionado.pkl"
    joblib.dump(mejor_modelo_obj, ruta_modelo_salida)
    print(f"Modelo guardado exitosamente en: {ruta_modelo_salida}")