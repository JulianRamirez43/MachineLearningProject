import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

def preprocess_and_split(df):
    """
    Realiza la limpieza inicial, divide los datos y aplica el ColumnTransformer.
    """
    # 1. Limpieza inicial (eliminación de variables redundantes/fuga identificadas en EDA)
    columnas_a_eliminar = [
        'puntaje', 'creditos_sectorFinanciero', 'cuota_pactada', 
        'saldo_principal', 'saldo_mora_codeudor'
    ]
    df = df.drop(columns=columnas_a_eliminar, errors='ignore')

    # 2. Eliminar nulos marginales en puntaje_datacredito
    df = df.dropna(subset=['puntaje_datacredito'])

    # 3. Binarizar saldo_mora (0 si es nulo o 0, 1 si es mayor a 0)
    df['saldo_mora'] = df['saldo_mora'].fillna(0)
    df['saldo_mora'] = (df['saldo_mora'] > 0).astype(int)

    # 4. Manejo de fecha: extraer mes y eliminar la columna original
    if 'fecha_prestamo' in df.columns:
        df['fecha_prestamo'] = pd.to_datetime(df['fecha_prestamo'])
        df['mes_prestamo'] = df['fecha_prestamo'].dt.month
        df = df.drop(columns=['fecha_prestamo'])

    # 5. Limpieza de tendencia_ingresos (Convertir números atípicos a NaN para que SimpleImputer los reemplace por la moda)
    if 'tendencia_ingresos' in df.columns:
        categorias_validas = ['Creciente', 'Decreciente', 'Estable']
        df.loc[~df['tendencia_ingresos'].isin(categorias_validas), 'tendencia_ingresos'] = np.nan

    # 6. Manejo de tipo_credito (si viene sin codificar desde el Excel crudo)
    if 'tipo_credito' in df.columns:
        conteos = df['tipo_credito'].value_counts()
        categorias_minoritarias = conteos[conteos < 50].index
        df['tipo_credito'] = df['tipo_credito'].replace(categorias_minoritarias, 'Otros')
        df = pd.get_dummies(df, columns=['tipo_credito'], dtype=int)

    # 7. Definir variables predictoras (X) y objetivo (y)
    X = df.drop(columns=['Pago_atiempo'])
    y = df['Pago_atiempo']

    # 8. Dividir en Entrenamiento y Prueba (80% / 20%) manteniendo proporción con stratify
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 9. Definir los grupos de columnas para el ColumnTransformer
    numeric_features = [
        'capital_prestado', 'plazo_meses', 'edad_cliente', 'salario_cliente', 
        'total_otros_prestamos', 'puntaje_datacredito', 'cant_creditosvigentes', 
        'huella_consulta', 'saldo_mora', 'saldo_total', 'creditos_sectorCooperativo', 
        'creditos_sectorReal', 'promedio_ingresos_datacredito', 'mes_prestamo'
    ]
    categoric_features = ['tipo_laboral']
    categoric_ordinal_features = ['tendencia_ingresos']

    # Pipeline Numérico: Imputación por la media
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean'))
    ])

    # Pipeline Categórico Nominal: Imputación por moda + OneHotEncoder
    categoric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Pipeline Categórico Ordinal: Imputación por moda + OrdinalEncoder
    orden_tendencia = [['Decreciente', 'Estable', 'Creciente']]
    ordinal_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ordinal', OrdinalEncoder(categories=orden_tendencia))
    ])

    # Ensamblar ColumnTransformer según rúbrica
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categoric_transformer, categoric_features),
            ('ord', ordinal_transformer, categoric_ordinal_features)
        ],
        remainder='passthrough'
    )

    # Ajustar y transformar X_train; solo transformar X_test (evita Data Leakage)
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    # Obtener nombres finales de las columnas
    cat_onehot_cols = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categoric_features).tolist()
    remainder_cols = [col for col in X.columns if col not in numeric_features + categoric_features + categoric_ordinal_features]
    column_names = numeric_features + cat_onehot_cols + categoric_ordinal_features + remainder_cols

    # Convertir a DataFrames de Pandas
    X_train_df = pd.DataFrame(X_train_trans, columns=column_names, index=X_train.index)
    X_test_df = pd.DataFrame(X_test_trans, columns=column_names, index=X_test.index)

    return X_train_df, X_test_df, y_train, y_test, preprocessor


if __name__ == "__main__":
    # Definir rutas relativas dinámicas desde la raíz del proyecto
    base_dir = Path(__file__).resolve().parent.parent
    ruta_raw = base_dir / "data" / "raw" / "base_datos.xlsx"
    ruta_processed = base_dir / "data" / "processed"

    print(f"Cargando datos crudos desde: {ruta_raw}")
    df_raw = pd.read_excel(ruta_raw)

    X_train, X_test, y_train, y_test, preprocessor = preprocess_and_split(df_raw)

    # Crear la carpeta data/processed si aún no existe
    ruta_processed.mkdir(parents=True, exist_ok=True)

    # Exportar los datos limpios y transformados
    X_train.to_csv(ruta_processed / "X_train.csv", index=False)
    X_test.to_csv(ruta_processed / "X_test.csv", index=False)
    y_train.to_csv(ruta_processed / "y_train.csv", index=False)
    y_test.to_csv(ruta_processed / "y_test.csv", index=False)

    print("¡Ingeniería de características completada con éxito!")
    print(f"Archivos procesados guardados en: {ruta_processed}")