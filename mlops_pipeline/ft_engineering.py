import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


# ---------------------------------------------------------------------------
# Reporte de calidad (equivale a las secciones 1.1, 2 y 4 del EDA).
# Solo informa; no modifica el DataFrame.
# ---------------------------------------------------------------------------
def reporte_calidad(df):
    """Imprime duplicados, balance de la variable objetivo, nulos y reglas de validación."""
    print("=" * 60)
    print("REPORTE DE CALIDAD DE DATOS")
    print("=" * 60)

    # Duplicados (solo se reportan, no se eliminan)
    n_dup = df.duplicated().sum()
    print(f"\nFilas duplicadas exactas: {n_dup} ({n_dup / len(df) * 100:.2f}%)")

    # Distribución de la variable objetivo
    if 'Pago_atiempo' in df.columns:
        conteo = df['Pago_atiempo'].value_counts().sort_index()
        pct = (df['Pago_atiempo'].value_counts(normalize=True).sort_index() * 100).round(2)
        print("\nDistribución de Pago_atiempo:")
        print(pd.DataFrame({'registros': conteo, 'porcentaje (%)': pct}))
        print(f"Razón clase mayoritaria / minoritaria: {conteo.max() / conteo.min():.1f} a 1")

    # Nulos
    nulos = df.isnull().sum()
    print("\nNulos por columna (solo las que tienen):")
    print(nulos[nulos > 0] if (nulos > 0).any() else "Ninguna")

    # Reglas de validación
    fechas = pd.to_datetime(df['fecha_prestamo'], errors='coerce') if 'fecha_prestamo' in df.columns else None
    reglas = {
        'salario_cliente < 0': (df['salario_cliente'] < 0).sum(),
        'salario_cliente == 0': (df['salario_cliente'] == 0).sum(),
        'edad_cliente fuera de 18-100': (~df['edad_cliente'].between(18, 100)).sum(),
        'capital_prestado <= 0': (df['capital_prestado'] <= 0).sum(),
        'plazo_meses <= 0': (df['plazo_meses'] <= 0).sum(),
        'total_otros_prestamos < 0': (df['total_otros_prestamos'] < 0).sum(),
        'cant_creditosvigentes < 0': (df['cant_creditosvigentes'] < 0).sum(),
        'huella_consulta < 0': (df['huella_consulta'] < 0).sum(),
    }
    if fechas is not None:
        reglas['fecha_prestamo nula (NaT)'] = fechas.isna().sum()
        reglas['fecha_prestamo en el futuro'] = (fechas > pd.Timestamp.today()).sum()

    resultado = pd.DataFrame({'registros que incumplen': pd.Series(reglas)})
    resultado['% del total'] = (resultado['registros que incumplen'] / len(df) * 100).round(3)
    print("\nReglas de validación:")
    print(resultado)
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Limpieza + split + ColumnTransformer
# ---------------------------------------------------------------------------
def preprocess_and_split(df, verbose=False):
    """
    Replica la limpieza del EDA, divide los datos y aplica el ColumnTransformer.
    La imputación por media/moda se hace dentro del pipeline (ajustada solo con
    train) para evitar Data Leakage.
    """
    if verbose:
        reporte_calidad(df)

    df = df.copy()

    # 1. Eliminar variables con alta multicolinealidad / posible fuga (EDA sección 2)
    columnas_a_eliminar = [
        'puntaje', 'creditos_sectorFinanciero', 'cuota_pactada',
        'saldo_principal', 'saldo_mora_codeudor'
    ]
    df = df.drop(columns=columnas_a_eliminar, errors='ignore')

    # 2. puntaje_datacredito: eliminar los pocos nulos (0 es un puntaje válido, no se puede
    #    imputar con 0) y convertir a entero (no hay decimales reales)
    df = df.dropna(subset=['puntaje_datacredito'])
    df['puntaje_datacredito'] = df['puntaje_datacredito'].astype(int)

    # 3. saldo_mora: nulos -> 0 y binarizar (1 si hay saldo en mora, 0 si no)
    df['saldo_mora'] = df['saldo_mora'].fillna(0)
    df['saldo_mora'] = (df['saldo_mora'] > 0).astype(int)

    # 4. tendencia_ingresos: valores atípicos (numéricos) -> NaN; el imputer los
    #    reemplaza por la moda dentro del pipeline
    if 'tendencia_ingresos' in df.columns:
        categorias_validas = ['Creciente', 'Decreciente', 'Estable']
        df.loc[~df['tendencia_ingresos'].isin(categorias_validas), 'tendencia_ingresos'] = np.nan

    # 5. fecha_prestamo: convertir (errors='coerce' como en el EDA), extraer el mes
    #    y eliminar la columna original
    if 'fecha_prestamo' in df.columns:
        df['fecha_prestamo'] = pd.to_datetime(df['fecha_prestamo'], errors='coerce')
        df['mes_prestamo'] = df['fecha_prestamo'].dt.month
        df = df.drop(columns=['fecha_prestamo'])

    # 6. tipo_credito: agrupar categorías con menos de 50 registros en 'Otros' y aplicar dummies
    if 'tipo_credito' in df.columns:
        conteos = df['tipo_credito'].value_counts()
        categorias_minoritarias = conteos[conteos < 50].index
        df['tipo_credito'] = df['tipo_credito'].replace(categorias_minoritarias, 'Otros')
        df = pd.get_dummies(df, columns=['tipo_credito'], dtype=int)

    # 7. Regla de validación: conservar solo edades lógicas (18 a 100)
    df = df[df['edad_cliente'].between(18, 100)].copy()

    # 8. Atributo derivado: ratio deuda / ingreso (evita división por cero)
    df['ratio_deuda_ingreso'] = np.where(
        df['salario_cliente'] > 0,
        df['total_otros_prestamos'] / df['salario_cliente'],
        0
    )

    # NUEVO: conservar copia del dataset limpio (antes del split)
    df_limpio = df.copy()

    # 9. Variables predictoras (X) y objetivo (y)
    X = df.drop(columns=['Pago_atiempo'])
    y = df['Pago_atiempo']

    # 10. Split 80/20 estratificado (hay desbalance de clases)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 11. Grupos de columnas para el ColumnTransformer
    numeric_features = [
        'capital_prestado', 'plazo_meses', 'edad_cliente', 'salario_cliente',
        'total_otros_prestamos', 'puntaje_datacredito', 'cant_creditosvigentes',
        'huella_consulta', 'saldo_mora', 'saldo_total', 'creditos_sectorCooperativo',
        'creditos_sectorReal', 'promedio_ingresos_datacredito', 'mes_prestamo',
        'ratio_deuda_ingreso'
    ]
    categoric_features = ['tipo_laboral']
    categoric_ordinal_features = ['tendencia_ingresos']

    # Numérico: imputación por la media (saldo_total, promedio_ingresos_datacredito, etc.)
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean'))
    ])

    # Categórico nominal: moda + OneHotEncoder
    categoric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Categórico ordinal: moda + OrdinalEncoder
    orden_tendencia = [['Decreciente', 'Estable', 'Creciente']]
    ordinal_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ordinal', OrdinalEncoder(categories=orden_tendencia))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categoric_transformer, categoric_features),
            ('ord', ordinal_transformer, categoric_ordinal_features)
        ],
        remainder='passthrough'  # dummies de tipo_credito
    )

    # fit solo con train; transform en test (evita Data Leakage)
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    # Nombres finales de columnas
    cat_onehot_cols = (
        preprocessor.named_transformers_['cat']
        .named_steps['onehot']
        .get_feature_names_out(categoric_features)
        .tolist()
    )
    remainder_cols = [
        col for col in X.columns
        if col not in numeric_features + categoric_features + categoric_ordinal_features
    ]
    column_names = numeric_features + cat_onehot_cols + categoric_ordinal_features + remainder_cols

    X_train_df = pd.DataFrame(X_train_trans, columns=column_names, index=X_train.index)
    X_test_df = pd.DataFrame(X_test_trans, columns=column_names, index=X_test.index)

    return X_train_df, X_test_df, y_train, y_test, preprocessor, df_limpio


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    ruta_raw = base_dir / "data" / "raw" / "base_datos.xlsx"
    ruta_processed = base_dir / "data" / "processed"

    print(f"Cargando datos crudos desde: {ruta_raw}")
    df_raw = pd.read_excel(ruta_raw)

    X_train, X_test, y_train, y_test, preprocessor, df_limpio = preprocess_and_split(df_raw, verbose=True)

    ruta_processed.mkdir(parents=True, exist_ok=True)

    # >>> NUEVO: guardar la base limpia completa <<<
    df_limpio.to_csv(ruta_processed / "base_de_datos.csv", index=False)


    X_train.to_csv(ruta_processed / "X_train.csv", index=False)
    X_test.to_csv(ruta_processed / "X_test.csv", index=False)
    y_train.to_csv(ruta_processed / "y_train.csv", index=False)
    y_test.to_csv(ruta_processed / "y_test.csv", index=False)

    print(f"X_train: {X_train.shape} | X_test: {X_test.shape}")
    print("¡Ingeniería de características completada con éxito!")
    print(f"Archivos procesados guardados en: {ruta_processed}")