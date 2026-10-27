import pandas as pd

# 1. Cargar la base de datos original desde la carpeta raw
ruta_entrada = '../data/raw/base_datos.xlsx'
df = pd.read_excel(ruta_entrada)

# 2. Aplicar las transformaciones de ingeniería de características
conteos = df['tipo_credito'].value_counts()
umbral = 50 
categorias_minoritarias = conteos[conteos < umbral].index
df['tipo_credito'] = df['tipo_credito'].replace(categorias_minoritarias, 'Otros')
df['tipo_credito'] = df['tipo_credito'].astype(str)

# 3. Guardar el resultado en la carpeta processed
ruta_salida = '../data/processed/base_de_datos.csv'
df.to_csv(ruta_salida, index=False)