
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import ks_2samp, chi2_contingency, entropy

def calculate_psi(expected, actual, buckets=10):
    """Calcula el Population Stability Index (PSI)."""
    breakpoints = np.linspace(min(expected), max(expected), buckets + 1)
    expected_percents = np.histogram(expected, breakpoints)[0] / len(expected)
    actual_percents = np.histogram(actual, breakpoints)[0] / len(actual)

    # Evitar divisiones por cero
    expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
    actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)

    return np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))

def calculate_js(expected, actual, buckets=20):
    """Calcula la Divergencia de Jensen-Shannon."""
    breakpoints = np.linspace(min(expected.min(), actual.min()), max(expected.max(), actual.max()), buckets + 1)
    p = np.histogram(expected, breakpoints)[0] / len(expected)
    q = np.histogram(actual, breakpoints)[0] / len(actual)
    
    p, q = p / np.sum(p), q / np.sum(q)
    m = 0.5 * (p + q)
    return 0.5 * entropy(p, m) + 0.5 * entropy(q, m)

def calculate_drift_metrics(df_hist, df_act):
    """Calcula todas las métricas de drift por cada columna."""
    resultados = []
    for col in df_hist.columns:
        res = {'Variable': col}
        
        # Si es categórica
        if df_hist[col].dtype == 'object' or df_hist[col].nunique() < 5:
            res['Tipo'] = 'Categórica'
            contingency = pd.crosstab(df_hist[col], df_act[col])
            chi2, p_value, _, _ = chi2_contingency(contingency)
            res.update({'KS_pvalue': np.nan, 'PSI': np.nan, 'JS': np.nan, 'Chi2_pvalue': round(p_value, 4)})
            res['Drift'] = 'SÍ' if p_value < 0.05 else 'NO'
            
        # Si es numérica
        else:
            res['Tipo'] = 'Numérica'
            ks_stat, p_value = ks_2samp(df_hist[col].dropna(), df_act[col].dropna())
            psi = calculate_psi(df_hist[col].dropna().values, df_act[col].dropna().values)
            js = calculate_js(df_hist[col].dropna().values, df_act[col].dropna().values)
            res.update({'KS_pvalue': round(p_value, 4), 'PSI': round(psi, 4), 'JS': round(js, 4), 'Chi2_pvalue': np.nan})
            res['Drift'] = 'SÍ' if psi > 0.2 or p_value < 0.05 else 'NO'
            
        resultados.append(res)
    return pd.DataFrame(resultados)

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    ruta = base_dir / "data" / "processed"
    
    print("Cargando datos (Histórico vs Actual)...")
    df_hist = pd.read_csv(ruta / "X_train.csv")
    df_act = pd.read_csv(ruta / "X_test.csv")
    
    print("Calculando métricas de Drift...")
    df_drift = calculate_drift_metrics(df_hist, df_act)
    
    df_drift.to_csv(ruta / "drift_results.csv", index=False)
    print(f"Resultados guardados en: {ruta / 'drift_results.csv'}")
    print(df_drift.head())