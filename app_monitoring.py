import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="Monitoreo Data Drift", layout="wide")
st.title("📊 Panel de Monitoreo y Data Drift")

@st.cache_data
def load_data():
    base_dir = Path(__file__).resolve().parent
    ruta = base_dir / "data" / "processed"
    df_hist = pd.read_csv(ruta / "X_train.csv")
    df_act = pd.read_csv(ruta / "X_test.csv")
    df_drift = pd.read_csv(ruta / "drift_results.csv")
    return df_hist, df_act, df_drift

df_hist, df_act, df_drift = load_data()

# 1. SEMÁFORO
st.subheader("🚦 Estado General del Modelo")
vars_drift = len(df_drift[df_drift['Drift'] == 'SÍ'])
total_vars = len(df_drift)

c1, c2, c3 = st.columns(3)
c1.metric("Variables Analizadas", total_vars)
c2.metric("Variables con Drift", vars_drift)
if vars_drift > 0:
    c3.error(f"⚠️ ALERTA: {vars_drift} variables con Drift.")
    st.warning("Recomendación: Revisar variables y considerar reentrenamiento.")
else:
    c3.success("✅ Estable. Sin Drift significativo.")

# 2. GRÁFICOS COMPARATIVOS
st.subheader("📈 Análisis de Distribución por Variable")
var_sel = st.selectbox("Seleccione variable:", df_hist.columns)
metricas = df_drift[df_drift['Variable'] == var_sel].iloc[0]

col1, col2 = st.columns([2, 1])
with col1:
    if metricas['Tipo'] == 'Numérica':
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=df_hist[var_sel], name='Histórico', opacity=0.7, marker_color='blue', histnorm='probability density'))
        fig.add_trace(go.Histogram(x=df_act[var_sel], name='Actual', opacity=0.7, marker_color='red', histnorm='probability density'))
        fig.update_layout(barmode='overlay', title='Comparación Histórico vs Actual')
        st.plotly_chart(fig, use_container_width=True)
    else:
        h = df_hist[var_sel].value_counts(normalize=True).reset_index()
        a = df_act[var_sel].value_counts(normalize=True).reset_index()
        h.columns, a.columns = ['Cat', 'Hist'], ['Cat', 'Act']
        df_m = pd.merge(h, a, on='Cat', how='outer').fillna(0)
        fig = go.Figure(data=[
            go.Bar(name='Histórico', x=df_m['Cat'], y=df_m['Hist'], marker_color='blue'),
            go.Bar(name='Actual', x=df_m['Cat'], y=df_m['Act'], marker_color='red')
        ])
        fig.update_layout(barmode='group', title='Frecuencias Categóricas')
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("### Métricas")
    if metricas['Tipo'] == 'Numérica':
        st.metric("PSI", metricas['PSI'])
        st.metric("KS p-value", metricas['KS_pvalue'])
        st.metric("JS Div", metricas['JS'])
    else:
        st.metric("Chi2 p-value", metricas['Chi2_pvalue'])
    if metricas['Drift'] == 'SÍ':
        st.error("Drift Detectado")
    else:
        st.success("Variable Estable")

# 3. TABLA
st.subheader("📋 Tabla de Resultados")
def color_drift(val):
    return 'background-color: #ffcccc' if val == 'SÍ' else ''
st.dataframe(df_drift.style.applymap(color_drift, subset=['Drift']), use_container_width=True)

# 4. TENDENCIA TEMPORAL
st.subheader("🕰️ Tendencia del Drift en el Tiempo")
fechas = pd.date_range(end=pd.Timestamp.today(), periods=6, freq='ME')
psi_sim = np.linspace(metricas['PSI'] * 0.5, metricas['PSI'], 6) if metricas['Tipo'] == 'Numérica' else np.zeros(6)
df_t = pd.DataFrame({'Fecha': fechas, 'Métrica': psi_sim})
fig_t = px.line(df_t, x='Fecha', y='Métrica', markers=True, title='Evolución del Drift (Simulada)')
fig_t.add_hline(y=0.2, line_dash="dash", line_color="red", annotation_text="Umbral Crítico")
st.plotly_chart(fig_t, use_container_width=True)