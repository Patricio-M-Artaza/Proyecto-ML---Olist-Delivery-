NOMBRES_ESTADOS = {
    "AC": "Acre", "AL": "Alagoas", "AP": "Amapá", "AM": "Amazonas",
    "BA": "Bahia", "CE": "Ceará", "DF": "Distrito Federal",
    "ES": "Espírito Santo", "GO": "Goiás", "MA": "Maranhão",
    "MT": "Mato Grosso", "MS": "Mato Grosso do Sul", "MG": "Minas Gerais",
    "PA": "Pará", "PB": "Paraíba", "PR": "Paraná", "PE": "Pernambuco",
    "PI": "Piauí", "RJ": "Rio de Janeiro", "RN": "Rio Grande do Norte",
    "RS": "Rio Grande do Sul", "RO": "Rondônia", "RR": "Roraima",
    "SC": "Santa Catarina", "SP": "São Paulo", "SE": "Sergipe",
    "TO": "Tocantins",
}


def mostrar_estado(codigo):
    return f"{NOMBRES_ESTADOS.get(codigo, codigo)} ({codigo})"

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Riesgo de retraso en pedidos", page_icon="📦")


@st.cache_resource
def cargar_modelo():
    return joblib.load("models/final_model.pkl")


@st.cache_data
def cargar_opciones():
    df = pd.read_csv(
        "data/processed/pedidos_modelo.csv",
        usecols=["customer_state", "estado_vendedor", "categoria"],
    )
    return (
        sorted(df["customer_state"].dropna().unique()),
        sorted(df["estado_vendedor"].dropna().unique()),
        sorted(df["categoria"].dropna().unique()),
    )


modelo = cargar_modelo()
estados_cliente, estados_vendedor, categorias = cargar_opciones()

st.title("📦 ¿Va a llegar tarde este pedido?")
st.write(
    "Completá los datos del pedido y el modelo estima la probabilidad "
    "de que llegue después del día prometido."
)

col1, col2 = st.columns(2)
with col1:
    customer_state = st.selectbox("Estado del cliente (Brasil)", estados_cliente, format_func=mostrar_estado)
    estado_vendedor = st.selectbox("Estado del vendedor (Brasil)", estados_vendedor, format_func=mostrar_estado)
    categoria = st.selectbox("Categoría del producto", categorias)
with col2:
    dias_prometidos = st.number_input("Días prometidos de entrega", min_value=1, max_value=200, value=23)
    peso_total_g = st.number_input("Peso total (g)", min_value=0, value=1000)
    flete_total = st.number_input("Flete (R$)", min_value=0.0, value=20.0)

if st.button("Calcular riesgo"):
    pedido = pd.DataFrame([{
        "dias_prometidos": dias_prometidos,
        "peso_total_g": peso_total_g,
        "flete_total": flete_total,
        "customer_state": customer_state,
        "estado_vendedor": estado_vendedor,
        "categoria": categoria,
        "mismo_estado": int(customer_state == estado_vendedor),
    }])
    prob = modelo.predict_proba(pedido)[0, 1]

    st.metric("Probabilidad de llegar tarde", f"{prob:.1%}")
    st.caption("Los datos son de Olist, un marketplace de Brasil: los estados son los 27 estados brasileños.")
    st.write(f"Es {prob / 0.068:.1f} veces el promedio general (6,8 %).")
    st.caption(
        "El modelo sirve para priorizar pedidos con más riesgo, "
        "no para asegurar que un pedido va a llegar tarde."
    )