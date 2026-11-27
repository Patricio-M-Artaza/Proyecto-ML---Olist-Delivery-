import pandas as pd
from sklearn.model_selection import train_test_split

RUTA_DATOS = "data/processed/pedidos_modelo.csv"
TARGET = "tarde_dia"
FEATURES = ["dias_prometidos", "peso_total_g", "flete_total",
            "customer_state", "estado_vendedor", "categoria", "mismo_estado"]


def cargar_datos(ruta=RUTA_DATOS):
    return pd.read_csv(ruta)


def preparar_datos(df):
    df = df.copy()
    df["mismo_estado"] = (df["customer_state"] == df["estado_vendedor"]).astype(int)
    return df


def dividir_datos(df):
    X = df[FEATURES]
    y = df[TARGET]
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)