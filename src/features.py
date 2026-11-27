def agregar_mismo_estado(df):
    df = df.copy()
    df["mismo_estado"] = (df["customer_state"] == df["estado_vendedor"]).astype(int)
    return df