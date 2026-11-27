import os
import numpy as np
import joblib
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.data_processing import cargar_datos, preparar_datos, dividir_datos
from src.evaluation import evaluar, captura_por_riesgo

COLS_NUM = ["dias_prometidos", "peso_total_g", "flete_total"]
COLS_CAT = ["customer_state", "estado_vendedor", "categoria"]
COLS_BIN = ["mismo_estado"]


def crear_preprocesador():
    return ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), COLS_NUM),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="desconocido")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=100)),
        ]), COLS_CAT),
        ("bin", "passthrough", COLS_BIN),
    ], sparse_threshold=0)


def crear_pipeline_xgb():
    modelo = XGBClassifier(
        learning_rate=0.05, max_depth=6, min_child_weight=1,
        n_estimators=400, random_state=42, eval_metric="logloss",
    )
    return Pipeline([("pre", crear_preprocesador()), ("modelo", modelo)])


def crear_pipeline_no_supervisado():
    pre = ColumnTransformer([
        ("cola_larga", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("log1p", FunctionTransformer(np.log1p)),
            ("scaler", StandardScaler()),
        ]), ["peso_total_g", "flete_total"]),
        ("dias", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), ["dias_prometidos"]),
    ])
    return Pipeline([
        ("pre", pre),
        ("pca", PCA(n_components=2)),
        ("kmeans", KMeans(n_clusters=5, random_state=42, n_init=10)),
    ])


def entrenar_modelo_final():
    df = preparar_datos(cargar_datos())
    X_train, X_test, y_train, y_test = dividir_datos(df)

    pipe = crear_pipeline_xgb()
    pipe.fit(X_train, y_train)

    proba = evaluar(pipe, X_test, y_test)
    captura_por_riesgo(y_test, proba)

    os.makedirs("models", exist_ok=True)
    joblib.dump(pipe, "models/final_model.pkl")


def entrenar_no_supervisado():
    df = preparar_datos(cargar_datos())
    X_train, X_test, y_train, y_test = dividir_datos(df)

    pipe = crear_pipeline_no_supervisado()
    pipe.fit(X_train)

    os.makedirs("models", exist_ok=True)
    joblib.dump(pipe, "models/pipeline_no_supervisado.pkl")


if __name__ == "__main__":
    entrenar_modelo_final()
    entrenar_no_supervisado()