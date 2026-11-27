import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def evaluar(pipe, X_test, y_test):
    proba = pipe.predict_proba(X_test)[:, 1]
    print("ROC-AUC:", round(roc_auc_score(y_test, proba), 3))
    print("PR-AUC:", round(average_precision_score(y_test, proba), 3))
    return proba


def captura_por_riesgo(y_test, proba, niveles=(0.05, 0.10, 0.20, 0.30)):
    y = np.asarray(y_test)
    orden = np.argsort(-proba)
    total_tardes = y.sum()
    for n in niveles:
        k = int(len(y) * n)
        captura = y[orden[:k]].sum() / total_tardes
        print(f"Top {n:.0%} más riesgoso: captura {captura:.1%} de las tardes ({captura / n:.1f} veces el azar)")