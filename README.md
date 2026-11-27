# Prediccion de pedidos que llegan tarde (Olist)

## Objetivo

Predecir que pedidos de un marketplace van a llegar despues del dia prometido al cliente, para poder avisarle o mejorar ese envio antes de que ocurra el retraso.

El problema es de clasificacion: el target es `tarde_dia` (1 si el pedido se entrego despues del dia prometido, sin contar la hora).

## Datos

- Fuente: Brazilian E-Commerce Public Dataset by Olist (Kaggle), pedidos de un marketplace de Brasil.
- Se usan 96.470 pedidos entregados. Solo el 6,8 % llego tarde, asi que las clases estan muy desbalanceadas.
- Archivo procesado: `data/processed/pedidos_modelo.csv`.
- Variables del modelo (todas se conocen en el momento de la compra):

| Variable | Tipo | Descripcion |
|---|---|---|
| `dias_prometidos` | numerica | Dias entre la compra y la fecha estimada de entrega |
| `peso_total_g` | numerica | Peso total del pedido en gramos |
| `flete_total` | numerica | Costo del flete |
| `customer_state` | categorica | Estado del cliente |
| `estado_vendedor` | categorica | Estado del vendedor |
| `categoria` | categorica | Categoria del producto |
| `mismo_estado` | binaria | 1 si vendedor y cliente estan en el mismo estado |

## Instalacion

```bash
python3 -m pip install -r requirements.txt
```

En Mac, XGBoost necesita ademas: `brew install libomp`.

## Ejecucion

Desde la raiz del repositorio:

```bash
# Entrenar y guardar los modelos en models/
python3 -m src.training

# Abrir la demo
python3 -m streamlit run app_streamlit/app.py
```

La demo carga `models/final_model.pkl`. El analisis completo esta en los notebooks de la carpeta `notebooks/`: fuentes, limpieza y EDA, y entrenamiento y evaluacion.

## Metodo y evaluacion

**Preparacion.** Division 80 % train y 20 % test, estratificada, con `random_state=42`. El preprocesado va dentro de un pipeline: mediana y `StandardScaler` en las numericas; valor "desconocido" y `OneHotEncoder` en las categoricas; `mismo_estado` sin cambios. Todo se ajusta solo con train.

**Metrica.** Como solo el 6,8 % llega tarde, la metrica principal es PR-AUC (area bajo la curva precision-recall). El baseline (siempre predecir la clase mas comun) da PR-AUC 0,068 y ROC-AUC 0,5.

**Modelos supervisados.** Se compararon 5 modelos con validacion cruzada de 5 folds estratificados, sobre train:

| Modelo | ROC-AUC | PR-AUC |
|---|---|---|
| XGBoost | 0,710 | 0,165 |
| HistGradientBoosting | 0,707 | 0,162 |
| Random Forest | 0,692 | 0,145 |
| Regresion logistica | 0,693 | 0,143 |
| Arbol de decision | 0,671 | 0,118 |

Se uso GridSearchCV con `average_precision` para XGBoost y HistGradientBoosting, sin mejora real. Modelo final: XGBoost (`learning_rate=0.05`, `max_depth=6`, `min_child_weight=1`, `n_estimators=400`), porque tiene el mejor PR-AUC en validacion, aunque queda casi empatado con HistGradientBoosting.

**Estadistica.** Test chi-cuadrado entre `customer_state` y `tarde_dia`: chi-cuadrado = 1785,8 con 26 grados de libertad, p < 0,001 y V de Cramer = 0,14 (relacion real pero moderada). Para las metricas finales se calcularon intervalos de confianza por bootstrap (ver Resultados).

**Modelo no supervisado.** PCA con 2 componentes (explican el 87 % de la varianza de peso, flete y dias prometidos) y KMeans con 5 grupos (silhouette ~0,35). Se uso para explorar si hay tipos de pedido con mas retrasos.

## Resultados y limitaciones

**Resultados en test** (modelo final XGBoost, 19.294 pedidos):

| | ROC-AUC | PR-AUC |
|---|---|---|
| Baseline | 0,5 | 0,068 |
| XGBoost | 0,725 | 0,168 |

Intervalos de confianza al 95 % (bootstrap con 1.000 remuestreos del test, sin reentrenar): ROC-AUC entre 0,711 y 0,740, PR-AUC entre 0,152 y 0,186, y el 10 % mas riesgoso detecta entre 27,5 % y 32,0 % de los retrasos. Ningun intervalo incluye el valor del azar, asi que el modelo es mejor que adivinar.

El PR-AUC es unas 2,5 veces mejor que el azar. Si se revisa solo el grupo de pedidos con mas riesgo:

| Se revisa el... | Retrasos detectados | Veces mejor que el azar |
|---|---|---|
| 5 % mas riesgoso | 16,8 % | 3,4 |
| 10 % mas riesgoso | 29,8 % | 3,0 |
| 20 % mas riesgoso | 47,4 % | 2,4 |
| 30 % mas riesgoso | 59,8 % | 2,0 |

Hallazgos del analisis: los pedidos al nordeste de Brasil llegan tarde mucho mas seguido (~21 % contra ~4,5 % en otros estados), y los pedidos donde vendedor y cliente estan en el mismo estado llegan tarde menos (4,5 % contra 8,1 %).

**Limitaciones.**
- El modelo sirve para priorizar pedidos con riesgo, no para asegurar que uno va a llegar tarde: en el 10 % mas riesgoso, la tasa real de retrasos ronda el 20 %.
- Solo usa 7 variables. Los 5 modelos rinden parecido, lo que sugiere que el limite esta en la informacion disponible y no en el algoritmo. Variables nuevas (por ejemplo la distancia) podrian mejorarlo.
- Los grupos del KMeans muestran una diferencia debil (4,9 % a 8,6 % de retrasos en train) que se reduce en test (5,5 % a 8,0 %). Es un resultado exploratorio, no un hallazgo firme.
- Los datos son de un solo marketplace de Brasil; no se puede asumir que sirvan para otros paises o empresas.

## Estructura del proyecto

```
.
|-- app_streamlit/     # demo en Streamlit
|-- data/processed/    # datos listos para el modelo
|-- docs/              # memoria y graficas
|-- models/            # modelos guardados (.pkl)
|-- notebooks/         # analisis: fuentes, EDA, entrenamiento
|-- src/
|   |-- data_processing.py   # carga y division de datos
|   |-- training.py          # pipelines y entrenamiento
|   `-- evaluation.py        # metricas
|-- requirements.txt
`-- README.md

