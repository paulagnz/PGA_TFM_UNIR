"""
Este script calcula los errores RMSE, MAE y R² del modelo CatBoost mediante la 
validación cruzada temporal. Los resultados permiten comparar el rendimiento de 
este modelo con otros y seleccionar el mejor.

"""

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pandas.tseries.offsets import DateOffset


#Ingeniería de características
def variables_calendario(df: pd.DataFrame, columna_fecha: str) -> pd.DataFrame:
    df = df.copy()
    
    # Sacamos el año y el mes de la columna de fecha
    df["year"] = df[columna_fecha].dt.year
    df["month"]=df[columna_fecha].dt.month
    
    #Índice temporal de los datos para poder saber el orden de estos
    df["indice"]=np.arange(len(df)) 
    
    #Variables circulares 
    df["msin"] = np.sin(2*np.pi*df["month"]/12)
    df["mcos"] = np.cos(2*np.pi*df["month"]/12)
    
    #Variables estacionales
    df["post_covid"] =(df[columna_fecha]>="2021-01-01").astype(int)
    df["verano"] = df["month"].isin([6,7,8]).astype(int)
    df["fin_trimestre"] = df["month"].isin([3,6,9,12]).astype(int)
    df["diciembre"] = (df["month"]==12).astype(int)
    
    return df


#Carga de datos
df = pd.read_excel("Datos_16-25_Seg.xlsx")
df["Dia 1 Mes"] = pd.to_datetime(df["Dia 1 Mes"])
df = variables_calendario(df, "Dia 1 Mes")

df_agg = df.groupby(["Dia 1 Mes", "Segmento"]).agg({"Matricul.": "sum"}).reset_index()
df_agg = variables_calendario(df_agg, "Dia 1 Mes")


#variables
features = ["Segmento","year", "month","msin", "mcos",
    "post_covid", "verano", "fin_trimestre", "diciembre"]

X = df_agg[features]
y = df_agg["Matricul."]


#Validación cruzada
cat_features = ["Segmento"] #variable categórica

tscv = TimeSeriesSplit(n_splits=5)
rmse_list, mae_list, r2_list = [], [], []

for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model = CatBoostRegressor(
        iterations=300,
        learning_rate=0.05,
        depth=6,
        random_seed=42,
        verbose=0)

    model.fit(X_train, y_train, cat_features=cat_features)
    y_pred = model.predict(X_test)

    rmse_list.append(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae_list.append(mean_absolute_error(y_test, y_pred))
    r2_list.append(r2_score(y_test, y_pred))

print("Resultados de CatBoost:")
print(f"RMSE medio: {np.mean(rmse_list):.2f}")
print(f"MAE medio: {np.mean(mae_list):.2f}")
print(f"R² medio: {np.mean(r2_list):.3f}")
