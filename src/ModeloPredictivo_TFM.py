"""
Modelo predictivo de matriculaciones por segmento
----------------------------------------

Este script abre una interfaz gráfica que permite cargar un archivo Excel
con datos históricos de matriculaciones y guarda las predicciones
para los próximos 6 meses, separadas por segmento de modelo de coche.

He estructurado el script de manera modular para que, si en el futuro se quiere ampliar,
probar otros modelos o generar nuevas variables, sea más fácil hacerlo.

"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from tkinter import Tk, filedialog, messagebox
from tkinter.ttk import Button
from pandas.tseries.offsets import DateOffset

#Ingeniería de características
def variables_calendario(df: pd.DataFrame, columna_fecha: str) -> pd.DataFrame:
    """
    
    En esta función se aplica la ingeniería de variables explicada en el apartado 5.3 del trabajo de final de máster.
    
    Genera variables relacionadas con los datos para que el modelo pueda predecir mejor las matriculaciones, 
    es decir, se usa para enriquecer el dataset antes de entrenar el modelo de predicción.
    
    """
    df = df.copy()
    
    # Sacamos el año y el mes de la columna de fecha
    df["year"] = df[columna_fecha].dt.year
    df["month"]=df[columna_fecha].dt.month
    #df["semana"]=df[columna_fecha].dt.isocalendar().week
    
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

#Función de entrenamiento o carga del modelo
def entrenar_modelo(X, y, ruta_modelo: str = "modelo.json") -> XGBRegressor:
    """
    Esta función primero comprueba si existe un modelo entrenado, y lo utiliza si existe. No obstante,
    si no existe, entrena un modelo XGBoost nuevo.

    """
    modelo=XGBRegressor()
    if Path(ruta_modelo).exists(): #Por si quiero utilizar un modelo ya entrenado
        modelo.load_model(ruta_modelo)
    else: #entrenamiento del modelo
        modelo=XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            random_state=42, #para poder replicar resultdos
            n_jobs=-1,)
        modelo.fit(X, y)
        modelo.save_model(ruta_modelo)
    return modelo


# ----- Lógica principal -----

def ejecutar_modelo():
    """
    
    Flujo completo a ejecutar desde la carga de datos, la predicción y el guardado de predicciones.
    
    """
    try:
        hist = filedialog.askopenfilename(
            title="Selecciona el Excel de entrada",
            filetypes=[("Excel files", "*.xlsx")])
        if not hist:
            return  #El usuario lo canceló

        # Preparar datos
        df = pd.read_excel(hist)
        nece_cols = {"Dia 1 Mes", "Segmento", "Matricul."}
        if not nece_cols.issubset(df.columns):
            messagebox.showerror("Error de formato",
                "El archivo debe contener las columnas 'Dia 1 Mes', 'Segmento' y 'Matricul.'")
            return

        df["Dia 1 Mes"] = pd.to_datetime(df["Dia 1 Mes"])
        df = variables_calendario(df, "Dia 1 Mes")

        df_agg=df.groupby(["Dia 1 Mes", "Segmento"]).agg({"Matricul.": "sum"}).reset_index()
        #df_agg
        df_agg=variables_calendario(df_agg, "Dia 1 Mes")

        df_dummies = pd.get_dummies(df_agg["Segmento"], prefix="segmento")
        df_model = pd.concat([df_agg, df_dummies], axis=1)
        features =["year",
            "month",
            "msin",
            "mcos",
            "post_covid",
            "verano",
            "fin_trimestre",
            "diciembre",
        ] + list(df_dummies.columns)

        X = df_model[features]
        y = df_model["Matricul."]

        # Aplica función modelo
        model = entrenar_modelo(X, y)

        #Generar combinaciones de fecha y segmento para los próximos 6 meses
        ult_fecha = df_model["Dia 1 Mes"].max()
        future_dates = pd.date_range(start=ult_fecha+DateOffset(months=1),periods=6,freq="MS")

        pred_rows = []
        segmentos = df_model["Segmento"].unique()
        for date in future_dates:
            for seg in segmentos:
                row = {"Dia 1 Mes": date,
                    "Segmento": seg,
                    "year": date.year,
                    "month":date.month,
                    "msin": np.sin(2*np.pi*date.month/12),
                    "mcos": np.cos(2*np.pi*date.month/12),
                    "post_covid":int(date >= pd.to_datetime("2021-01-01")),
                    "verano":int(date.month in [6,7,8]),
                    "fin_trimestre":int(date.month in [3,6,9,12]),
                    "diciembre":int(date.month==12),
                }
                for dcol in df_dummies.columns:
                    row[dcol] = 1 if dcol==f"segmento_{seg}" else 0
                pred_rows.append(row)

        df_future = pd.DataFrame(pred_rows)
        X_future = df_future[features]
        df_future["Matriculaciones"] = model.predict(X_future)

        #Resultado final ordenado
        df_resultado = df_future[["Dia 1 Mes","Segmento","Matriculaciones"]].copy()
        df_resultado.columns = ["Fecha","Segmento","Matriculaciones"]

        #Guardar resultado
        save_path = filedialog.asksaveasfilename(
            title="Guardar predicciones como...",
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx")])
        if save_path:
            df_resultado.to_excel(save_path,index=False)
            messagebox.showinfo("Éxito","Predicciones guardadas correctamente.")

    #Error al guardar
    except Exception as e:
        messagebox.showerror("Error inesperado", f"Ocurrió un error:\n{e}")

# ----- GUI -----

def lanzar_gui():
    """
    Abre la ventana principal donde se selecciona el archivo Excel y se genra la predicción.
    """

    root = Tk()
    root.title("Predicción de Matriculaciones por Segmento")
    root.geometry("350x100")

    Button(
        root,
        text="Seleccionar archivo y predecir",
        command=ejecutar_modelo).pack(pady=30)

    root.mainloop()

#Para lanzar la interfaz con el archivo '.exe'

if __name__ == "__main__":
    #Como se empaquetará con PyInstaller, hay que ajustar la ruta de trabajo
    if getattr(sys, "frozen", False):
        os.chdir(sys._MEIPASS)  #type: ignore, por pyinstaller

    lanzar_gui() #lanza la interfaz
