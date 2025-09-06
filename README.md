# Predicción de la demanda de vehículos mediante Machine Learning aplicada a una marca del sector de la automoción

**Trabajo de Fin de Máster**

Máster Universitario en Análisis y Visualización de Datos Masivos / Visual Analytics and Big Data – UNIR 2025

**Paula González Alonso**

## Objetivo del proyecto
El objetivo principal de este proyecto ha sido desarrollar una solución capaz de predecir la demanda mensual de matriculaciones por segmento de modelo de coche a partir de datos históricos del mercado Español. La idea no es solo construir un modelo que funcione bien, sino también presentar los resultados de forma clara y útil para perfiles no técnicos, facilitando su aplicación en contextos reales de planificación y negocio.

## ¿Qué contiene el repositorio?
Este repositorio incluye todos lo que se ha ido desarrollando durante el Trabajo de Fin de Máster:
- **Código fuente** del modelo predictivo, y el estudio estadístico.
- **Archivo ejecutable** que permite al usuario generar nuevas predicciones sin necesidad de ser un perfil técnico.
- **Datos históricos** de matriculaciones con el formato necesario para probar el modelo predictivo.
- **Predicciones generadas** por el modelo XGBoost para los próximos 6 meses.
- **Panel de Power BI** donde se visualiza:
	- Evolución histórica de la demanda de matriculaciones.
 	- Predicciones por segmento.

## Descargar el modelo predictivo (archivo ejecutable)
Puedes descargar el ejecutable del modelo desde el siguiente enlace:

`ModeloPredictivo_6meses.exe`: https://drive.google.com/file/d/151FYYBfmQNgsYZ0sULoh_CVE21O6ajFV/view?usp=sharing

> El archivo ejecutable supera el límite de tamaño permitido por GitHub, por eso se ha compartido mediante un enlace de Google Drive.

## Herramientas utilizadas
- **Python 3.10** para todo el desarrollo del modelo y la interfaz
- **XGBoost** como modelo predictivo
- **PyInstaller** para compilar la aplicación en un archivo ejecutable
- **Power BI Desktop** para la visualización de los resultados
