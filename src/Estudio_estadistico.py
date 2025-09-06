import pandas as pd

df = pd.read_excel('Datos_16-25_Seg.xlsx')

df['Dia 1 Mes'] = pd.to_datetime(df['Dia 1 Mes'])

#Análisis descriptivo de matriculaciones
print("Resumen estadístico de los datos:")
print(df['Matricul.'].describe())

#Análisis por segmento
print("\nEstadísticas por segmento:")
estadisticas_segmento = df.groupby('Segmento')['Matricul.'].agg(['mean', 'std', 'min', 'max', 'sum']).sort_values(by='mean', ascending=False)
print(estadisticas_segmento)

#Comprobación de valores nulos
print("\nConteo de valores nulos por columna:")
print(df.isnull().sum())
