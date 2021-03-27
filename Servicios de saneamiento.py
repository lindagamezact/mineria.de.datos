#Servicios de saneamiento

import io 
from bs4 import BeautifulSoup
import pandas as pd 
from tabulate import tabulate
import statistics
import numpy as np

def print_tabulate(df: pd.DataFrame):
    print(tabulate(df, headers=df.columns, tablefmt='orgtbl'))

df = pd.read_csv("C:/Users/linda/OneDrive/Documentos/LINDA GAMEZ/7MO SEMESTRE/MINERÍA DE DATOS/mineria.de.datos/cvs/Basic and safely managed sanitation services.csv")
#print_tabulate(df)
df = df.drop(['Display Value'], axis=1)
df = df.drop(['PUBLISH STATES'], axis=1)
#print(df.columns)
df.columns =['Indicador','Anio', 'Region de la OMS', 'Pais', 'Tipo del Area', 'Porcentaje']
num=['Anio','Porcentaje']
df[num] = df[num].apply(pd.to_numeric, errors='coerce', axis=1)

df['Categoria'] = np.where(df['Indicador']=="Population using safely managed sanitation services (%)", 'Servicios de saneamiento gestionados de forma segura ', 'Al menos servicios basicos de saneamiento ')

df.to_csv("C:/Users/linda/OneDrive/Documentos/LINDA GAMEZ/7MO SEMESTRE/MINERÍA DE DATOS/mineria.de.datos/cvs/Basic and safely managed sanitation services limpio.csv", index=False)

df_complete = pd.read_csv("C:/Users/linda/OneDrive/Documentos/LINDA GAMEZ/7MO SEMESTRE/MINERÍA DE DATOS/mineria.de.datos/cvs/Basic and safely managed sanitation services limpio.csv")

#Promedio por region y tipo de area y Categoria
df_by_dep = df_complete.groupby(["Region de la OMS","Categoria","Tipo del Area"])[["Porcentaje"]].aggregate(statistics.mean)

#Minimo por region y tipo de area
#df_by_dep = df_complete.groupby(["Region de la OMS","Categoria","Tipo del Area"])[["Porcentaje"]].aggregate(min)

#Maximo por region y tipo de area
#df_by_dep = df_complete.groupby(["Region de la OMS","Categoria","Tipo del Area"])[["Porcentaje"]].aggregate(max)

df_by_dep.reset_index(inplace=True)
#df_by_dep.set_index("Categoria", inplace=True)
print_tabulate(df_by_dep)
