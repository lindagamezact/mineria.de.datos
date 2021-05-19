#Estadisticos 

import io 
from bs4 import BeautifulSoup
import pandas as pd 
from tabulate import tabulate
import statistics
import numpy as np

import statsmodels.api as sm
from statsmodels.formula.api import ols


def print_tabulate(df: pd.DataFrame):
    print(tabulate(df, headers=df.columns, tablefmt='orgtbl'))

df_complete = pd.read_csv("C:/Users/linda/OneDrive/Documentos/LINDA GAMEZ/7MO SEMESTRE/MINERÍA DE DATOS/mineria.de.datos/cvs/Basic and safely managed sanitation services limpio.csv")

#-------------------------------------------------------------AC-----------------------------------------------------------------------------
#AC es agrupación por año y categoría tomando en cuenta como dato numérico el porcentaje de la población que utiliza servicios de 
# saneamiento gestionados de forma segura para el calculo de los estadísticos 
#(Media/Mediana/Variación/Dispersión/Mínimo/Máximo) del porcentaje de la población de cierta categoría (usan al menos.../usan servicios de saneamiento...) en cierto año 

def TE_AC(*op1): 
    lis_op=[] #Lista para juntar todos los estadisticos en un DataFrame
    for op in op1:
        df_by_AC = df_complete.groupby(["Anio","Categoria"])[["Porcentaje"]].aggregate(op)
        df_by_AC.reset_index(inplace=True)
        lis_op.append(df_by_AC['Porcentaje']) #Agrega datos a la lista
    
    #DataFrame con todos los estadisticos
    com = pd.DataFrame({"Categoria":df_by_AC['Categoria'],
        "Promedio":lis_op[0],"Mediana":lis_op[1],"Varianza":lis_op[2],"Desviacion_Estandar":lis_op[3],"Minimo":lis_op[4],"Maximo":lis_op[5] })
    com.set_index(df_by_AC['Anio'], inplace=True)
    print_tabulate(com.head(10))

TE_AC(statistics.mean,statistics.median,statistics.variance,statistics.stdev,min,max)

#-------------------------------------------------------------AR-----------------------------------------------------------------------------
#AR es agrupación por año y región de la OMS tomando en cuenta como dato numérico el porcentaje de la población que utiliza servicios de 
#saneamiento gestionados de forma segura para el calculo de los estadísticos 
#(Media/Mediana/Variación/Dispersión/Mínimo/Máximo) del porcentaje de la población de cierta región (Europa, America,...) en cierto año 

def TE_AR(*op1): #Año, region
    lis_op=[] 
    for op in op1:
        df_by_AR = df_complete.groupby(["Anio","Region de la OMS"])[["Porcentaje"]].aggregate(op)
        df_by_AR.reset_index(inplace=True)
        lis_op.append(df_by_AR['Porcentaje']) 
    
    #DataFrame con todos los estadisticos
    com = pd.DataFrame({"Region":df_by_AR['Region de la OMS'],
        "Promedio":lis_op[0],"Mediana":lis_op[1],"Varianza":lis_op[2],"Desviacion_Estandar":lis_op[3],"Minimo":lis_op[4],"Maximo":lis_op[5]})
    com.set_index(df_by_AR['Anio'], inplace=True)
    print_tabulate(com.head(10))
    
TE_AR(statistics.mean,statistics.median,statistics.variance,statistics.stdev,min,max)

#-------------------------------------------------------------AT-----------------------------------------------------------------------------
#AT es agrupación por año y tipo de área tomando en cuenta como dato numérico el porcentaje de la población que utiliza servicios de 
#saneamiento gestionados de forma segura para el calculo de los estadísticos 
#(Media/Mediana/Variación/Dispersión/Mínimo/Máximo) del porcentaje de la población de cierto tipo de área (rural, urbano, total) en cierto año 

def TE_AT(*op1): 
    lis_op=[] 
    for op in op1:
        df_by_AT = df_complete.groupby(["Anio","Tipo del Area"])[["Porcentaje"]].aggregate(op)
        df_by_AT.reset_index(inplace=True)
        lis_op.append(df_by_AT['Porcentaje']) 
    
    #DataFrame con todos los estadisticos
    com = pd.DataFrame({"Area":df_by_AT['Tipo del Area'],
        "Promedio":lis_op[0],"Mediana":lis_op[1],"Varianza":lis_op[2],"Desviacion_Estandar":lis_op[3],"Minimo":lis_op[4],"Maximo":lis_op[5]})
    com.set_index(df_by_AT['Anio'], inplace=True)
    print_tabulate(com.head(10))

TE_AT(statistics.mean,statistics.median,statistics.variance,statistics.stdev,min,max)

#-------------------------------------------------------------A-----------------------------------------------------------------------------
#A es agrupación solo por año tomando en cuenta como dato numérico el porcentaje de la población que utiliza servicios de 
# saneamiento gestionados de forma segura para el calculo de los estadísticos 
#(Media/Mediana/Variación/Dispersión/Mínimo/Máximo) del porcentaje de la población que utiliza servicios de saneamiento gestionados 
# de forma segura

def TE_A(*op1): 
    lis_op=[] 
    for op in op1:
        df_by_A = df_complete.groupby(["Anio"])[["Porcentaje"]].aggregate(op)
        df_by_A.reset_index(inplace=True)
        lis_op.append(df_by_A['Porcentaje'])         
    
    #DataFrame con todos los estadisticos
    com = pd.DataFrame({"Promedio":lis_op[0],"Mediana":lis_op[1],"Varianza":lis_op[2],"Desviacion_Estandar":lis_op[3],
        "Minimo":lis_op[4],"Maximo":lis_op[5]})
    com.set_index(df_by_A['Anio'], inplace=True)
    print_tabulate(com.head(8))
    
TE_A(statistics.mean,statistics.median,statistics.variance,statistics.stdev,min,max)

#-------------------------------------------------------------ABC-----------------------------------------------------------------------------
#ABC se refiere a que se pueden agregar 3 diferentes categorías 

#TE_abc es una función donde se agrupan 3 diferentes categorías y se hace el calculo del promedio del porcentaje de la población que utiliza 
# servicios de saneamiento gestionados de forma segura. 

def TE_abc(a:str,b:str,c:str): 
    
    df_by_abc = df_complete.groupby([a,b,c])[["Porcentaje"]].aggregate(statistics.mean)
    df_by_abc.reset_index(inplace=True)
    df_by_abc.set_index("Anio",inplace=True)
    df_by_abc.rename(columns={"Porcentaje":"Promedio"},inplace=True)
    
    #Data frame con estadístico (más significante)
    print_tabulate(df_by_abc.head(10))
 
TE_abc('Anio','Region de la OMS', 'Pais') #Tomando en cuenta el año, la región y el país 
TE_abc('Anio', 'Categoria', 'Region de la OMS') #Tomando en cuenta el año, la categoría y la región 
TE_abc('Anio', 'Categoria', 'Tipo del Area') #Tomando en cuenta el año, la categoría y el tipo de área 
TE_abc('Anio', 'Region de la OMS', 'Tipo del Area')  #Tomando en cuenta el año, la región y el tipo de área 
TE_abc('Anio', 'Categoria', 'Pais') #Tomando en cuenta el año, la categoría y el país 

