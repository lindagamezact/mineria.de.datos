#ANOVA 

import io 
from bs4 import BeautifulSoup
import pandas as pd 
from tabulate import tabulate
import statistics
import numpy as np

import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import shapiro
from scipy.stats import bartlett
import pingouin as pg
from statsmodels.stats.multicomp import pairwise_tukeyhsd

def print_tabulate(df: pd.DataFrame):
    print(tabulate(df, headers=df.columns, tablefmt='github'))

df_complete = pd.read_csv("C:/Users/linda/OneDrive/Documentos/LINDA GAMEZ/7MO SEMESTRE/MINERÍA DE DATOS/Datasets/Basic and safely managed sanitation services limpio.csv")

#Función para el ANOVA cuando la agrupacion se conforma de los años y 1 categoría más
def anova(a:str,df:pd.DataFrame): 
    print("*****ANOVA*****")
    modl = ols(f"Promedio ~ {a}", data=df).fit()
    anova_df = sm.stats.anova_lm(modl, typ=2)
    pvan = anova_df["PR(>F)"][0]
    print(f"Obtuvimos un p valor de: {pvan}, por lo tanto:")
    if anova_df["PR(>F)"][0] < 0.05:
        print("El ANOVA nos dice que SI hay diferencias en los promedios de los datos")
        #print(anova_df)
        # Prueba tukey
        print("*Prueba Tukey*")
        print("Ya que si hay diferencias se hace la Prueba Tukey y se obtuvieron los siguientes grupos con diferencias:")
        tcomp = pairwise_tukeyhsd(endog=df['Promedio'], groups=df[a], alpha=0.05)
        print(tcomp)
    else:
        #print("Estadistico: Promedio")       
        print("El ANOVA nos dice que NO hay diferencias en los promedios de los datos")
        #print(anova_df)
    
    print("***SUPUESTOS DEL ANOVA***")
    #Prueba Normalidad Shapiro-Wilk test
    print("*Normalidad*")
    statSW,pSW = shapiro(modl.resid)
    print("Para el supuesto de normalidad se hizo el Shapiro-Wilk test y obtuvimos: ")
    print(f"p-valor= {pSW}, por lo tanto:")
    if pSW < 0.05: 
        print("La variable dependiente (residuales) NO se distribuye normalmente")
        #Levene´s test Prueba de Homogeneidad
        print("*Prueba de Homogeneidad*")
        print("Ya que NO es normal se hizo el Levene´s test: ")
        lev = pg.homoscedasticity(df,dv='Promedio',group=a,method='levene')
        plev = lev.pval.values
        print(f"Obtuvimos un p valor de: {plev}, por lo tanto:")
        if plev < 0.05: 
            print("NO hay homogeneidad de varianzas")
        else: 
            print("SI hay homogeneidad de varianzas")
    else: 
        print("La variable dependiente (residuales) SI se distribuye normalmente")
        #Bartlett's test Prueba de Homogeneidad
        print("*Prueba de Homogeneidad*")
        print("Ya que SI es normal se hizo el Bartlett's test: ")
        bar = pg.homoscedasticity(df,dv='Promedio',group=a,method='bartlett')
        pbar = bar.pval.values
        print(f"Obtuvimos un p valor de: {pbar}, por lo tanto:")
        if pbar < 0.05: 
            print("NO hay homogeneidad de varianzas")
        else: 
            print("SI hay homogeneidad de varianzas")

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

    agrupar = "Categoria"
    aA = "AC"

    #ANOVA para el promedio de los datos 
    anova(agrupar,com)    

TE_AC(statistics.mean,statistics.median,statistics.variance,statistics.stdev,min,max)

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

    agrupar = "Region"
    aA = "AR"

    #ANOVA para el promedio de los datos 
    anova(agrupar,com)   
    
TE_AR(statistics.mean,statistics.median,statistics.variance,statistics.stdev,min,max)

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

    agrupar = "Area"
    aA = "AT"

    #ANOVA para el promedio de los datos 
    anova(agrupar,com)

TE_AT(statistics.mean,statistics.median,statistics.variance,statistics.stdev,min,max)

