#Graficas 

import io 
from bs4 import BeautifulSoup
import pandas as pd 
from tabulate import tabulate
import statistics
import numpy as np
from numpy.core.fromnumeric import size 

import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import shapiro
from scipy.stats import bartlett
import pingouin as pg
from statsmodels.stats.multicomp import pairwise_tukeyhsd

def print_tabulate(df: pd.DataFrame):
    print(tabulate(df, headers=df.columns, tablefmt='orgtbl'))

df_complete = pd.read_csv("C:/Users/linda/OneDrive/Documentos/LINDA GAMEZ/7MO SEMESTRE/MINERÍA DE DATOS/mineria.de.datos/cvs/Basic and safely managed sanitation services limpio.csv")

#Funcion para el formato del eje x (Años) de las gráficas
def gxanio_format(df: pd.DataFrame): 
    plt.xticks(df.index)
    plt.xticks(rotation ='vertical')
    plt.xlabel("Años")

#Funcion para las graficas cuando la agrupacion se conforma de los años y 1 categoría más
def plot_by_A(df: pd.DataFrame, cat:str,lar:str,ag:str,a:str)->None: 
    df[df[ag] == cat].plot(y =lar) 
    gxanio_format(df)
    plt.title(cat)    
    plt.savefig(f"C:/Users/linda/OneDrive/Documentos/LINDA GAMEZ/7MO SEMESTRE/MINERÍA DE DATOS/mineria.de.datos/img/{a}_{lar}_{cat}.png",bbox_inches='tight')
    plt.close()
    df[df[ag] == cat].boxplot(lar,by =ag)
    plt.savefig(f"C:/Users/linda/OneDrive/Documentos/LINDA GAMEZ/7MO SEMESTRE/MINERÍA DE DATOS/mineria.de.datos/img/{a}_{lar}_bplt_{cat}.png",bbox_inches='tight')
    plt.close()

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

    #Un estadístico por boxplot
    for lis in com.columns[1:len(com.columns)]:
        com.boxplot(lis,by=agrupar)
        plt.xticks(rotation =5)
        plt.tight_layout()
        plt.savefig(f"C:/Users/linda/OneDrive/Documentos/LINDA GAMEZ/7MO SEMESTRE/MINERÍA DE DATOS/mineria.de.datos/img/{aA}_{lis}_boxplot.png")
        plt.close()
    
    for cat in set(com[agrupar]):
        for lar in com.columns[1:len(com.columns)]: 
            plot_by_A(com, cat,lar,agrupar,aA)

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

    #Un estadístico por boxplot
    for lis in com.columns[1:len(com.columns)]:
        com.boxplot(lis,by=agrupar)
        plt.xticks(rotation =18)
        plt.tight_layout()
        plt.savefig(f"C:/Users/linda/OneDrive/Documentos/LINDA GAMEZ/7MO SEMESTRE/MINERÍA DE DATOS/mineria.de.datos/img/{aA}_{lis}_boxplot.png")
        plt.close()
    
    #Graficas
    for cat in set(com[agrupar]):
        for lar in com.columns[1:len(com.columns)]: 
            plot_by_A(com, cat,lar,agrupar,aA)
  
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

    #Un estadístico por boxplot
    for lis in com.columns[1:len(com.columns)]:
        com.boxplot(lis,by=agrupar)
        plt.xticks(rotation =18)
        plt.tight_layout()
        plt.savefig(f"C:/Users/linda/OneDrive/Documentos/LINDA GAMEZ/7MO SEMESTRE/MINERÍA DE DATOS/mineria.de.datos/img/{aA}_{lis}_boxplot.png")
        plt.close()
    
    #Graficas
    for cat in set(com[agrupar]):
        for lar in com.columns[1:len(com.columns)]: 
            plot_by_A(com, cat,lar,agrupar,aA)
    
TE_AT(statistics.mean,statistics.median,statistics.variance,statistics.stdev,min,max)

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
    
    #Gráficas
    for lar in com.columns: 
        graf = plt.figure()
        plt.plot(com.index,com[lar])
        gxanio_format(com)
        plt.title(lar)
        graf.savefig(f"C:/Users/linda/OneDrive/Documentos/LINDA GAMEZ/7MO SEMESTRE/MINERÍA DE DATOS/mineria.de.datos/img/A_{lar}.png",bbox_inches='tight')
        plt.close()
   
TE_A(statistics.mean,statistics.median,statistics.variance,statistics.stdev,min,max)

def TE_abc(a:str,b:str,c:str): 
    
    df_by_abc = df_complete.groupby([a,b,c])[["Porcentaje"]].aggregate(statistics.mean)
    df_by_abc.reset_index(inplace=True)
    df_by_abc.set_index("Anio",inplace=True)
    df_by_abc.rename(columns={"Porcentaje":"Promedio"},inplace=True)
    print_tabulate(df_by_abc.head(10))

    aA = a[0]+b[0]+c[0]

    #Una categoría de la agrupación por boxplot
    for cat in set(df_by_abc[b]):
        bp1 = df_by_abc[df_by_abc[b] == cat]
        bp1.boxplot("Promedio",by=c)
        plt.xticks(rotation ='vertical')
        plt.title(b+": "+cat)
        plt.ylabel("Promedio")
        plt.savefig(f"C:/Users/linda/OneDrive/Documentos/LINDA GAMEZ/7MO SEMESTRE/MINERÍA DE DATOS/mineria.de.datos/img/{aA}_{cat}_boxplot.png",bbox_inches='tight')
        plt.close()
    
    #Graficas 
    for cat in set(df_by_abc[b]):
        a1 = df_by_abc[df_by_abc[b] == cat]
        for c2 in a1[c] :
            a2 = a1[a1[c] == c2]
            a2.plot(y='Promedio')
            plt.title(b+": "+cat,loc="left")
            plt.title(c2,loc="right")
            gxanio_format(a2)
            plt.savefig(f"C:/Users/linda/OneDrive/Documentos/LINDA GAMEZ/7MO SEMESTRE/MINERÍA DE DATOS/mineria.de.datos/img/{aA}_Promedio_{cat}_{c2}.png",bbox_inches='tight')
            plt.close()
            a2.boxplot("Promedio",by=c)
            plt.title(b+": "+cat)
            plt.savefig(f"C:/Users/linda/OneDrive/Documentos/LINDA GAMEZ/7MO SEMESTRE/MINERÍA DE DATOS/mineria.de.datos/img/{aA}_Promedio_bplt_{cat}_{c2}.png",bbox_inches='tight')
            plt.close()

TE_abc('Anio','Region de la OMS', 'Pais') 
TE_abc('Anio', 'Categoria', 'Region de la OMS')
TE_abc('Anio', 'Categoria', 'Tipo del Area')
TE_abc('Anio', 'Region de la OMS', 'Tipo del Area')
TE_abc('Anio', 'Categoria', 'Pais')

