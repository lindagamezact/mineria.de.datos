#Regresion lineal para los servicios de saneamiento 

import io
from numpy.core.fromnumeric import size 
import pandas as pd 
from tabulate import tabulate
import statistics
import numpy as np

import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
import numbers
import random 

def print_tabulate(df: pd.DataFrame):
    print(tabulate(df, headers=df.columns, tablefmt='github'))

df_complete = pd.read_csv("C:/Users/linda/OneDrive/Documentos/LINDA GAMEZ/7MO SEMESTRE/MINERÍA DE DATOS/mineria.de.datos/cvs/Basic and safely managed sanitation services limpio.csv")

def transform_variable(df: pd.DataFrame,x: str) -> pd.Series: 
    if isinstance(df[x][0], int): 
        return df[x]
    else: 
        return [i for i in range(0, len(df[x]))]

#Funcion regresion lineal para solo la cat con años 
def linear_regression(df:pd.DataFrame, x: str, y: str,t:str)-> None: 
    fixed_x= transform_variable(df, x)
    model = sm.OLS(df[y],sm.add_constant(fixed_x)).fit()  
    print (model.summary())

    coef=pd.read_html(model.summary().tables[1].as_html(),header=0,index_col=0)[0]['coef']
    df.plot(x=x,y=y, kind='scatter')
    plt.plot(df[x],[pd.DataFrame.mean(df[y]) for _ in range(0, len(df[x]))], color='orange')
    plt.plot(df[x],[ coef.values[1] * x + coef.values[0] for x in range(0, len(df[x]))], color='red')
    plt.xticks(df[x],rotation=90)
    plt.title(t)
    plt.xlabel("Años")
    #plt.show()
    plt.savefig(f"C:/Users/linda/OneDrive/Documentos/LINDA GAMEZ/7MO SEMESTRE/MINERÍA DE DATOS/mineria.de.datos/img2/RL_{t}.png",bbox_inches='tight')
    plt.close()

#Funcion regresion lineal para las que cuentan con categoría de año y una más 
def linear_regression2(df:pd.DataFrame, x: str, y: str,a:str,b:str,c:str)-> None: 
    fixed_x= transform_variable(df, x)
    model = sm.OLS(df[y],sm.add_constant(fixed_x)).fit()  
    print (model.summary())

    coef=pd.read_html(model.summary().tables[1].as_html(),header=0,index_col=0)[0]['coef']
    df.plot(x=x,y=y, kind='scatter')
    plt.plot(df[x],[pd.DataFrame.mean(df[y]) for _ in range(0, len(df[x]))], color='bisque')
    co=random.choice(( ['pink','red','orange','green','lime','salmon','maroon','aqua','olive','violet','navy','slateblue','peru','rosybrown']))
    plt.plot(df[x],[ coef.values[1] * x + coef.values[0] for x in range(0, len(df[x]))], color=co)
    plt.xticks(df[x],rotation=90)
    plt.title(f"Regresión {a}: {c}")
    plt.xlabel("Años")
    plt.savefig(f"C:/Users/linda/OneDrive/Documentos/LINDA GAMEZ/7MO SEMESTRE/MINERÍA DE DATOS/mineria.de.datos/img2/RL_{b}_{a}_{c}.png",bbox_inches='tight')
    plt.close()

def TE_A(*op1): 
    lis_op=[] 
    for op in op1:
        df_by_A = df_complete.groupby(["Anio"])[["Porcentaje"]].aggregate(op)
        df_by_A.reset_index(inplace=True)
        lis_op.append(df_by_A['Porcentaje'])         
    #DataFrame con todos los estadisticos
    com = pd.DataFrame({"Promedio":lis_op[0],"Mediana":lis_op[1],"Varianza":lis_op[2],"Desviación Estándar":lis_op[3],
        "Mínimo":lis_op[4],"Máximo":lis_op[5]})
    com.set_index(df_by_A['Anio'], inplace=True)

    tit = "Regresión datos completos"

    #Tabla para regresion lineal 
    lr_t = pd.DataFrame({"Anio":df_by_A['Anio'],"Promedio_del_%_de_pob":lis_op[0] } )
    print(tit.center(150,"*"))
    print_tabulate(lr_t)

    #Regresion lineal 
    linear_regression(lr_t, 'Anio', 'Promedio_del_%_de_pob',tit)

TE_A(statistics.mean,statistics.median,statistics.variance,statistics.stdev,min,max)

def TE_AR(*op1): #Año, region
    lis_op=[] 
    for op in op1:
        df_by_AR = df_complete.groupby(["Anio","Region de la OMS"])[["Porcentaje"]].aggregate(op)
        df_by_AR.reset_index(inplace=True)
        lis_op.append(df_by_AR['Porcentaje']) 
    #DataFrame con todos los estadisticos
    com = pd.DataFrame({"Region":df_by_AR['Region de la OMS'],
        "Promedio":lis_op[0],"Mediana":lis_op[1],"Varianza":lis_op[2],"Desviación Estándar":lis_op[3],"Mínimo":lis_op[4],"Máximo":lis_op[5]})
    com.set_index(df_by_AR['Anio'], inplace=True)

    agrupar = "Region"
    aA = "AR" 

    #Regresion lineal 
    for cat in set(com[agrupar]):
        grupo_lr = com[com[agrupar] == cat]
        lr_t = pd.DataFrame({"Promedio_del_%_de_pob":grupo_lr["Promedio"]})
        lr_t.reset_index(inplace=True)
        print(cat.center(150,"*"))
        print_tabulate(lr_t.head(5))
        linear_regression2(lr_t,'Anio','Promedio_del_%_de_pob',agrupar,aA,cat)

TE_AR(statistics.mean,statistics.median,statistics.variance,statistics.stdev,min,max)

def TE_AC(*op1): 
    lis_op=[] #Lista para juntar todos los estadisticos en un DataFrame
    for op in op1:
        df_by_AC = df_complete.groupby(["Anio","Categoria"])[["Porcentaje"]].aggregate(op)
        df_by_AC.reset_index(inplace=True)
        lis_op.append(df_by_AC['Porcentaje']) #Agrega datos a la lista
    #DataFrame con todos los estadisticos
    com = pd.DataFrame({"Categoria":df_by_AC['Categoria'],
        "Promedio":lis_op[0],"Mediana":lis_op[1],"Varianza":lis_op[2],"Desviación Estándar":lis_op[3],"Mínimo":lis_op[4],"Máximo":lis_op[5] })
    com.set_index(df_by_AC['Anio'], inplace=True)

    agrupar = "Categoria"
    aA = "AC"

    #Regresion lineal 
    for cat in set(com[agrupar]):
        grupo_lr = com[com[agrupar] == cat]
        lr_t = pd.DataFrame({"Promedio_del_%_de_pob":grupo_lr["Promedio"]})
        lr_t.reset_index(inplace=True)
        print(cat.center(150,"*"))
        print_tabulate(lr_t.head(5))
        linear_regression2(lr_t,'Anio','Promedio_del_%_de_pob',agrupar,aA,cat)
 
TE_AC(statistics.mean,statistics.median,statistics.variance,statistics.stdev,min,max)

def TE_AT(*op1): 
    lis_op=[] 
    for op in op1:
        df_by_AT = df_complete.groupby(["Anio","Tipo del Area"])[["Porcentaje"]].aggregate(op)
        df_by_AT.reset_index(inplace=True)
        lis_op.append(df_by_AT['Porcentaje']) 
    #DataFrame con todos los estadisticos
    com = pd.DataFrame({"Area":df_by_AT['Tipo del Area'],
        "Promedio":lis_op[0],"Mediana":lis_op[1],"Varianza":lis_op[2],"Desviación Estándar":lis_op[3],"Mínimo":lis_op[4],"Máximo":lis_op[5]})
    com.set_index(df_by_AT['Anio'], inplace=True)

    agrupar = "Area"
    aA = "AT"

    #Regresion lineal 
    for cat in set(com[agrupar]):
        grupo_lr = com[com[agrupar] == cat]
        lr_t = pd.DataFrame({"Promedio_del_%_de_pob":grupo_lr["Promedio"]})
        lr_t.reset_index(inplace=True)
        print(cat.center(150,"*"))
        print_tabulate(lr_t.head(5))
        linear_regression2(lr_t,'Anio','Promedio_del_%_de_pob',agrupar,aA,cat)

TE_AT(statistics.mean,statistics.median,statistics.variance,statistics.stdev,min,max)