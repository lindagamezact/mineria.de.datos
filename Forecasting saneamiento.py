#Forecasting saneamiento 

from tabulate import tabulate
import pandas as pd 
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statistics
import numbers
from statsmodels.stats.outliers_influence import summary_table
from typing import Tuple, Dict
import numpy as np

def print_tabulate(df: pd.DataFrame):
    print(tabulate(df, headers=df.columns, tablefmt='github'))

df_complete = pd.read_csv("C:/Users/linda/OneDrive/Documentos/LINDA GAMEZ/7MO SEMESTRE/MINERÍA DE DATOS/mineria.de.datos/cvs/Basic and safely managed sanitation services limpio.csv")

#Funcion transformar la variable para poder hacer calculos 
def transform_variable(df: pd.DataFrame, x:str)->pd.Series:
    if isinstance(df[x][0], int):
        return df[x] 
    else:
        return pd.Series([i for i in range(0, len(df[x]))])

#Funcion para obtener resultados de regresion lineal 
def linear_regressionF(df: pd.DataFrame, x:str, y: str)->Dict[str, float]:
    fixed_x = transform_variable(df, x)
    model= sm.OLS(df[y],sm.add_constant(fixed_x), alpha=0.1).fit()
    bands = pd.read_html(model.summary().tables[1].as_html(),header=0,index_col=0)[0]
    print_tabulate(pd.read_html(model.summary().tables[1].as_html(),header=0,index_col=0)[0])
    coef = pd.read_html(model.summary().tables[1].as_html(),header=0,index_col=0)[0]['coef']
    r_2_t = pd.read_html(model.summary().tables[0].as_html(),header=None,index_col=None)[0]
    return {'m': coef.values[1], 'b': coef.values[0], 'r2': r_2_t.values[0][3], 'r2_adj': r_2_t.values[1][3], 'low_band': bands['[0.025'][0], 'hi_band': bands['0.975]'][0]}

#Funcion para hacer el plot con el intervalo de confianza para el pronóstico 
def plt_lr(df: pd.DataFrame, x:str, y: str, m: float, b: float, r2: float, r2_adj: float, low_band: float, hi_band: float, colors: Tuple[str,str]):
    fixed_x = transform_variable(df, x)
    df.plot(x=x,y=y, kind='scatter')
    plt.plot(df[x],[ m * x + b for _, x in fixed_x.items()], color=colors[0])
    plt.fill_between(df[x],
                     [ m * x  + low_band for _, x in fixed_x.items()],
                     [ m * x + hi_band for _, x in fixed_x.items()], alpha=0.2, color=colors[1])

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

    tit = "Regresión datos completos" #titulos más que nada para diseño 
    tit2 = "Datos completos"

    #Tabla para hacer regresion lineal, se toma en cuenta solo el promedio del porcentaje de la población ya que es el más importante de todos (da mejor explicación de la info)
    lr_t = pd.DataFrame({"Anio":df_by_A['Anio'],"Promedio_del_%_de_pob":lis_op[0] } )
    print(tit.center(150,"*"))
    #print_tabulate(lr_t)

    #Forecasting tomando en cuenta todos los datos, solo separados por año
    a= linear_regressionF(lr_t, 'Anio', 'Promedio_del_%_de_pob')
    plt_lr(df=lr_t, x="Anio", y="Promedio_del_%_de_pob", colors=('orange', 'lime'), **a)
    plt.title(f'Forecasting {tit2}')
    plt.xticks(lr_t['Anio'],rotation=90)
    #plt.show()
    plt.savefig(f"C:/Users/linda/OneDrive/Documentos/LINDA GAMEZ/7MO SEMESTRE/MINERÍA DE DATOS/mineria.de.datos/img2/F_{tit2}.png",bbox_inches='tight')
    plt.close() 

TE_A(statistics.mean,statistics.median,statistics.variance,statistics.stdev,min,max)

#TODAS LAS FUNCIONES QUE TIENEN TE_A(NOMBRECAT) USAN LAS MISMAS FORMULAS, SOLO CAMBIAN LOS DATOS QUE CONTIENEN SEGUN LA CATEGORÍA

#Función con año y región 
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

    #Forecasting tomando en cuenta que se tiene la categoría de años y una extra (dividida por región) 
    for cat in set(com[agrupar]): #for para hacer gráficas de cada una de las subcategorías (ej. en región se tiene Americas, Europe,...)
        grupo_lr = com[com[agrupar] == cat] 
        lr_t = pd.DataFrame({"Promedio_del_%_de_pob":grupo_lr["Promedio"]}) 
        lr_t.reset_index(inplace=True)
        #lr_t es el data frame con solo el promedio del porcentaje de la población para las categorías que se toman en cuenta, es el estadistico más importante para los datos que se toman en cuenta
        print(cat.center(150,"*"))
        a= linear_regressionF(lr_t, 'Anio', 'Promedio_del_%_de_pob')
        plt_lr(df=lr_t, x="Anio", y="Promedio_del_%_de_pob", colors=('orange', 'lime'), **a)
        plt.title(f'Forecasting {cat}')
        plt.xticks(lr_t['Anio'],rotation=90)
        #plt.show()
        plt.savefig(f"C:/Users/linda/OneDrive/Documentos/LINDA GAMEZ/7MO SEMESTRE/MINERÍA DE DATOS/mineria.de.datos/img2/F_{aA}_{agrupar}_{cat}.png",bbox_inches='tight')
        plt.close() 

TE_AR(statistics.mean,statistics.median,statistics.variance,statistics.stdev,min,max)

#Función con año y categoría 
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

    #Forecasting tomando en cuenta que se tiene la categoría de años y una extra (dividida por tipo de categoría) 
    for cat in set(com[agrupar]):
        grupo_lr = com[com[agrupar] == cat]
        lr_t = pd.DataFrame({"Promedio_del_%_de_pob":grupo_lr["Promedio"]}) 
        lr_t.reset_index(inplace=True)
        #lr_t data frame con solo el promedio del porcentaje de la población para las categorías que se toman en cuenta, es el estadístico más importante para los datos que se toman en cuenta
        print(cat.center(150,"*"))
        a= linear_regressionF(lr_t, 'Anio', 'Promedio_del_%_de_pob')
        plt_lr(df=lr_t, x="Anio", y="Promedio_del_%_de_pob", colors=('orange', 'lime'), **a)
        plt.title(f'Forecasting {cat}')
        plt.xticks(lr_t['Anio'],rotation=90)
        #plt.show()
        plt.savefig(f"C:/Users/linda/OneDrive/Documentos/LINDA GAMEZ/7MO SEMESTRE/MINERÍA DE DATOS/mineria.de.datos/img2/F_{aA}_{agrupar}_{cat}.png",bbox_inches='tight')
        plt.close() 
 
TE_AC(statistics.mean,statistics.median,statistics.variance,statistics.stdev,min,max)

#función con año y tipo de área 
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

    #Forecasting tomando en cuenta que se tiene la categoría de años y una extra (dividida por tipo de área) 
    for cat in set(com[agrupar]):
        grupo_lr = com[com[agrupar] == cat]
        lr_t = pd.DataFrame({"Promedio_del_%_de_pob":grupo_lr["Promedio"]}) 
        lr_t.reset_index(inplace=True)
        #lr_t data frame con solo el promedio del porcentaje de la población para las categorías que se toman en cuenta, es el estadístico más importante para los datos que se toman en cuenta
        print(cat.center(150,"*"))
        a= linear_regressionF(lr_t, 'Anio', 'Promedio_del_%_de_pob')
        plt_lr(df=lr_t, x="Anio", y="Promedio_del_%_de_pob", colors=('orange', 'lime'), **a)
        plt.title(f'Forecasting {cat}')
        plt.xticks(lr_t['Anio'],rotation=90)
        #plt.show()
        plt.savefig(f"C:/Users/linda/OneDrive/Documentos/LINDA GAMEZ/7MO SEMESTRE/MINERÍA DE DATOS/mineria.de.datos/img2/F_{aA}_{agrupar}_{cat}.png",bbox_inches='tight')
        plt.close() 

TE_AT(statistics.mean,statistics.median,statistics.variance,statistics.stdev,min,max)
