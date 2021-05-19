#Servicios de saneamiento

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
import numbers
import random 

def print_tabulate(df: pd.DataFrame):
    print(tabulate(df, headers=df.columns, tablefmt='orgtbl'))

#------------------------------------------------------- Data cleaning and parsing -------------------------------------------------------------
df = pd.read_csv("C:/Users/linda/OneDrive/Documentos/LINDA GAMEZ/7MO SEMESTRE/MINERÍA DE DATOS/mineria.de.datos/cvs/Basic and safely managed sanitation services.csv")
#print_tabulate(df)
df = df.drop(['Display Value'], axis=1)
df = df.drop(['PUBLISH STATES'], axis=1)
#print(df.columns)
df.columns =['Indicador','Anio', 'Region de la OMS', 'Pais', 'Tipo del Area', 'Porcentaje']
pd.to_numeric(df['Porcentaje'])
pd.to_numeric(df['Anio'],downcast='integer')

#num=['Anio','Porcentaje']
#df[num] = df[num].apply(pd.to_numeric, errors='coerce', axis=1)

df['Categoria'] = np.where(df['Indicador']=="Population using safely managed sanitation services (%)", 'Servicios de saneamiento gestionados de forma segura ', 'Al menos servicios basicos de saneamiento ')

df.to_csv("C:/Users/linda/OneDrive/Documentos/LINDA GAMEZ/7MO SEMESTRE/MINERÍA DE DATOS/mineria.de.datos/cvs/Basic and safely managed sanitation services limpio.csv", index=False)

df_complete = pd.read_csv("C:/Users/linda/OneDrive/Documentos/LINDA GAMEZ/7MO SEMESTRE/MINERÍA DE DATOS/mineria.de.datos/cvs/Basic and safely managed sanitation services limpio.csv")

#---------------------------------------------------------------------------------------------------------------------------------------------------

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

#Funcion para transformar para RL 
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
    print_tabulate(com.head(8))

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

    #ANOVA para el promedio de los datos 
    anova(agrupar,com)   

    #Regresion lineal 
    for cat in set(com[agrupar]):
        grupo_lr = com[com[agrupar] == cat]
        lr_t = pd.DataFrame({"Promedio_del_%_de_pob":grupo_lr["Promedio"]})
        lr_t.reset_index(inplace=True)
        print(cat.center(150,"*"))
        print_tabulate(lr_t.head(5))
        linear_regression2(lr_t,'Anio','Promedio_del_%_de_pob',agrupar,aA,cat)

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

    #ANOVA para el promedio de los datos 
    anova(agrupar,com) 

    #Regresion lineal 
    for cat in set(com[agrupar]):
        grupo_lr = com[com[agrupar] == cat]
        lr_t = pd.DataFrame({"Promedio_del_%_de_pob":grupo_lr["Promedio"]})
        lr_t.reset_index(inplace=True)
        print(cat.center(150,"*"))
        print_tabulate(lr_t.head(5))
        linear_regression2(lr_t,'Anio','Promedio_del_%_de_pob',agrupar,aA,cat)
    
TE_AR(statistics.mean,statistics.median,statistics.variance,statistics.stdev,min,max)

#-------------------------------------------------------------AT-----------------------------------------------------------------------------
#AT es agrupación por año y tipo de area tomando en cuenta como dato numérico el porcentaje de la población que utiliza servicios de 
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
    
    #ANOVA para el promedio de los datos 
    anova(agrupar,com) 

    #Regresion lineal 
    for cat in set(com[agrupar]):
        grupo_lr = com[com[agrupar] == cat]
        lr_t = pd.DataFrame({"Promedio_del_%_de_pob":grupo_lr["Promedio"]})
        lr_t.reset_index(inplace=True)
        print(cat.center(150,"*"))
        print_tabulate(lr_t.head(5))
        linear_regression2(lr_t,'Anio','Promedio_del_%_de_pob',agrupar,aA,cat)

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
    
    #Gráficas
    for lar in com.columns: 
        graf = plt.figure()
        plt.plot(com.index,com[lar])
        gxanio_format(com)
        plt.title(lar)
        graf.savefig(f"C:/Users/linda/OneDrive/Documentos/LINDA GAMEZ/7MO SEMESTRE/MINERÍA DE DATOS/mineria.de.datos/img/A_{lar}.png",bbox_inches='tight')
        plt.close()

    tit = "Regresión datos completos"

    #Tabla para regresion lineal 
    lr_t = pd.DataFrame({"Anio":df_by_A['Anio'],"Promedio_del_%_de_pob":lis_op[0] } )
    print(tit.center(150,"*"))
    print_tabulate(lr_t)

    #Regresion lineal 
    linear_regression(lr_t, 'Anio', 'Promedio_del_%_de_pob',tit)
    
TE_A(statistics.mean,statistics.median,statistics.variance,statistics.stdev,min,max)

#-------------------------------------------------------------ABC-----------------------------------------------------------------------------
#ABC se refiere a que se pueden agregar 3 diferentes categorías 

#TE_abc es una función donde se agrupan 3 diferentes categorías y se hace el calculo del promedio del porcentaje de la población que utiliza 
# servicios de saneamiento gestionados de forma segura. La función también incluye las graficas necesarias para las agrupaciones. 
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

#Agrupación por año, región de la OMS y país. Contiene: 
# Tabla donde se agrupa en cada año el promedio de los porcentajes que tiene cada país de cierta región sin tomar en cuenta el tipo de area ni
#  la categoría en la que se encuentra.
# Boxplots del promedio del porcentaje de la población... por región de la OMS con sus respectivos países. 
# Plots individuales del promedio del % para cada país a través de los años
# Boxplots individuales del promedio del % a través de los años para cada país 

TE_abc('Anio', 'Categoria', 'Region de la OMS')
TE_abc('Anio', 'Categoria', 'Tipo del Area')
TE_abc('Anio', 'Region de la OMS', 'Tipo del Area')
TE_abc('Anio', 'Categoria', 'Pais')
