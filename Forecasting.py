#Forecasting 

import matplotlib.pyplot as plt
import statsmodels.api as sm
import numbers
import pandas as pd
from tabulate import tabulate
from statsmodels.stats.outliers_influence import summary_table
from typing import Tuple, Dict
import numpy as np

def print_tabulate(df: pd.DataFrame):
    print(tabulate(df, headers=df.columns, tablefmt="orgtbl"))

def transform_variable(df: pd.DataFrame, x:str)->pd.Series:
    if isinstance(df[x][df.index[0]], numbers.Number):
        return df[x] # type: pd.Series
    else:
        return pd.Series([i for i in range(0, len(df[x]))])

def linear_regression(df: pd.DataFrame, x:str, y: str)->Dict[str, float]:
    fixed_x = transform_variable(df, x)
    model= sm.OLS(list(df[y]),sm.add_constant(fixed_x), alpha=0.05).fit()
    bands = pd.read_html(model.summary().tables[1].as_html(),header=0,index_col=0)[0]
    #print_tabulate(pd.read_html(model.summary().tables[1].as_html(),header=0,index_col=0)[0])
    coef = pd.read_html(model.summary().tables[1].as_html(),header=0,index_col=0)[0]['coef']
    r_2_t = pd.read_html(model.summary().tables[0].as_html(),header=None,index_col=None)[0]
    return {'m': coef.values[1], 'b': coef.values[0], 'r2': r_2_t.values[0][3], 'r2_adj': r_2_t.values[1][3], 'low_band': bands['[0.025'][0], 'hi_band': bands['0.975]'][0]}

def plt_lr(df: pd.DataFrame, x:str, y: str, m: float, b: float, r2: float, r2_adj: float, low_band: float, hi_band: float, colors: Tuple[str,str]):
    fixed_x = transform_variable(df, x)
    plt.plot(df[x],[ m * x + b for _, x in fixed_x.items()], color=colors[0])
    plt.fill_between(df[x],
                     [ m * x  + low_band for _, x in fixed_x.items()],
                     [ m * x + hi_band for _, x in fixed_x.items()], alpha=0.2, color=colors[1])

def normalize_distribution(dist: np.array, n: int)->np.array:
    b = dist - min(dist) + 0.000001
    c = (b / np.sum(b)) * n
    return np.round(c)

#begin_date = '2016-01-01'
#end_date = '2022-01-01'
#date_range = pd.date_range(start=begin_date, end=end_date, freq='1D')
#norm_dist = np.random.standard_normal(len(date_range))
#sales = normalize_distribution(norm_dist, 1500*len(date_range))
#df = pd.DataFrame({'Fecha': date_range, 'Ventas': sales}) 
#df.to_csv("C:/Users/linda/OneDrive/Documentos/LINDA GAMEZ/7MO SEMESTRE/MINERÍA DE DATOS/mineria.de.datos/cvs/sales.csv", index=False)

full_df = pd.read_csv("C:/Users/linda/OneDrive/Documentos/LINDA GAMEZ/7MO SEMESTRE/MINERÍA DE DATOS/mineria.de.datos/cvs/sales.csv")
print(f"full -> mean: {np.mean(full_df['Ventas'])}, sd: {np.std(full_df['Ventas'])}")
df = full_df.tail(50)
x = "Fecha"
y= "Ventas" 
# full_df.plot(x=x,y=y, kind='scatter')
# plt.xticks(rotation=90)
# plt.savefig('img/full_ventas_Fecha_m.png')
# plt.close()

df.plot(x=x,y=y, kind='scatter')
plt.title("Forecasting Ventas")
a = linear_regression(df, x,y)
plt_lr(df=df, x=x, y=y, colors=('red', 'orange'), **a)
a = linear_regression(df.tail(7), x,y)
plt_lr(df=df.tail(7), x=x, y=y, colors=('peru', 'indianred'), **a)
df_j = df[pd.to_datetime(df[x]).dt.dayofweek == 5]
print_tabulate(df_j)
a = linear_regression(df_j, x,y)
plt_lr(df=df_j, x=x, y=y, colors=('slategrey', 'slategrey'), **a)

plt.xticks(rotation=90)
#plt.show()
plt.savefig(f"C:/Users/linda/OneDrive/Documentos/LINDA GAMEZ/7MO SEMESTRE/MINERÍA DE DATOS/mineria.de.datos/img2/F_Ventas.png",bbox_inches='tight')
plt.close()    

df2 = full_df.loc[(pd.to_datetime(full_df[x])>='2019-11-11') & (pd.to_datetime(full_df[x]) < '2020-01-02')]
dfs = [
    ('50D', df), 
    ('10D', df.tail(10)),
    ('30D', df.tail(30)),
    ('7D', df.tail(7)), 
    ('14D', df.tail(14)), 
    ('Miercoles', df[pd.to_datetime(df[x]).dt.dayofweek == 3]),
    ('Jueves', df[pd.to_datetime(df[x]).dt.dayofweek == 4]),
    ('Viernes', df[pd.to_datetime(df[x]).dt.dayofweek == 5]), #tiene una r2 alta/no tan mala (0.825) y r2 adj tambien (0.796), se esperan 2141
    ('Sabado', df[pd.to_datetime(df[x]).dt.dayofweek == 6]),
    ('50D-1Y', df2),
    ('10D-Y', df2.tail(10)),
    ('30D-Y', df2.tail(30)),
    ('7D-Y', df2.tail(7)),
    ('14D-Y', df2.tail(14)),
    ('Miercoles-Y', df2[pd.to_datetime(df2[x]).dt.dayofweek == 3]),
    ('Jueves-Y', df2[pd.to_datetime(df2[x]).dt.dayofweek == 4]),
    ('Viernes-Y', df2[pd.to_datetime(df2[x]).dt.dayofweek == 5]),
    ('Sabado-Y', df2[pd.to_datetime(df2[x]).dt.dayofweek == 6]),
]

lrs = [(title, linear_regression(_df,x=x,y=y), len(_df)) for title, _df in dfs]
lrs_p = [(title, lr_dict["m"]*size  + lr_dict["b"], lr_dict) for title, lr_dict, size in lrs]
sospl=[(title[1], ) for title in lrs_p]
#print(lrs_p)

llen=len(lrs_p)

for i in range(llen): 
    print("------------------------------------------------------------------------------------------------------------------")
    print(lrs_p[i])

#El que tiene un mejor pronóstico es 'Viernes' ya que de todos los que se hicieron es el que tiene una mejor r2 (max) al igual que el r2 ajustado 
# Dice que se puede alcanzar a tener una venta de 2141.0 con una r2 de 0.825 y r2 ajustada de 0.796, 
# se podría decir que es un pronóstico aceptable ya que r2 no está muy lejano al 1, sin embargo, no es la mejor r2 o tan confiable como podría ser estar arriba de 0.9

print("  Maximo de los pronosticos:  ".center(100,"-"))
print(max(sospl)) #El valor del viernes también es el valor máximo de ventas que se obtuvo de todas las opciones de pronóstico
