#Clasificacion tumor 

from numpy.lib import index_tricks
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List
import numpy as np
from functools import reduce
from scipy.stats import mode

def print_tabulate(df: pd.DataFrame):
    print(tabulate(df, headers=df.columns, tablefmt='github'))

def get_cmap(n, name="hsv"):
    """Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name."""
    return plt.cm.get_cmap(name, n)

#Grafico de dispersion de la clasificación de los datos
def scatter_group_by(
    file_path: str, df: pd.DataFrame, x_column: str, y_column: str, label_column: str
):
    fig, ax = plt.subplots()
    labels = pd.unique(df[label_column])
    cmap = get_cmap(len(labels) + 1)
    for i, label in enumerate(labels):
        filter_df = df.query(f"{label_column} == '{label}'")
        ax.scatter(filter_df[x_column], filter_df[y_column], label=label, color=cmap(i))
    ax.legend()
    plt.title("Clasificación") #Titulo 
    plt.xlabel(x_column) #nombre/ etiqueta del eje x
    plt.ylabel(y_column) #nombre/ etiqueta del eje y
    plt.savefig(file_path)
    plt.close()

def euclidean_distance(p_1: np.array, p_2: np.array) -> float:
    return np.sqrt(np.sum((p_2 - p_1) ** 2))

def k_nearest_neightbors(
    points: List[np.array], labels: np.array, input_data: List[np.array], k: int
):
    input_distances = [
        [euclidean_distance(input_point, point) for point in points]
        for input_point in input_data
    ]
    points_k_nearest = [
        np.argsort(input_point_dist)[:k] for input_point_dist in input_distances
    ]
    return [
        mode([labels[index] for index in point_nearest])
        for point_nearest in points_k_nearest
    ]

#Leer el csv de los datos de tumor 
df = pd.read_csv("C:/Users/linda/OneDrive/Documentos/LINDA GAMEZ/7MO SEMESTRE/MINERÍA DE DATOS/mineria.de.datos/cvs/tumor.csv")

df['Tipo'] = np.where(df['diagnosis']=="M", 'Maligno', 'Benigno') #crear nueva columna para que no diga solo M y B, sino Maligno y Benigno
df = df.drop(['diagnosis'], axis=1) #quitar la columna que solo tiene M y B 
#print_tabulate(df.head())
#print(df.columns)

#Crear DataFrame solo con los datos de X y Y acomodados junto con el tipo de tumor: 
df_class= pd.DataFrame({"Tipo": df["Tipo"], "Lisura_Promedio": df["smoothness_mean"],"Compacidad_Promedio": df["compactness_mean"]})
print_tabulate(df_class.head(30))

scatter_group_by("C:/Users/linda/OneDrive/Documentos/LINDA GAMEZ/7MO SEMESTRE/MINERÍA DE DATOS/mineria.de.datos/img2/tumor.png", df_class, "Lisura_Promedio", "Compacidad_Promedio", "Tipo")
list_t = [
    (np.array(tuples[1:2]), tuples[0])
    for tuples in df_class.itertuples(index=False, name=None)
]
points = [point for point, _ in list_t]
labels = [label for _, label in list_t]

kn = k_nearest_neightbors(
    points,
    labels,
    [np.array([.25, .16]), np.array([.05, .12]), np.array([.1, .1]), np.array([.08, .12]),np.array([.23, .10]),np.array([.05, .08]) ], #puntos que quisiera ver de que tipo lo clasifica
    3, #al quedar la gráfica muy pegada decidí poner k=3 (# chico) para que no tomara muchos puntos si estos ya estaban cerca (no tienen varianza muy alta)
)

print(kn)
