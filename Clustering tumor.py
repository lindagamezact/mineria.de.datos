#Clustering tumor 

import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List
import numpy as np
from functools import reduce
from scipy.stats import mode
from sklearn.cluster import KMeans

def print_tabulate(df: pd.DataFrame):
    print(tabulate(df, headers=df.columns, tablefmt='github'))

#Color para gráfica 
def get_cmap(n, name="hsv"):
    """Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name."""
    return plt.cm.get_cmap(name, n)

#Grafico de dispersion de los datos originales con sus grupos 
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
    plt.title("Clasificación")
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    #plt.show()
    plt.savefig(file_path)
    plt.close()

#Distancia de los puntos 
def euclidean_distance(p_1: np.array, p_2: np.array) -> float:
    return np.sqrt(np.sum((p_2 - p_1) ** 2))

#Algoritmo k means 
def k_means(points: List[np.array], k: int, n:str):
    DIM = len(points[0])
    N = len(points)
    num_cluster = k
    iterations = 15

    x = np.array(points)
    y = np.random.randint(0, num_cluster, N)

    mean = np.zeros((num_cluster, DIM))
    for t in range(iterations):
        for k in range(num_cluster):
            mean[k] = np.mean(x[y == k], axis=0)
        for i in range(N):
            dist = np.sum((mean - x[i]) ** 2, axis=1)
            pred = np.argmin(dist)
            y[i] = pred

    for kl in range(num_cluster):
        xp = x[y == kl, 0]
        yp = x[y == kl, 1]
        plt.scatter(xp, yp)
    plt.title("Clustering") #titulo de plot 
    plt.xlabel("Lisura_Promedio") #nombre/etiqueta del eje x 
    plt.ylabel("Compacidad_Promedio") #nombre/etiqueta del eje y 
    #plt.show()
    plt.savefig(f"C:/Users/linda/OneDrive/Documentos/LINDA GAMEZ/7MO SEMESTRE/MINERÍA DE DATOS/mineria.de.datos/img2/kmeans_tumor_{n}.png") #guardar gráfica 
    plt.close()
    return mean

df = pd.read_csv("C:/Users/linda/OneDrive/Documentos/LINDA GAMEZ/7MO SEMESTRE/MINERÍA DE DATOS/mineria.de.datos/cvs/tumor.csv") #leer el archivo csv de los datos

df['Tipo'] = np.where(df['diagnosis']=="M", 'Maligno', 'Benigno') #nueva columna con la palabra completa del tipo de tumor 
df = df.drop(['diagnosis'], axis=1) #quitar la columna que solo tiene M y B 

#DataFrame con los datos que se necesitan para las gráficas y resultados del clustering 
#La x es Lisura Promedio y la y es Compacidad Promedio 
#Tipo son los 2 grupos de tipo de tumor (Maligno y Benigno) 
df_class= pd.DataFrame({"Lisura_Promedio": df["smoothness_mean"],"Compacidad_Promedio": df["compactness_mean"],"Tipo": df["Tipo"]}) 
print_tabulate(df_class.head(30)) #Imprime 30 datos del data frame acomodado 

#Grafica igual al de clasificación, es la clasificación de los datos originales según lo que muestra
scatter_group_by("C:/Users/linda/OneDrive/Documentos/LINDA GAMEZ/7MO SEMESTRE/MINERÍA DE DATOS/mineria.de.datos/img2/clusters_tumor.png", df_class, "Lisura_Promedio", "Compacidad_Promedio", "Tipo")

#se obtiene x y y con el nombre del grupo, esto es lo que se necesita para hacer los datos que van a la función de k means 
list_t = [
    (np.array(tuples[0:2]), tuples[2])
    for tuples in df_class.itertuples(index=False, name=None)
]
#print(list_t) 

#Datos que van a la función de kmeans 
points = [point for point, _ in list_t]
#labels = [label for _, label in list_t]

kn = k_means(
    points,
    2, #puse 2 porque es el número de grupos que tienen los datos originales, están divididos en Malignos y Benignos 
    "kn", #para hacer cambio del nombre de la gráfica y que no lo hiciera sobre la misma 
)
print(kn)

#Si yo pudiera decidir el número de grupos que tienen mis datos podría hacer: 
kn2 = k_means(
    points,
    4, #prueba para ver como quedaba si yo propusiera el número de agrupaciones 
    "kn2", #para hacer cambio del nombre de la gráfica y que no lo hiciera sobre la misma 
)
print(kn2)
