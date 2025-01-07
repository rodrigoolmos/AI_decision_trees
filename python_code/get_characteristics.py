import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_correlation(file_path):
    """
    Lee un archivo CSV y calcula la correlación de todas las columnas con respecto a la última columna.
    Asume que el archivo no tiene encabezados explícitos y que todas las filas son datos.

    :param file_path: Ruta al archivo CSV.
    :return: None
    """
    try:
        # Leer el dataset sin encabezados
        data = pd.read_csv(file_path, header=None)
        
        # Verificar si el dataset tiene columnas
        if data.shape[1] < 2:
            raise ValueError("El dataset debe tener al menos dos columnas.")
        
        # Obtener el índice de la última columna
        last_column_index = data.columns[-1]
        
        # Calcular correlación de todas las columnas con la última columna
        correlations = data.corr()[last_column_index].drop(last_column_index)
        
        # Formatear los resultados para imprimirlos como "Col X"
        print(f"Correlación de las columnas con respecto a la columna {last_column_index}:")
        for col_index, corr_value in correlations.items():
            print(f"Col {col_index:<2} {corr_value:.6f}")
    
        print("Valores Nulos")
        print(data.isnull().sum())

        print("Valores Duplicados")
        print(data.duplicated().sum())

        print("Caracteristicas")
        print(data.describe())

        sns.pairplot(data=data,hue=data.columns[-1])
        plt.show()

    except Exception as e:
        print(f"Error: {e}")

# Ejemplo de uso
file_path = "/home/rodrigo/tfm/datasets/SoA/paper1/diabetes.csv"  # Cambiar por la ruta de tu archivo CSV
print(file_path)
correlations = calculate_correlation(file_path)
