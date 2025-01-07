import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from tkinter import Tk, filedialog
# Variables globales para almacenar los puntos
class_1_points = []
class_2_points = []
spray_density = 10  # Cantidad de puntos por clic
spray_radius = 0.5  # Radio del spray
# Función para manejar clics del ratón
def onclick(event):
    global spray_density, spray_radius
    if event.inaxes:
        # Determinar la clase según el botón del ratón
        current_class = 1 if event.button == 1 else 0
        for _ in range(spray_density):
            # Generar puntos aleatorios alrededor del clic dentro del radio
            offset_x, offset_y = np.random.uniform(-spray_radius, spray_radius, 2)
            new_x = event.xdata + offset_x
            new_y = event.ydata + offset_y
            # Validar si el punto está dentro del frame de dibujo
            if new_x >= -6 and new_x <= 6 and new_y >= -6 and new_y <= 6:
                if current_class == 1:
                    class_1_points.append([new_x, new_y])
                    event.inaxes.plot(new_x, new_y, 'bo', markersize=2)  # Puntos azules para clase 1
                else:
                    class_2_points.append([new_x, new_y])
                    event.inaxes.plot(new_x, new_y, 'ro', markersize=2)  # Puntos rojos para clase 0
        event.inaxes.figure.canvas.draw()
# Función para finalizar la entrada de puntos
def finish(event):
    plt.close()
# Función para cargar datos desde un CSV
def load_from_csv():
    root = Tk()
    root.withdraw()  # Ocultar la ventana principal
    filepath = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if filepath:
        data = pd.read_csv(filepath)
        if 'Feature 1' in data.columns and 'Feature 2' in data.columns and 'Class' in data.columns:
            X = data[["Feature 1", "Feature 2"]].values
            y = data["Class"].values
            print(f"Datos cargados desde {filepath}")
            return X, y
        else:
            print("El archivo CSV no tiene las columnas necesarias (Feature 1, Feature 2, Class).")
    return None, None
# Interfaz para decidir entre cargar CSV o dibujar puntos
def main():
    global spray_density, spray_radius
    print("Selecciona una opción:")
    print("1: Dibujar nuevos puntos")
    print("2: Cargar puntos desde un archivo CSV")
    choice = input("Ingresa 1 o 2: ")
    if choice == '1':
        # Interfaz para dibujar puntos
        fig, ax = plt.subplots(figsize=(6, 6))
        plt.title("Haz clic para agregar puntos. Azul: Clase 1, Rojo: Clase 0")
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        # Agregar sliders para densidad y radio
        ax_density = plt.axes([0.2, 0.02, 0.4, 0.03], facecolor='lightgrey')
        ax_radius = plt.axes([0.2, 0.06, 0.4, 0.03], facecolor='lightgrey')
        slider_density = Slider(ax_density, 'Spray Density', 1, 100, valinit=spray_density, valstep=1)
        slider_radius = Slider(ax_radius, 'Spray Radius', 0.1, 5.0, valinit=spray_radius, valstep=0.1)
        # Actualizar valores desde los sliders
        def update_density(val):
            global spray_density
            spray_density = int(slider_density.val)
            print(f"Densidad del spray actualizada: {spray_density}")
        def update_radius(val):
            global spray_radius
            spray_radius = slider_radius.val
            print(f"Radio del spray actualizado: {spray_radius}")
        slider_density.on_changed(update_density)
        slider_radius.on_changed(update_radius)
        fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()
        # Convertir puntos a numpy arrays
        X = np.vstack((class_1_points, class_2_points)) if class_1_points or class_2_points else np.empty((0, 2))
        y = np.hstack((np.ones(len(class_1_points)), np.zeros(len(class_2_points)))) if class_1_points or class_2_points else np.empty(0)
    elif choice == '2':
        X, y = load_from_csv()
    else:
        print("Opción inválida.")
        return
    if X is None or y is None or X.size == 0 or y.size == 0:
        print("No hay datos disponibles para procesar.")
        return
    # Exportar los datos a un archivo CSV
    def export_to_csv(X, y, filename="features.csv"):
        if X.size == 0 or y.size == 0:
            print("No hay datos para exportar.")
            return
        data = pd.DataFrame(X, columns=["Feature 1", "Feature 2"])
        data["Class"] = y.astype(int)
        data.to_csv(filename, index=False)
        print(f"Datos exportados a {filename}")
    export_to_csv(X, y)
    # Función para graficar regiones de decisión
    def plot_decision_regions(model, X, y, feature_pair):
        feature1, feature2 = feature_pair
        x_min, x_max = X[:, feature1].min() - 1, X[:, feature1].max() + 1
        y_min, y_max = X[:, feature2].min() - 1, X[:, feature2].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                             np.arange(y_min, y_max, 0.1))
        grid = np.c_[xx.ravel(), yy.ravel()]
        # Predicción del modelo
        Z = model.predict(grid)
        Z = (Z > 0.5).astype(int).reshape(xx.shape)
        # Graficar regiones de decisión
        plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
        plt.scatter(X[:, feature_pair[0]], X[:, feature_pair[1]], c=y, edgecolor='k', cmap=plt.cm.RdYlBu)
        plt.xlabel(f"Feature {feature1}")
        plt.ylabel(f"Feature {feature2}")
        plt.title("LightGBM - Visualización de regiones de decisión")
        plt.show()
    # Dividir datos
    if X.size > 0 and y.size > 0:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # Crear y entrenar modelo LightGBM
        train_data = lgb.Dataset(X_train, label=y_train)
        params = {
            'objective': 'binary',
            'metric': 'binary_error',
            'boosting_type': 'gbdt',
            'num_leaves': 256,
            'num_trees': 128,
            'learning_rate': 0.1,
            'feature_fraction': 0.9,
        }
        model = lgb.train(params, train_data, num_boost_round=100)
        # Graficar regiones de decisión para las características seleccionadas
        plot_decision_regions(model, X, y, feature_pair=(0, 1))

        # Predecir en el conjunto de prueba
        y_pred = model.predict(X_test, num_iteration=model.best_iteration)
        y_pred_binary = [1 if pred > 0.5 else 0 for pred in y_pred]

        # Evaluar el modelo
        accuracy = accuracy_score(y_test, y_pred_binary)
        print(f"\nAccuracy en el conjunto de prueba: {accuracy:.4f}")

    else:
        print("No hay suficientes datos para entrenar el modelo.")
if __name__ == "__main__":
    main()
