import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Leer el dataset en formato CSV
file_path = "/home/rodrigo/Documents/AI_decision_trees/datasets/SoA/paper5/winequality-white.csv"  # Cambia esto por la ruta a tu dataset
df = pd.read_csv(file_path)

# Barajar los datos antes de dividirlos
df = df.sample(frac=1).reset_index(drop=True)

# Asegúrate de que la última columna sea la etiqueta/target
X = df.iloc[:, :-1]  # Todas las columnas excepto la última
y = df.iloc[:, -1]   # La última columna como target

# Verificar el número de clases en la columna objetivo
num_classes = y.nunique()

# Configurar el objetivo y la métrica automáticamente
if num_classes > 2:
    objective = 'multiclass'
    metric = 'multi_logloss'
else:
    objective = 'binary'
    metric = 'binary_logloss'

# Dividir los datos en entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el dataset para LightGBM
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# Configuración de parámetros del modelo
params = {
    'objective': objective,
    'metric': metric,
    'boosting_type': 'gbdt',
    'learning_rate': 0.1,
    'num_leaves': 255,
    'num_trees': 64,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'seed': 42
}

# Si es multiclase, añadir el número de clases al parámetro
if num_classes > 2:
    params['num_class'] = num_classes

print(f"Numero de clases: {num_classes}")

# Entrenar el modelo
print("Entrenando el modelo...")
model = lgb.train(
    params,
    train_data,
    valid_sets=[train_data, test_data]
)

# Predecir en el conjunto de prueba
if num_classes > 2:
    # Para multiclase, obtenemos la clase con mayor probabilidad
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    y_pred_labels = [list(pred).index(max(pred)) for pred in y_pred]
else:
    # Para binario, usamos un umbral de 0.5
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    y_pred_labels = [1 if pred > 0.5 else 0 for pred in y_pred]

# Evaluar el modelo
accuracy = accuracy_score(y_test, y_pred_labels)
print(f"\nAccuracy en el conjunto de prueba: {accuracy:.4f}")

# Mostrar reporte de clasificación
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred_labels))
