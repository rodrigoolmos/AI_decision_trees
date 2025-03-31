import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import lightgbm as lgb
import matplotlib.pyplot as plt
from graphviz import Digraph
from collections import deque
import struct

#######################################################
##                                                   ##
##  dataset source: https://www.kaggle.com/datasets  ##
##                                                   ##
#######################################################

# Función para procesar un árbol y exportarlo en el formato (DFS, preorden)
def process_tree(tree, n_nodes_and_leaves):
    node_leaf_value = []
    feature_index = []
    next_node_right_index = []
    leaf_or_node = []

    node_stack = [(tree, 0)]
    node_index = 0
    index_mapping = {}

    while node_stack:
        node, current_index = node_stack.pop()
        index_mapping[current_index] = node_index

        if 'split_index' in node:
            # Nodo interno
            node_leaf_value.append(node['threshold'])
            feature_index.append(node['split_feature'])
            leaf_or_node.append(1)  # Nodo interno

            left_child = node['left_child']
            right_child = node['right_child']

            left_index = current_index * 2 + 1
            right_index = current_index * 2 + 2

            node_stack.append((right_child, right_index))
            node_stack.append((left_child, left_index))

            next_node_right_index.append(right_index)

            node_index += 1
        else:
            # Hoja
            node_leaf_value.append(int(round(node['leaf_value'] * 1000000)))
            feature_index.append(0)  # No es relevante en hojas
            leaf_or_node.append(0)  # Hoja
            next_node_right_index.append(0)  # No es relevante en hojas

            node_index += 1

    # Remapear índices para que sean correctos
    next_node_right_index = [index_mapping.get(idx, 0) for idx in next_node_right_index]

    # Rellenar con ceros hasta n_nodes_and_leaves
    while len(node_leaf_value) < n_nodes_and_leaves:
        node_leaf_value.append(0.0)
        feature_index.append(0)
        next_node_right_index.append(0)
        leaf_or_node.append(0)

    return node_leaf_value, feature_index, next_node_right_index, leaf_or_node

# Función para procesar el modelo y extraer cada árbol
def parse_model(booster, n_nodes_and_leaves):
    model = booster.dump_model()
    trees = []
    
    for tree_info in model['tree_info']:
        tree = tree_info['tree_structure']
        tree_data = process_tree(tree, n_nodes_and_leaves)
        trees.append(tree_data)
    
    return trees

# Función que exporta la información de los árboles a un fichero header en C (trees.h)
def export_trees_to_header(trees, num_nodes_and_leaves, header_file_name):
    tree_lines = []
    num_trees = len(trees)
    tree_lines.append("// Estructuras de árbol generadas automáticamente")
    tree_lines.append("#ifndef TREES_H")
    tree_lines.append("#define TREES_H")
    tree_lines.append("")
    tree_lines.append(f"const unsigned long long tree[{num_trees}][{num_nodes_and_leaves}] = {{")
    for tree in trees:
        node_leaf_value, feature_index, next_node_right_index, leaf_or_node = tree
        node_values = []
        for i in range(num_nodes_and_leaves):
            # Se empaqueta la información de cada nodo en el mismo orden que para el fichero binario:
            # 1 byte: leaf_or_node, 1 byte: feature_index, 1 byte: next_node_right_index, 1 byte: 0xff, luego 4 bytes: valor
            if leaf_or_node[i] == 0:
                # Hoja: se guarda el valor como entero
                packed_val = struct.pack("B B B B i", leaf_or_node[i], feature_index[i], next_node_right_index[i], 0xff, int(node_leaf_value[i]))
            else:
                # Nodo interno: se guarda el valor como float
                packed_val = struct.pack("B B B B f", leaf_or_node[i], feature_index[i], next_node_right_index[i], 0xff, float(node_leaf_value[i]))
            # Interpretar los 8 bytes como un entero sin signo de 64 bits (little-endian)
            node_int = struct.unpack("<Q", packed_val)[0]
            # Formatear en hexadecimal con 16 dígitos
            node_values.append(f"0x{node_int:016X}")
        tree_line = "    {" + ", ".join(node_values) + "}"
        tree_lines.append(tree_line + ",")
    tree_lines.append("};")
    tree_lines.append("")
    tree_lines.append("#endif // TREES_H")
    
    with open(header_file_name, "w") as f:
        f.write("\n".join(tree_lines))
    print(f"Header exportado en: {header_file_name}")

# Función para entrenar, evaluar y exportar el modelo (binario y header)
def train_model_parse_and_store(data, output_model_name, num_trees=200, learning_rate=0.1, n_jobs=72, test_size=0.8, max_depth=10):
    # Se genera el nombre del header a partir del nombre del modelo binario
    header_file_name = output_model_name.rsplit('.', 1)[0] + '.h'
    
    # Separar características y etiqueta
    X = data.drop('Outcome', axis=1)
    y = data['Outcome']
    num_leaves = int(2**(max_depth - 1))
    num_nodes_and_leaves = num_leaves * 2

    # División del dataset en entrenamiento y prueba
    train_size = 1 - test_size
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, train_size=train_size, random_state=42)

    lightgbm_model = lgb.LGBMClassifier(objective='binary', learning_rate=learning_rate, 
                                        n_estimators=num_trees, n_jobs=n_jobs, num_leaves=num_leaves, max_depth=max_depth)
    lightgbm_model.fit(X_train, y_train)
    y_pred_lgb = lightgbm_model.predict(X_test)
    lgb_accuracy = accuracy_score(y_test, y_pred_lgb)
    lgb_auc = roc_auc_score(y_test, y_pred_lgb)

    print(f"LightGBM - Número de árboles: {lightgbm_model.n_estimators}")
    print(f"LightGBM - Accuracy: {lgb_accuracy:.4f}, AUC: {lgb_auc:.4f}")

    # Procesar el modelo para extraer la estructura de los árboles
    trees = parse_model(lightgbm_model.booster_, num_nodes_and_leaves)

    # Exportar la estructura en un fichero binario
    with open(output_model_name, 'wb') as f:
        f.write(b'model')
        for node_leaf_value, feature_index, next_node_right_index, leaf_or_node in trees:
            for index in range(len(feature_index)):
                if leaf_or_node[index] == 0:
                    f.write(struct.pack('B', leaf_or_node[index]))
                    f.write(struct.pack('B', feature_index[index]))
                    f.write(struct.pack('B', next_node_right_index[index]))
                    f.write(struct.pack('B', 0xff))
                    f.write(struct.pack('i', int(node_leaf_value[index])))
                else:
                    f.write(struct.pack('B', leaf_or_node[index]))
                    f.write(struct.pack('B', feature_index[index]))
                    f.write(struct.pack('B', next_node_right_index[index]))
                    f.write(struct.pack('B', 0xff))
                    f.write(struct.pack('f', node_leaf_value[index]))
    print(f"Modelo binario exportado en: {output_model_name}")

    # Exportar la información en un fichero header en formato C
    export_trees_to_header(trees, num_nodes_and_leaves, header_file_name)

def preprocess_data(data):
    data.replace({'M': 0, 'F': 1, 'M ': 0, 'F ': 1, ' M ': 0, ' F ': 1, ' M': 0, ' F': 1,
                  'Yes': 1, 'No': 0, 'YES': 1, 'NO': 0}, inplace=True)
    return data

# Ejemplos de carga y procesamiento de datasets

# ----------------- Diabetes -----------------
path = "./datasets/diabetes.csv"
column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 
                'DiabetesPedigreeFunction', 'Age', 'Outcome']
data = pd.read_csv(path, names=column_names)
train_model_parse_and_store(data, "./trained_models/diabetes.model",
                            num_trees=128, learning_rate=0.5, n_jobs=72, test_size=0.2, max_depth=8)

# ----------------- Heart Attack -----------------
path = "./datasets/Heart_Attack.csv"
column_names = ['age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg', 'thalachh', 
                'exng', 'oldpeak', 'slp', 'caa', 'thall', 'Outcome']
data = pd.read_csv(path, names=column_names)
train_model_parse_and_store(data, "./trained_models/heart_attack.model",
                            num_trees=128, learning_rate=0.5, n_jobs=72, test_size=0.2, max_depth=8)

# ----------------- Lung Cancer -----------------
path = "./datasets/Lung_Cancer_raw.csv"
column_names = ['GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE', 
                'CHRONIC_DISEASE', 'FATIGUE', 'ALLERGY', 'WHEEZING', 'ALCOHOL_CONSUMING', 
                'COUGHING', 'SHORTNESS_OF_BREATH', 'SWALLOWING_DIFFICULTY', 'CHEST_PAIN', 'Outcome']
data = pd.read_csv(path, names=column_names)
processed_data = preprocess_data(data)
output_path = "./datasets/Lung_Cancer_processed_dataset.csv"
processed_data.to_csv(output_path, index=False, header=False)
train_model_parse_and_store(processed_data, "./trained_models/lung_cancer.model",
                            num_trees=128, learning_rate=0.5, n_jobs=72, test_size=0.1, max_depth=8)

# ----------------- Anemia -----------------
path = "./datasets/anemia.csv"
column_names = ['Number', 'Sex', 'Red_Pixel', 'Green_pixel', 'Blue_pixel', 'Hb', 'Outcome']
data = pd.read_csv(path, names=column_names)
sorted_data = data.iloc[:, 1:]  # Eliminamos la primera columna inútil
processed_data = preprocess_data(sorted_data)
output_path = "./datasets/anemia_processed_dataset.csv"
processed_data.to_csv(output_path, index=False, header=False)
train_model_parse_and_store(processed_data, "./trained_models/anemia.model",
                            num_trees=128, learning_rate=0.5, n_jobs=72, test_size=0.6, max_depth=8)

# ----------------- Alzheimer -----------------
path = "./datasets/alzheimers_disease_data.csv"
column_names = ['PatientID', 'Age', 'Gender', 'Ethnicity', 'EducationLevel', 'BMI', 'Smoking', 
                'AlcoholConsumption', 'PhysicalActivity', 'DietQuality', 'SleepQuality', 
                'FamilyHistoryAlzheimers', 'CardiovascularDisease', 'Diabetes', 'Depression', 
                'HeadInjury', 'Hypertension', 'SystolicBP', 'DiastolicBP', 'CholesterolTotal', 
                'CholesterolLDL', 'CholesterolHDL', 'CholesterolTriglycerides', 'MMSE', 
                'FunctionalAssessment', 'MemoryComplaints', 'BehavioralProblems', 'ADL', 
                'Confusion', 'Disorientation', 'PersonalityChanges', 'DifficultyCompletingTasks', 
                'Forgetfulness', 'Outcome', 'DoctorInCharge']
data = pd.read_csv(path, names=column_names)
sorted_data = data.iloc[:, 1:]  # Eliminar PatientID
sorted_data = sorted_data.iloc[:, :-1]  # Eliminar DoctorInCharge
output_path = "./datasets/alzheimers_processed_dataset.csv"
sorted_data.to_csv(output_path, index=False, header=False)
train_model_parse_and_store(sorted_data, "./trained_models/alzheimers.model",
                            num_trees=128, learning_rate=0.5, n_jobs=72, test_size=0.6, max_depth=8)
