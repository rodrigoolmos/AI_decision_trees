#!/usr/bin/env python3
import pandas as pd
import argparse

def csv_to_header(csv_file, header_file):
    # Leer el CSV sin encabezados (se asume que cada fila es una entrada)
    df = pd.read_csv(csv_file, header=None)
    
    N_FEATURE = 32
    N_ITEMS = len(df)
    
    lines = []
    # Encabezado del header
    lines.append("#ifndef DATA_H")
    lines.append("#define DATA_H")
    lines.append("")
    lines.append("#include <stdint.h>")
    lines.append(f"#define N_FEATURE {N_FEATURE}")
    lines.append(f"#define N_ITEMS {N_ITEMS}")
    lines.append("")
    lines.append("struct feature {")
    lines.append("    float features[N_FEATURE];")
    lines.append("    uint8_t prediction;")
    lines.append("};")
    lines.append("")
    lines.append("struct feature features[N_ITEMS] = {")
    
    # Para cada fila del CSV
    for _, row in df.iterrows():
        # Convertir la fila a lista
        row_list = row.tolist()
        # Si la fila tiene 33 o más columnas: los primeros 32 son features y el 33º es la predicción.
        # En caso contrario, se usan los valores disponibles y se asigna 0 para la predicción.
        if len(row_list) >= N_FEATURE + 1:
            features_list = row_list[:N_FEATURE]
            prediction = row_list[N_FEATURE]
        else:
            features_list = row_list[:(N_FEATURE-1)]
            prediction = row_list[-1]
        
        # Rellenar con ceros si hay menos de 32 features
        if len(features_list) < N_FEATURE:
            features_list += [0.0] * (N_FEATURE - len(features_list))
        
        # Convertir los valores a formato float con 6 decimales
        features_str = ", ".join(f"{float(val):.6f}" for val in features_list)
        try:
            pred_int = int(float(prediction))
        except:
            pred_int = 0
        
        # Formatear la línea correspondiente
        lines.append(f"    {{ {{ {features_str} }}, {pred_int} }},")
    
    lines.append("};")
    lines.append("")
    lines.append("#endif // DATA_H")
    
    with open(header_file, "w") as f:
        f.write("\n".join(lines))
    
    print(f"Header file exported to: {header_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convierte un CSV a un fichero header en C con el formato especificado."
    )
    parser.add_argument("csv_file", help="Ruta del archivo CSV de entrada")
    parser.add_argument("header_file", help="Ruta del fichero header de salida")
    
    args = parser.parse_args()
    csv_to_header(args.csv_file, args.header_file)
