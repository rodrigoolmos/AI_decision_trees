#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lee un fichero binario con:
  - 5 bytes de cabecera ASCII
  - matriz de uint64_t de tamaño ROWS×COLS
y escribe en texto la declaración C con valores en hexadecimal:
  uint64_t array_leido[ROWS][COLS] = {{…},{…},…};
"""

import struct

# Parámetros de la matriz
HEADER_SIZE = 5           # bytes de cabecera ASCII
ROWS = 128                # número de filas
COLS = 256                # número de columnas
UINT64_SIZE = 8           # bytes por uint64_t
ROW_BYTES = COLS * UINT64_SIZE
ROW_FMT = '<' + 'Q' * COLS  # little-endian uint64_t × COLS

INPUT_FILE = '/home/rodrigo/Documents/AI_decision_trees/trained_models/caracterizacion_frec.model'
OUTPUT_FILE = '/home/rodrigo/Documents/AI_decision_trees/trained_models/caracterizacion_frec.c'

def read_matrix(path):
    """Lee la cabecera ASCII y devuelve la matriz como lista de tuplas."""
    with open(path, 'rb') as f:
        header = f.read(HEADER_SIZE).decode('ascii', errors='replace')
        print(f"Cabecera ASCII leída: {header!r}")
        matrix = []
        for fila in range(ROWS):
            chunk = f.read(ROW_BYTES)
            if len(chunk) != ROW_BYTES:
                raise EOFError(f"Se esperaban {ROW_BYTES} bytes para la fila {fila}, "
                               f"pero llegaron {len(chunk)}.")
            row = struct.unpack(ROW_FMT, chunk)
            matrix.append(row)
    return matrix

def write_c_array_hex(matrix, path):
    """Escribe la matriz en formato C con cada valor en hexadecimal."""
    with open(path, 'w') as f:
        f.write(f"uint64_t array_leido[{ROWS}][{COLS}] = {{\n")
        for i, row in enumerate(matrix):
            # formatear cada valor como 0xhhhhhhhhhhhhhhhh
            hex_vals = ', '.join(f"0x{val:016x}" for val in row)
            sep = ',' if i < ROWS - 1 else ''
            f.write(f"    {{{hex_vals}}}{sep}\n")
        f.write("};\n")
    print(f"Matriz hexadecimal escrita en {path!r}.")

def main():
    matrix = read_matrix(INPUT_FILE)
    write_c_array_hex(matrix, OUTPUT_FILE)

if __name__ == '__main__':
    main()
