#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Robótica Computacional - 
# Grado en Ingeniería Informática (Cuarto)
# Práctica: Resolución de la cinemática inversa mediante CCD
#           (Cyclic Coordinate Descent).

import sys
from math import *
import numpy as np
import matplotlib.pyplot as plt
import colorsys as cs
import json  # Para cargar configuración desde un archivo
import time  # Para gestionar el cierre automático de gráficos

# ******************************************************************************
# Declaración de funciones

def muestra_origenes(O, final=0):
    # Muestra los orígenes de coordenadas para cada articulación
    print('Origenes de coordenadas:')
    for i in range(len(O)):
        print('(O' + str(i) + ')0\t= ' + str([round(j, 3) for j in O[i]]))
    if final:
        print('E.Final = ' + str([round(j, 3) for j in final]))

def muestra_robot(O, obj):
    # Crear una figura con tamaño personalizado
    plt.figure(figsize=(12, 8))  
    plt.xlim(-L, L)
    plt.ylim(-L, L)
    
    # Mostrar cuadrícula y etiquetas de los ejes
    plt.grid(True)

    T = [np.array(o).T.tolist() for o in O]
    
    # Lista para almacenar las etiquetas de la leyenda
    handles = []
    labels = []

    # Dibujar las líneas entre las articulaciones
    for i in range(1, len(T)):
        plt.plot([T[i-1][0], T[i][0]], [T[i-1][1], T[i][1]], color=cs.hsv_to_rgb(i / float(len(T)), 1, 1))
    
    # Dibujar los puntos de las articulaciones y agregar las coordenadas a la leyenda
    for i in range(len(T)):
        point, = plt.plot(T[i][0], T[i][1], 'o', color=cs.hsv_to_rgb(i / float(len(T)), 1, 1))
        handles.append(point)
        labels.append(f'Articulación {i} ({round(T[i][0], 2)}, {round(T[i][1], 2)})')

    # Dibujar el objetivo final
    goal, = plt.plot(obj[0], obj[1], '*', label='Objetivo', color='red')
    handles.append(goal)
    labels.append(f'Objetivo ({round(obj[0], 2)}, {round(obj[1], 2)})')
    
    # Añadir título
    plt.title('Visualización del Robot y su Objetivo', fontsize=16, fontweight='bold')
    
    # Añadir la leyenda debajo del gráfico
    plt.legend(handles=handles, labels=labels, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)

    plt.pause(0.0001) 
    plt.show(block=False)
    
    time.sleep(2)  
    plt.close()


def matriz_T(d, th, a, al):
    return [[cos(th), -sin(th) * cos(al),  sin(th) * sin(al), a * cos(th)],
            [sin(th),  cos(th) * cos(al), -sin(al) * cos(th), a * sin(th)],
            [      0,            sin(al),           cos(al),           d],
            [      0,                  0,                  0,           1]]

def cin_dir(th, a):
    # Sea 'th' el vector de thetas
    # Sea 'a' el vector de longitudes
    T = np.identity(4)
    o = [[0, 0]]
    for i in range(len(th)):
        T = np.dot(T, matriz_T(0, th[i], a[i], 0))
        tmp = np.dot(T, [0, 0, 0, 1])
        o.append([tmp[0], tmp[1]])
    return o

# ******************************************************************************

# Cargar configuración desde archivo
config_file = "./json/robot_data.json"
try:
    with open(config_file, 'r') as f:
        config = json.load(f)
        EPSILON = config["EPSILON"] 
        articulaton_limit = config["articulaton_limit"]  
        articulaton_limit = [[radians(i) for i in j] for j in articulaton_limit] 
        prismatic_limit = config["prismatic_limit"] 
        th = config["th"] 
        th = [radians(i) for i in th]
        a = config["a"] 
        articulation_type = config["articulation_type"] 
except Exception as e:
    sys.exit(f"Error al cargar la configuración: {e}")


# introducción del punto para la cinemática inversa
num_variables = 3
if len(sys.argv) != num_variables:
    sys.exit("python " + sys.argv[0] + " x y")
objetivo = [np.float64(i) for i in sys.argv[1:]]

print("- Posición inicial:")
O = cin_dir(th, a)
muestra_origenes(O)

# Calcular longitud total para representación gráfica
target_distance = np.linalg.norm(np.subtract(objetivo, O[-1][-1]))
L = target_distance

# Calculamos la posición inicial

dist = float("inf")
prev = 0.
iteracion = 1

# Mostrar la posición inicial antes de iniciar el bucle
print("\n- Posición inicial:")
muestra_origenes(O)
muestra_robot(O, objetivo)

start_time = time.time()

# Bucle FABRIK
while dist > EPSILON and abs(prev - dist) > EPSILON / 100.:
    prev = dist

    # Paso hacia adelante
    O[-1] = objetivo
    for i in range(len(O) - 2, -1, -1):
        direction = np.subtract(O[i], O[i + 1])
        direction = direction / np.linalg.norm(direction)
        O[i] = O[i + 1] + direction * a[i]

    # Paso hacia atrás
    O[0] = [0, 0]
    for i in range(1, len(O)):
        direction = np.subtract(O[i], O[i - 1])
        direction = direction / np.linalg.norm(direction)
        O[i] = O[i - 1] + direction * a[i - 1]

    # Calcular la distancia al objetivo
    dist = np.linalg.norm(np.subtract(objetivo, O[-1]))

    # Mostrar la iteración actual
    print("\n- Iteración " + str(iteracion) + ':')
    muestra_origenes(O)
    muestra_robot(O, objetivo)
    print("Distancia al objetivo = " + str(round(dist, 5)))

    iteracion += 1

end_time = time.time()
execution_time = end_time - start_time

if dist <= EPSILON:
    print("\n" + str(iteracion) + " iteraciones para converger.")
    print("- Tiempo de ejecución: " + str(round(execution_time, 5)) + " segundos.")

else:
    print("\nNo hay convergencia tras " + str(iteracion) + " iteraciones.")
print("- Umbral de convergencia epsilon: " + str(EPSILON))
print("- Distancia al objetivo:          " + str(round(dist, 5)))
print("- Valores finales de las articulaciones:")
for i in range(len(th)):
    print("  theta" + str(i + 1) + " = " + str(round(th[i], 3)))
for i in range(len(th)):
    print("  L" + str(i + 1) + "     = " + str(round(a[i], 3)))