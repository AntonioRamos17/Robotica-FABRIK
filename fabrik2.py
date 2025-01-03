import sys
from math import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import colorsys as cs
from mpl_toolkits.mplot3d import art3d
import json
import time

# Declaración de funciones
def muestra_origenes(O, final=0): 
    print('Origenes de coordenadas:')
    for i in range(len(O)):
        coords = ', '.join([f"{coord:.3f}" for coord in O[i]])
        print(f"(O{i})0\t= [{coords}]")
    if final:
        final_coords = ', '.join([f"{coord:.3f}" for coord in final])
        print(f"E.Final = [{final_coords}]")

def muestra_robot(O, obj, title, ax): 
    ax.clear()
    ax.set_xlim([-L, L])
    ax.set_ylim([-L, L])
    ax.set_zlim([-L, L])
    ax.grid(True)

    T = [np.array(o).T.tolist() for o in O] 
    handles = []
    labels = []

    for i in range(1, len(T)): 
        ax.plot([T[i-1][0], T[i][0]], [T[i-1][1], T[i][1]], [T[i-1][2], T[i][2]], color=cs.hsv_to_rgb(i / float(len(T)), 1, 1))
    
    for i in range(len(T)): 
        point, = ax.plot([T[i][0]], [T[i][1]], [T[i][2]], 'o', color=cs.hsv_to_rgb(i / float(len(T)), 1, 1))
        handles.append(point)
        labels.append(f'Articulación {i} ({round(T[i][0], 2)}, {round(T[i][1], 2)}, {round(T[i][2], 2)})')
        if i < len(T) - 1:
            alcance = a[i] 
            circulo = plt.Circle((T[i][0], T[i][1]), alcance, color=cs.hsv_to_rgb(i / float(len(T)), 1, 0.5), fill=False, linestyle='--')
            ax.add_patch(circulo)
            art3d.pathpatch_2d_to_3d(circulo, z=T[i][2], zdir="z")
    
    goal, = ax.plot([obj[0]], [obj[1]], [obj[2]], '*', label='Objetivo', color='red')
    handles.append(goal)
    labels.append(f'Objetivo ({round(obj[0], 2)}, {round(obj[1], 2)}, {round(obj[2], 2)})')

    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.legend(handles=handles, labels=labels, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)

def matriz_T(d, th, a, al): 
    return [[cos(th), -sin(th) * cos(al), sin(th) * sin(al), a * cos(th)],
            [sin(th), cos(th) * cos(al), -sin(al) * cos(th), a * sin(th)],
            [0, sin(al), cos(al), d],
            [0, 0, 0, 1]]

def cin_dir(th, a): 
    T = np.identity(4)
    o = [[0, 0, 0]]
    for i in range(len(th)):
        T = np.dot(T, matriz_T(0, th[i], a[i], 0))
        tmp = np.dot(T, [0, 0, 0, 1])
        o.append([tmp[0], tmp[1], tmp[2]])
    return o

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
num_variables = 4
if len(sys.argv) != num_variables:
    sys.exit("python " + sys.argv[0] + " x y z")
objetivo = [np.float64(i) for i in sys.argv[1:]]
print("- Posición inicial:")
O = cin_dir(th, a)
muestra_origenes(O)

target_distance = np.linalg.norm(np.subtract(objetivo, O[-1]))
L = target_distance

dist = float("inf")
prev = 0.
iteracion = 1

print("\n- Posición inicial:")
muestra_origenes(O)

fig = plt.figure(figsize=(18, 8))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')

muestra_robot(O, objetivo, f'Iteración Inicial', ax1)
muestra_robot(O, objetivo, f'Iteración Inicial', ax2)
plt.pause(1)

start_time = time.time()

# Bucle FABRIK
while dist > EPSILON and abs(prev - dist) > EPSILON / 100.:
    prev = dist
    O[-1] = objetivo
    for i in range(len(O) - 2, -1, -1):
        direction = np.subtract(O[i], O[i + 1])
        direction = direction / np.linalg.norm(direction)
        O[i] = O[i + 1] + direction * a[i]
        muestra_robot(O, objetivo, f'Iteración {iteracion} - Backward', ax1)

    plt.pause(0.5)
    
    O[0] = [0, 0, 0]
    for i in range(1, len(O)):
        direction = np.subtract(O[i], O[i - 1])
        direction = direction / np.linalg.norm(direction)
        O[i] = O[i - 1] + direction * a[i - 1]
        muestra_robot(O, objetivo, f'Iteración {iteracion} - Forward', ax2)

    plt.pause(0.5)
    
    dist = np.linalg.norm(np.subtract(objetivo, O[-1]))

    print("\n- Iteración " + str(iteracion) + ':')
    muestra_origenes(O)
    print("Distancia al objetivo = " + str(round(dist, 5)))
    iteracion += 1

end_time = time.time()
execution_time = end_time - start_time

if dist <= EPSILON:
    print("\n" + str(iteracion) + " iteraciones para converger.")
    muestra_origenes(O, objetivo)
    print("- Tiempo de ejecución: " + str(round(execution_time, 5)) + " segundos.")
else:
    print("\nNo hay convergencia tras " + str(iteracion) + " iteraciones.")
    print("- Umbral de convergencia epsilon: " + str(EPSILON))
    print("- Distancia al objetivo: " + str(round(dist, 5)))
    print("- Valores finales de las articulaciones:")
    for i in range(len(th)):
        print(" theta" + str(i + 1) + " = " + str(round(th[i], 3)))
    for i in range(len(th)):
        print(" L" + str(i + 1) + " = " + str(round(a[i], 3)))
