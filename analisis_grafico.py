import csv
from io import open

import matplotlib.pyplot as plt


def main():
    # Lectura
    entradas = []
    with open('X_train.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            line = [float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4])]
            entradas.append(line)


    salida_esperada = []
    with open('Y_train.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for line in csv_reader:
            salida_esperada.append(int(line[0]))

    assert len(entradas) == len(salida_esperada)

    # Transformacion
    transformado = []
    for x in entradas[:300]:
        transformado.append([
            x[4] ** 2,
            x[0] * x[1] * x[2] * x[3]
        ])

    # Grafico
    colors = ['k', 'r']
    markers = ['x', 'o']
    cantidad_caracteristicas = len(transformado[0])
    for x1 in range(0, cantidad_caracteristicas):
        for x2 in range(x1, cantidad_caracteristicas):
            if x1 == x2:
                continue
            for i in range(0, len(transformado)):
                entrada_actual = transformado[i]
                plt.scatter(entrada_actual[x1], entrada_actual[x2], c=colors[salida_esperada[i]], marker=markers[salida_esperada[i]])
            # plt.plot()  # Muestra el grafico
            plt.savefig("datos/test/x{0}-x{1}.png".format(x1, x2))
            plt.clf()  # Limpia el grafico


if __name__ == "__main__":
    print("Inicio")
    main()
    print("Fin")
