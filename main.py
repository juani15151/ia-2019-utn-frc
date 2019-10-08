import csv
from io import open
import matplotlib.pyplot as plt


def main():
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

    colors = ['r', 'b']

    cantidad_caracteristicas = len(entradas[0])
    for x1 in range(0, cantidad_caracteristicas):
        for x2 in range(x1, cantidad_caracteristicas):
            for i in range(0, len(entradas)):
                entrada_actual = entradas[i]
                plt.scatter(entrada_actual[x1], entrada_actual[x2], c=colors[salida_esperada[i]])
            # plt.plot()  # Muestra el grafico
            plt.savefig("datos/x" + str(x1) + "-" + "x" + str(x2) + ".png")
            plt.clf()  # Limpia el grafico


    # Informacion obtenida:
    # X4 * Xi No aporta informacion



if __name__ == "__main__":
    print("Inicio")
    main()
    print("Fin")
