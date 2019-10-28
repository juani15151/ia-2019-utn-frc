import csv
import time

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import math

# LEER EL DATASET
entradas = []
with open('X_train.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        # line = [float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4])]
        media = 5  # Normalizar con media 0.
        line = [float(row[0]) - media, float(row[1]) - media, float(row[2]) - media, float(row[3]) - media, float(row[4]) - media]

        entradas.append(line)


salida_esperada = []
with open('Y_train.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for line in csv_reader:
        salida_esperada.append(int(line[0]))

assert len(entradas) == len(salida_esperada)

# TODO: No recortar las entradas.


n = int(len(entradas) * 0.75)  # Corte de los datos de entramiento.
print("Cantidad datos: " + str(len(entradas)))
p = len(entradas[0])  # Cantidad de entradas.

X = np.array(entradas[:n])
# X = X / np.sqrt(np.sum(X**2))  # Normalizar
Y = np.array(salida_esperada[:n])

X_hidden = np.array(entradas[n:])
# X_hidden = X_hidden / np.sqrt(np.sum(X_hidden**2))  # Normalizar
Y_hidden = np.array(salida_esperada[n:])

Y = Y[:, np.newaxis]


# CLASE DE LA CAPA DE LA RED

class neural_layer():

    # n_conn: Numero de conjuntos
    def __init__(self, n_conn, n_neur, act_f):
        self.act_f = act_f

        self.b = np.random.rand(1, n_neur) * 2 - 1
        self.W = np.random.rand(n_conn, n_neur) * 2 - 1


# FUNCIONES DE ACTIVACION
sigm = (lambda x: 1 / (1 + np.e ** (-x)),
        lambda x: x * (1 - x))

relu = lambda x: np.maximum(0, x)

# CREAMOS LA RED NEURONAL
def create_nn(topology, act_f):
    nn = []

    for l, layer in enumerate(topology[:-1]):
        nn.append(neural_layer(topology[l], topology[l + 1], act_f))

    return nn


# FUNCION DE ENTRENAMIENTO

# topologia # error Min. set entrenamiento / error Min set control
# topology = [p, 10, 1] # 0.2430 / 0.2511
# topology = [p, 11, 7, 1] # 0.2382 / 0.2499
topology = [p, p, 1]
# topology = [p, 10, 1]

neural_net = create_nn(topology, sigm)

l2_cost = (lambda Yp, Yr: np.mean((Yp - Yr) ** 2),
           lambda Yp, Yr: (Yp - Yr), 4, 5, 6)


def train(neural_net, X, Y, l2_cost, lr=0.5, train=True):
    out = [(None, X)]

    # Forward pass
    for l, layer in enumerate(neural_net):
        z = out[-1][1] @ neural_net[l].W + neural_net[l].b
        a = neural_net[l].act_f[0](z)

        out.append((z, a))

    if train:

        # Backward pass
        deltas = []

        for l in reversed(range(0, len(neural_net))):

            z = out[l + 1][0]
            a = out[l + 1][1]

            if l == len(neural_net) - 1:
                deltas.insert(0, l2_cost[1](a, Y) * neural_net[l].act_f[1](a))
            else:
                  deltas.insert(0, deltas[0] @ _W.T * neural_net[l].act_f[1](a))

            _W = neural_net[l].W

            # Gradient descent
            neural_net[l].b = neural_net[l].b - np.mean(deltas[0], axis=0, keepdims=True) * lr
            neural_net[l].W = neural_net[l].W - out[l][1].T @ deltas[0] * lr

    return out[-1][1]



# VISUALIZACIÓN Y TEST
neural_n = neural_net

loss = []
loss_hidden = []


for iteracion in range(4500):

    # Entrenemos a la red!
    if iteracion < 300:
        pY = train(neural_n, X, Y, l2_cost, lr=0.01)
    elif iteracion < 1500:
        pY = train(neural_n, X, Y, l2_cost, lr=0.01)
    elif iteracion < 2400:
        pY = train(neural_n, X, Y, l2_cost, lr=0.001)
    else:
        pY = train(neural_n, X, Y, l2_cost, lr=0.0001)

    if iteracion % 450 == 0:
        # print(pY)

        loss.append(l2_cost[0](pY, Y))

        error_set_oculto = train(neural_n, X_hidden, Y_hidden, l2_cost, train=False)[0][0]
        loss_hidden.append(l2_cost[0](error_set_oculto, Y_hidden))

        clear_output(wait=True)
        plt.plot(range(len(loss)), loss)
        plt.plot(range(len(loss_hidden)), loss_hidden, linestyle="dashed")
        plt.show()
        time.sleep(0.5)

print(min(loss))
print(min(loss_hidden))
print(neural_n[0].W)
# print(neural_n[1].W)
# print(neural_n[2].W)