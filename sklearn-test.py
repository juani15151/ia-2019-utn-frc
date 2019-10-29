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
        line = [float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4])]
        # line.append(line[4] / (line[0]*line[1]*line[2]*line[3]))
        # line.append(line[4]**2)
        # line.append(line[0]/line[4])
        # line.append(line[0]/line[1])
        # line.append(math.log(line[0]))
        # line.append(math.tan(line[1]))
        # line.append(math.tan(line[4]))

        entradas.append(line)


salida_esperada = []
with open('Y_train.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for line in csv_reader:
        salida_esperada.append(int(line[0]))

assert len(entradas) == len(salida_esperada)

entradas = np.array(entradas)
for i in range(len(entradas)):
    media = np.mean(entradas[i])
    maximo = np.max(entradas[i])
    for j in range(len(entradas[i])):
        entradas[i][j] = (entradas[i][j] - media) / maximo   # mover a media 0 y varianza 1

salida_esperada = np.array(salida_esperada)

## -- Mezclar el orden de los datos. -- (Sospecho que estan ordenados de alguna manera)
#
# rnd_seed = np.random.get_state()
# print(rnd_seed)
# np.random.shuffle(entradas)
# np.random.set_state(rnd_seed)
# np.random.shuffle(salida_esperada)

n = int(len(entradas) * 0.5)  # Corte de los datos de entramiento.
print("Cantidad datos: " + str(len(entradas)))
p = len(entradas[0])  # Cantidad de entradas.

X = np.array(entradas[:n])
Y = np.array(salida_esperada[:n])

X_hidden = np.array(entradas[n:])
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
sigm = (lambda x: 1 / (1 + np.e ** (-x)),  # forward
        lambda x: x * (1 - x))  # backward

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
# topology = [p, 7, 11, 13, 1]
topology = [p, 10, 25, 1]
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

try:
    for iteracion in range(450000):

        # Entrenemos a la red!
        # if iteracion < 11250:
        #     pY = train(neural_n, X, Y, l2_cost, lr=0.000001)
        # elif iteracion < 450:
        #     pY = train(neural_n, X, Y, l2_cost, lr=0.00001)
        # else:
        #     pY = train(neural_n, X, Y, l2_cost, lr=0.0000001)

        pY = train(neural_n, X, Y, l2_cost, lr=0.0001)

        if iteracion % 450 == 0:
            time.sleep(0.5)

        if iteracion % 4500 == 0:
            # print(pY)

            loss.append(l2_cost[0](pY, Y))

            error_set_oculto = train(neural_n, X_hidden, Y_hidden, l2_cost, train=False)[0][0]
            loss_hidden.append(l2_cost[0](error_set_oculto, Y_hidden))

            clear_output(wait=True)
            plt.plot(range(len(loss)), loss)
            plt.plot(range(len(loss_hidden)), loss_hidden, linestyle="dashed")
            plt.show()
            time.sleep(0.5)

finally: # Mostrar aunque se interrumpa.
    print(min(loss))
    print(min(loss_hidden))
    # print(neural_n[0].W)
# print(neural_n[1].W)
# print(neural_n[2].W)