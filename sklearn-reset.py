import csv

import numpy as np
import matplotlib.pyplot as plt

# LEER EL DATASET
entradas = []
with open('X_train.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        line = [float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4])]
        entradas.append(line)

p = len(line) # Cantidad de caracteristicas.

salida_esperada = []
with open('Y_train.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for line in csv_reader:
        salida_esperada.append(int(line[0]))

assert len(entradas) == len(salida_esperada)

n = 1500 # Cantidad de registros

X = np.array(entradas[:n])
X_hidden = np.array(entradas[n:])
Y = np.array(salida_esperada[:n])
Y_hidden = np.array(salida_esperada[n:])

# Convertir los Y en matriz columna.
Y = Y[:, np.newaxis]
Y_hidden = Y_hidden[:, np.newaxis]



# CLASE DE LA CAPA DE LA RED

class neural_layer():

    # TODO: Poner el procesamiento hacia adelante y atrás acá.

    def __init__(self, n_conn, n_neur, act_f):
        """

        :param n_conn: Numero de conexiones entrantes desde la capa anterior.
        :param n_neur: Numero de neuronas.
        :param act_f: Funcion de activacion.
        """
        self.act_f = act_f

        # TODO: Convertir a un unico vector de pesos W y agregar una entrada siempre en 1.
        # Vector B para cada neurona. (Vector columna con la entrada unitaria)
        self.b = np.random.rand(1, n_neur) * 2 - 1  # Random normalizado a valores entre -1 y 1.

        # Matriz de pesos para cada neurona.
        self.W = np.random.rand(n_conn, n_neur) * 2 - 1  # Random normalizado a valores entre -1 y 1.


# FUNCIONES DE ACTIVACION

# Se utilizan para introducir no linealidad
# y poder combinar neuronas sin que se reduzcan a uno única.

sigm = (
    lambda x: 1 / (1 + np.e ** (-x)),  # Funcion sigmoide
    lambda x: x * (1 - x)  # Derivada de la funcion de activacion
)

# Opcional. Para usarla buscar la derivada.
# relu = lambda x: np.maximum(0, x)  # fun relu _/ (y=0 hasta x=0 y luego y=x)


# CREAMOS LA RED NEURONAL

# Ejemplo manual:
# Util si se quieren combinar funciones de activacion.
# l0 = neural_layer(p, 4, sigm)
# l1 = neural_layer(4, 8, sigm)


# ...

def create_nn(topology, act_f):
    nn = []

    for l, layer in enumerate(topology[:-1]):  # Descartamos el ultimo para evitar IndexOutOfBounds
        nn.append(neural_layer(topology[l], topology[l + 1], act_f))

    return nn


# FUNCION DE ENTRENAMIENTO

# Siempre empieza con la cantidad de caracteristicas,
# y sale con 1 neurona porque clasifica de forma binaria.
topology = [p, 10, 20, 1]

neural_net = create_nn(topology, sigm)

# Funcion de costo
l2_cost = (
    lambda Yp, Yr: np.mean((Yp - Yr) ** 2),  # Error cuadratico medio.
    lambda Yp, Yr: (Yp - Yr)  # Derivada del error cuadratico medio.
)


def train(neural_net, X, Y, l2_cost, lr=0.5, train=True):
    # El vector de entradas es inicialmente los datos.
    # Despues son las salidas de la capa anterior.
    out = [(None, X)]

    # Forward pass
    for l, layer in enumerate(neural_net):
        # Calculamos la suma ponderada (W1*X1 + W2*X2 + ... + B)
        z = out[-1][1] @ neural_net[l].W + neural_net[l].b
        # Pasamos el resultado por la funcion de activacion.
        a = neural_net[l].act_f[0](z)

        # Guardamos salidas de la neurona y la activacion para la proxima capa.
        out.append((z, a))

    if train:

        # Backward pass
        deltas = []  # Valores en funcion al error.

        # Recorremos de atras hacia adelante.
        # TODO: Es bastante confuso el tema de los indices.
        for l in reversed(range(0, len(neural_net))):

            # Se usa l + 1 porque no tenemos realmente capa inicial
            # solo registramos las capas ocultas y la de salida
            z = out[l + 1][0]
            a = out[l + 1][1]

            # Calculo de los deltas de la ultima capa. Es directamente el error de la capa.
            if l == len(neural_net) - 1:
                # Formula en minuto 52 del video.
                # Insert 0 agrega al inicio. Porque estamos recorriendo de atras hacia adelante.
                deltas.insert(0, l2_cost[1](a, Y) * neural_net[l].act_f[1](a))
            # Calcular los deltas en funcion de los deltas de la capa previa.
            else:
                # Multiplicamos el error actual por la matriz de la capa anterior.
                # Tomamos la matriz de pesos transpuesta para que tenga sentido la operacion.
                deltas.insert(0, deltas[0] @ _W.T * neural_net[l].act_f[1](a))

            # Guardamos una copia del actual. Porque lo modificamos con el gradient descent.
            _W = neural_net[l].W

            # Gradient descent -> Actualizamos los pesos a medida que operamos.

            # Actualizamos el parametro independiente
            # TODO: Sacar el vector independiente y agregarlo a la matriz de pesos W.
            neural_net[l].b = neural_net[l].b - np.mean(deltas[0], axis=0, keepdims=True) * lr
            neural_net[l].W = neural_net[l].W - out[l][1].T @ deltas[0] * lr

    return out[-1][1]


# VISUALIZACIÓN Y TEST

import time

loss = []  # Costos (error).
loss_hidden = []

def error_real(salidas, salidas_esperadas):
    assert len(salidas) == len(salidas_esperadas)
    errores = 0.0
    for i in range(len(salidas)):
        if salidas[i] != salidas_esperadas[i]:
            errores += 1.0
    return errores / len(salidas)

sigm_binary = lambda x: 0 if x < 0.5 else 1

def binarizar_sigm(vector):
    binarizado = []
    for item in vector:
        binarizado.append(sigm_binary(item))
    return binarizado


for i in range(10000):

    # Entrenemos a la red!
    if i < 10:
        pY = train(neural_net, X, Y, l2_cost, lr=0.01)
    # elif i < 3900:
    #     pY = train(neural_net, X, Y, l2_cost, lr=0.001)
    # elif i < 7500:
    #     pY = train(neural_net, X, Y, l2_cost, lr=0.0001)
    else:
        pY = train(neural_net, X, Y, l2_cost, lr=0.001)

    # if i % 300 == 0:
    #     time.sleep(0.5)  # Evita CPU al 100%

    if i % 300 == 0:

        # print(pY)  # pY son las salidas de la ultima capa.
        # loss.append(l2_cost[0](pY, Y))
        salida_pY = binarizar_sigm(pY)

        loss.append(error_real(salida_pY, Y))

        error_set_oculto = train(neural_net, X_hidden, Y_hidden, l2_cost, train=False)
        salida_oculto = binarizar_sigm(error_set_oculto)

        # loss_hidden.append(l2_cost[0](salida, Y_hidden))
        loss_hidden.append(error_real(salida_oculto, Y_hidden))

        plt.show()
        plt.plot(range(i, i + len(loss[-10:])), loss[-10:])
        plt.plot(range(i, i + len(loss_hidden[-10:])), loss_hidden[-10:], linestyle="dashed")
        plt.show()
        time.sleep(0.5)


# Ya entreado. Aplicar
entradas = []
with open('X_test.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        line = [float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4])]
        entradas.append(line)

X_final = np.array(entradas)
Y_final = train(neural_net, X_final, None, l2_cost, train=False)
Y_final = binarizar_sigm(Y_final)

with open('Y_test.csv', 'w+') as csv_file:
    for i in Y_final:
        print(str(i), file=csv_file)
    csv_file.flush()
