import csv
import time

import numpy as np
import matplotlib.pyplot as plt


class IOUtils:

    @staticmethod
    def leer_csv(nombre_archivo):
        """
        Lee un archivo csv y parsea su contenido a float.
        :return: Matriz (vector de vectores) con los valores.
        """
        with open(nombre_archivo) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')

            filas = [valores for valores in csv_reader]

            # Retornamos el vector de vectores de entrada.
            # Parseando los datos a float.
            return np.array(filas).astype(float)

    @staticmethod
    def escribir_csv(nombre_archivo, lineas):
        """
        Sobrescribe el contenido del archivo con el contenido dado.
        :param nombre_archivo: Donde escribir. Si no existe lo crea.
        :param lineas: Vector con el contenido a escribir.
        :return: None
        """
        with open(nombre_archivo, 'w+') as csv_file:
            for linea in lineas:
                print(str(linea), file=csv_file)
            csv_file.flush()  # Obliga a escribir el contenido del buffer de salida.


class Utils:

    @staticmethod
    def dividir_conjunto(vector, proporcion):
        """
        Divide un vector segun un porcentaje dado.
        :param vector:
        :param proporcion: valor entre 0 y  1 usado para el corte.
        :return:
        """
        assert 0 < proporcion < 1
        corte = int(len(vector) * proporcion)

        return vector[corte:], vector[:corte]

    @staticmethod
    def binarizar_sigm(vector):
        """
        Toma un vector con valores de la funcion sigmoide y los binariza
        :param vector:
        :return:
        """
        binarizado = []
        for item in vector:
            binarizado.append(Utils.binarizar_valor(item, 0.5))
        return binarizado

    @staticmethod
    def binarizar_valor(x, corte):
        return 0 if x < corte else 1


# MAIN

# Leer el set de datos y salidas esperadas.
entradas = IOUtils.leer_csv("X_train.csv")  # Matriz (2000 x 5)
salida_esperada = IOUtils.leer_csv("Y_train.csv")  # Vector columna.

assert len(entradas) == len(salida_esperada)  # Los archivos deben tener la misma cantidad de registros.

# Separar parte del conjunto para medir el error. Necesario para detectar sobre-entrenamiento.
X_entrenamiento, X_test = Utils.dividir_conjunto(entradas, 0.75)
Y_entrenamiento, Y_test = Utils.dividir_conjunto(salida_esperada, 0.75)


class Capa:

    # TODO: Poner el procesamiento hacia adelante y atrás acá.

    def __init__(self, cantidad_entradas, cantidad_neuronas, act_f):
        """

        :param cantidad_entradas: Numero de conexiones entrantes desde la capa anterior.
        :param cantidad_neuronas: Numero de neuronas.
        :param act_f: Conjunto de funcion de activacion. Contiene la funcion de activacion y su derivada.
        """
        self.act_f = act_f

        # Vector B para cada neurona. (Vector columna con la entrada unitaria)
        self.b = np.random.rand(1, cantidad_neuronas) * 2 - 1  # Random normalizado a valores entre -1 y 1.

        # Matriz de pesos para cada neurona.
        self.W = np.random.rand(cantidad_entradas, cantidad_neuronas) * 2 - 1  # Random normalizado a valores entre -1 y 1.

    def procesar_lote(self, entradas):
        # Pasamos las entradas por el vector de pesos.
        salida_neurona = entradas @ self.W + self.b

        # Pasamos el resultado por la funcion de activacion.
        salida_activacion = self.act_f[0](salida_neurona)

        return salida_neurona, salida_activacion

# FUNCIONES DE ACTIVACION

# Se utilizan para introducir no linealidad
# y poder combinar neuronas sin que se reduzcan a uno única.

sigm = (
    lambda x: 1 / (1 + np.e ** (-x)),  # Funcion sigmoide
    lambda x: x * (1 - x)  # Derivada de la funcion de activacion
)


class RedNeuronal:

    def __init__(self, topologia):
        self.capas = []

        for i in range(len(topologia) - 1):
            self.capas.append(Capa(topologia[i], topologia[i + 1], sigm))

        # TODO: Creo que no hace falta la funcion del error aca.
        self.funcion_error = lambda Yp, Yr: np.mean((Yp - Yr) ** 2)  # Error cuadratico medio.
        self.funcion_error_derivada = lambda Yp, Yr: (Yp - Yr)

    def procesar(self, X):
        """
        Pasa la entrada a traves de cada capa de la red.
        :param X:
        :return:
        """
        for capa in self.capas:
            _, salida = capa.procesar_lote(X)
            X = salida

        return Utils.binarizar_sigm(salida)

    def entrenar(self, X, Y, lr=0.5, train=True):
        # El vector de entradas es inicialmente los datos.
        # Despues son las salidas de la capa anterior.
        out = [(None, X)]

        # Forward pass
        for l, layer in enumerate(self.capas):
            z, a = layer.procesar_lote(out[-1][1])

            # Guardamos salidas de la neurona y la activacion para la proxima capa.
            out.append((z, a))

        if train:

            # Backward pass
            deltas = []  # Valores en funcion al error.

            # Recorremos de atras hacia adelante.
            # TODO: Es bastante confuso el tema de los indices.
            for l in reversed(range(0, len(self.capas))):

                # Se usa l + 1 porque no tenemos realmente capa inicial
                # solo registramos las capas ocultas y la de salida
                z = out[l + 1][0]
                a = out[l + 1][1]

                # Calculo de los deltas de la ultima capa. Es directamente el error de la capa.
                if l == len(self.capas) - 1:
                    # Formula en minuto 52 del video.
                    # Insert 0 agrega al inicio. Porque estamos recorriendo de atras hacia adelante.
                    deltas.insert(0, self.funcion_error_derivada(a, Y) * self.capas[l].act_f[1](a))
                # Calcular los deltas en funcion de los deltas de la capa previa.
                else:
                    # Multiplicamos el error actual por la matriz de la capa anterior.
                    # Tomamos la matriz de pesos transpuesta para que tenga sentido la operacion.
                    deltas.insert(0, deltas[0] @ _W.T * self.capas[l].act_f[1](a))

                # Guardamos una copia del actual. Porque lo modificamos con el gradient descent.
                _W = self.capas[l].W

                # Gradient descent -> Actualizamos los pesos a medida que operamos.

                # Actualizamos el parametro independiente
                self.capas[l].b = self.capas[l].b - np.mean(deltas[0], axis=0, keepdims=True) * lr
                self.capas[l].W = self.capas[l].W - out[l][1].T @ deltas[0] * lr

        return Utils.binarizar_sigm(out[-1][1])


# FUNCION DE ENTRENAMIENTO

# Siempre empieza con la cantidad de caracteristicas,
# y sale con 1 neurona porque clasifica de forma binaria.
cantidad_caracteristicas = len(entradas[0])
topology = [cantidad_caracteristicas, 10, 20, 1]

neural_net = RedNeuronal(topology)

# VISUALIZACIÓN Y TEST


loss = []  # Costos (error).
loss_hidden = []

def error_real(salidas, salidas_esperadas):
    assert len(salidas) == len(salidas_esperadas)
    errores = 0.0
    for i in range(len(salidas)):
        if salidas[i] != salidas_esperadas[i]:
            errores += 1.0
    return errores / len(salidas)


for i in range(10000):

    # Entrenemos a la red!
    if i < 10:
        pY = neural_net.entrenar(X_entrenamiento, Y_entrenamiento, lr=0.01)
    else:
        pY = neural_net.entrenar(X_entrenamiento, Y_entrenamiento, lr=0.001)

    if i % 300 == 0:

        # print(pY)  # pY son las salidas de la ultima capa.
        # loss.append(l2_cost[0](pY, Y))
        salida_pY = pY

        loss.append(error_real(salida_pY, Y_entrenamiento))

        error_set_oculto = neural_net.entrenar(X_test, Y_test, train=False)

        # loss_hidden.append(l2_cost[0](salida, Y_hidden))
        loss_hidden.append(error_real(error_set_oculto, Y_test))

        plt.show()
        plt.plot(range(i, i + len(loss[-10:])), loss[-10:])
        plt.plot(range(i, i + len(loss_hidden[-10:])), loss_hidden[-10:], linestyle="dashed")
        plt.show()
        time.sleep(0.5)  # Importante para evitar CPU al 100%.


# Ya entreado. Aplicar
entradas = IOUtils.leer_csv("X_test.csv")

Y_final = neural_net.entrenar(entradas, None, train=False)

IOUtils.escribir_csv('Test.csv', Y_final)
