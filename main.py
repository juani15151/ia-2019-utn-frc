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

    @staticmethod
    def proporcion_error(salidas, salidas_esperadas):
        assert len(salidas) == len(salidas_esperadas)

        errores = 0
        for i in range(len(salidas)):
            if salidas[i] != salidas_esperadas[i]:
                errores += 1

        return errores / len(salidas)


class Capa:

    # TODO: Poner el procesamiento hacia adelante y atrás acá.

    def __init__(self, cantidad_entradas, cantidad_neuronas, act_f):
        """

        :param cantidad_entradas: Numero de conexiones entrantes desde la capa anterior.
        :param cantidad_neuronas: Numero de neuronas.
        :param act_f: Conjunto de funcion de activacion. Contiene la funcion de activacion y su derivada.
        """
        self.act_f = act_f  # Introduce no linealidad.

        # Vector B para cada neurona. (Vector columna con la entrada unitaria)
        # Random normalizado a valores entre -1 y 1.
        self.b = np.random.rand(1, cantidad_neuronas) * 2 - 1

        # Matriz de pesos para cada neurona.
        # Random normalizado a valores entre -1 y 1.
        self.W = np.random.rand(cantidad_entradas, cantidad_neuronas) * 2 - 1

    def procesar_lote(self, entradas):
        # Pasamos las entradas por el vector de pesos.
        salida_neurona = entradas @ self.W + self.b

        # Pasamos el resultado por la funcion de activacion.
        salida_activacion = self.act_f[0](salida_neurona)

        return salida_neurona, salida_activacion


class RedNeuronal:

    def __init__(self, topologia):
        self.capas = []

        # Funcion de activacion.
        sigm = (
            lambda x: 1 / (1 + np.e ** (-x)),  # Funcion sigmoide
            lambda x: x * (1 - x)  # Derivada de la funcion de activacion
        )

        for i in range(len(topologia) - 1):
            self.capas.append(Capa(topologia[i], topologia[i + 1], sigm))

        # Error cuadratico medio.
        self.funcion_error = lambda Yp, Yr: np.mean((Yp - Yr) ** 2)
        # Derivada del error cuadratico medio
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

    def entrenar(self, X, Y, lr=0.1):
        # El vector de entradas es inicialmente los datos.
        # Despues son las salidas de la capa anterior.
        salida = [(None, X)]

        # Forward pass
        for l, layer in enumerate(self.capas):
            valores_brutos, valores_activacion = layer.procesar_lote(salida[-1][1])

            # Guardamos salidas de la neurona y la activacion para la proxima capa.
            salida.append((valores_brutos, valores_activacion))

        # Backward pass
        deltas = []  # Valores en funcion al error.

        # Recorremos de atras hacia adelante.
        # TODO: Es bastante confuso el tema de los indices.
        for l in reversed(range(0, len(self.capas))):

            # Se usa l + 1 porque no tenemos realmente capa inicial
            # solo registramos las capas ocultas y la de salida
            valores_brutos = salida[l + 1][0]
            valores_activacion = salida[l + 1][1]

            # Calculo de los deltas de la ultima capa. Es directamente el error de la capa.
            if l == len(self.capas) - 1:
                # Formula en minuto 52 del video.
                # Insert 0 agrega al inicio. Porque estamos recorriendo de atras hacia adelante.
                deltas.insert(0, self.funcion_error_derivada(valores_activacion, Y)
                              * self.capas[l].act_f[1](valores_activacion))
            # Calcular los deltas en funcion de los deltas de la capa previa.
            else:
                # Multiplicamos el error actual por la matriz de la capa anterior.
                # Tomamos la matriz de pesos transpuesta para que tenga sentido la operacion.
                deltas.insert(0, deltas[0] @ _W.T * self.capas[l].act_f[1](valores_activacion))

            # Guardamos una copia del actual. Porque lo vamos a modificar.
            _W = self.capas[l].W

            # Gradient descent -> Actualizamos los pesos a medida que operamos.

            # Actualizamos el parametro independiente
            self.capas[l].b = self.capas[l].b - np.mean(deltas[0], axis=0, keepdims=True) * lr
            self.capas[l].W = self.capas[l].W - salida[l][1].T @ deltas[0] * lr

        # Salida de la funcion de activacion de la ultima capa
        return Utils.binarizar_sigm(salida[-1][1])


def main():
    # Leer el set de datos y salidas esperadas.
    X_desconocido = IOUtils.leer_csv("X_train.csv")  # Matriz (2000 x 5)
    salida_esperada = IOUtils.leer_csv("Y_train.csv")  # Vector columna.

    # Los archivos deben tener la misma cantidad de registros.
    assert len(X_desconocido) == len(salida_esperada)

    # Separar parte del conjunto para medir el error.
    X_entrenamiento, X_test = Utils.dividir_conjunto(X_desconocido, 0.75)
    Y_entrenamiento, Y_test = Utils.dividir_conjunto(salida_esperada, 0.75)

    # Siempre empieza con la cantidad de caracteristicas,
    # y sale con 1 neurona porque clasifica de forma binaria.
    cantidad_caracteristicas = len(X_desconocido[0])
    topologia = [cantidad_caracteristicas, 7, 11, 19, 5, 1]  # Min. 11% / 11%

    ref_neuronal = RedNeuronal(topologia)

    E_entrenamiento = []  # Error.
    E_test = []

    for i in range(20000):

        # Entrenamiento
        if i < 100:
            learning_rate = 0.1
        elif i < 10000:
            learning_rate = 0.001
        else:
            learning_rate = 0.0001

        salida_entrenamiento = ref_neuronal.entrenar(X_entrenamiento, Y_entrenamiento, learning_rate)

        # Visualizacion
        if i % 2000 == 0:
            print(i)

            E_entrenamiento.append(Utils.proporcion_error(salida_entrenamiento, Y_entrenamiento))

            salida_test = ref_neuronal.procesar(X_test)
            E_test.append(Utils.proporcion_error(salida_test, Y_test))

            plt.show()
            # Mostrar completo
            plt.plot(range(len(E_entrenamiento)), E_entrenamiento)
            plt.plot(range(len(E_test)), E_test, linestyle="dashed")

            plt.show()
            time.sleep(0.5)  # Importante para evitar CPU al 100%.

    # Ya entreada la red. Aplicar sobre set desconocido
    X_desconocido = IOUtils.leer_csv("X_test.csv")
    Y_desconocido = ref_neuronal.procesar(X_desconocido)

    IOUtils.escribir_csv('Test.csv', Y_desconocido)


# Punto de entrada al programa.
if __name__ == "__main__":
    main()
