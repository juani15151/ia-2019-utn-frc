from io import open

#escritura
archivo_texto = open("archivo.txt", "w")
frase="Estupendo dia para estudiar Python \n el miércoles"
archivo_texto.write(frase)
archivo_texto.close()

#lectura
archivo_texto = open("archivo.txt", "r")
texto=archivo_texto.read()
archivo_texto.close()
print(texto)

#readlines
archivo_texto = open("archivo.txt", "r")
lineas_texto=archivo_texto.readlines()
archivo_texto.close()
print(lineas_texto)

#readlines
archivo_texto = open("archivo.txt", "r")
lineas_texto=archivo_texto.readlines()
archivo_texto.close()
print(lineas_texto)