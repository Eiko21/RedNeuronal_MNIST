import gzip
import _pickle as cPickle

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as mp


# Translate a list of labels into an array of 0's and one 1.
# i.e.: 4 -> [0,0,0,0,1,0,0,0,0,0]
def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f, encoding='latin-1')
f.close()

train_x, train_y = train_set #CONJUNTO DE ENTRENAMIENTO (para entrenar)
valid_x, valid_y = valid_set #CONJUNTO DE VALIDACION (para mostrar el error (loss))
test_x, test_y = test_set #CONJUNTO DE TESTS (para determinar la precision de la red)


# ---------------- Visualizing some element of the MNIST dataset --------------

import matplotlib.cm as cm
import matplotlib.pyplot as plt

#plt.imshow(train_x[57].reshape((28, 28)), cmap=cm.Greys_r)
#plt.show()  # Let's see a sample
#print (train_y[57])

#CONSTRUIMOS UNA NUEVA MATRIZ A PARTIR DEL CONJUNTO DE ENTRENAMIENTO TRAIN_X (filas a coger)
x_data = train_x
y_data = one_hot(train_y, 10) #CONTRUIMOS EL NUEVO MODELO CON LAS ULTIMAS FILAS

y_validData = one_hot(valid_y, 10) #CONSTRUIMOS EL NUEVO CONJUNTO DE VALIDACION

x = tf.placeholder("float", [None, 784])  # samples #NUMERO DE MUESTRAS
y_ = tf.placeholder("float", [None, 10])  # labels #NUMERO DE ETIQUETAS

#PRIMERA CAPA
W1 = tf.Variable(np.float32(np.random.rand(784, 25)) * 0.1) #MATRIZ DE 784x25
b1 = tf.Variable(np.float32(np.random.rand(25)) * 0.1)  #VECTOR DE 25 ELEMENTOS
#SEGUNDA CAPA
W2 = tf.Variable(np.float32(np.random.rand(25, 10)) * 0.1)  #MATRIZ DE 25x10
b2 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)  #VECTOR DE 10 ELEMENTOS

h = tf.nn.sigmoid(tf.matmul(x, W1) + b1) #MULTIPLICACION DE LA MATRIZ CON EL VECTOR (1ra capa)
y = tf.nn.sigmoid(tf.matmul(h, W2) + b2) #MULTIPLICACION DE LA MATRIZ CON EL VECTOR (2da capa)

loss = tf.reduce_sum(tf.square(y_ - y))

#train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)  # learning rate: 0.01
#train = tf.train.GradientDescentOptimizer(0.02).minimize(loss)
train = tf.train.GradientDescentOptimizer(0.03).minimize(loss)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

print ("----------------------")
print ("   Start training...  ")
print ("----------------------")

batch_size = 10 #Tamaño del lote de muestras
errors = [] #Donde guardaremos los valores de los errores
epochs = [] #Donde guardaremos todas las epocas (sera el indice)
list_current_errors = [] #Lista de los errores actuales que se calculan
umbral = 0.001 #Umbral para la condicion de parada
tolerancia = 10  #Tolerancia total
tolerancia_count = 0 #Contador comparador de la tolerancia para la parada de la red

for epoch in range(100):
    for jj in range(int(len(train_x) / batch_size)):
        batch_xs = x_data[jj * batch_size: jj * batch_size + batch_size]
        batch_ys = y_data[jj * batch_size: jj * batch_size + batch_size]
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})  #CALCULAMOS LA SALIDA A PARTIR DE LOS LOTES ANTERIORES

    error = sess.run(loss, feed_dict={x: batch_xs, y_: batch_ys}) / batch_size #SE CALCULAN LAS PERDIDAS (DE CADA ITERACION)
    errors.append(loss) #Añadimos el nuevo valor de perdida

    current_error = sess.run(loss, feed_dict={x: valid_x, y_: y_validData}) / len(y_validData) #SE CALCULAN LOS ERRORES ACTUALES
    list_current_errors.append(current_error) #Añadimos el nuevo error actual calculado

    epochs.append(epoch) #Añadimos la nueva epoca
    print("Epoch #:", epoch, "Error: ", error)  #SE IMPRIME EL Nº DE EPOCA CON SU ERROR CORRESPONDIENTE
    print("Epoch #:", epoch, "Current Error: ", current_error) #SE IMPRIME EL Nº DE EPOCA CON SU ERROR CALCULADO

    #SE MUESTRAN LAS SALIDAS (result)
    result = sess.run(y, feed_dict={x: batch_xs})
    for b, r in zip(batch_ys, result):
        print (b, "-->", r)
    print ("----------------------------------------------------------------------------------")

    #EVALUAMOS LA CONDICIÓN DE PARADA DE LA RED
    #Calculamos el valor absoluto de la diferencia entre el error actual con el anterior y la comparamos con el umbral
    percentage = abs(list_current_errors[epoch] - list_current_errors[epoch - 1])
    if percentage < umbral:
        tolerancia_count += 1  #Si la diferencia no supera el umbral, la tolerancia se incrementa
    else:   #Si la supera, no aumenta
        tolerancia_count = 0

    #Si se supera la tolerancia minima, es que la red no puede aprender mas (se ha estabilizado), por lo que
    #se finaliza el entrenamiento
    if tolerancia_count > tolerancia:
        print("FIN DEL ENTRENAMIENTO")
        break

    print("--------------------------------------------")
    print("  GRAFICA DE EVOLUCION DEL ENTRENAMIENTO  ")
    print("--------------------------------------------")
    mp.title("Evolucion Errores")   #Establecemos el titulo de la grafica
    mp.plot(epochs, list_current_errors, label = 'Error Actual')    #Etiqueta para cada error actual
    #mp.plot(epochs, errors, label = 'Entrenamiento de la red')  #Etiqueta para cada error durante el entrenamiento
    mp.xlabel('Epocas') #Indicamos que el eje X serán las epocas
    mp.ylabel('Error')  #Indicamos que el eje Y serán los errores
    mp.legend()
    mp.show()

    #CONCLUSION: con un optimizador de 0.02, la red neuronal aprende continuamente hasta llegar a la epoca 23.
    # Una vez llega ahi no puede aprender mas y finaliza el entrenamiento, con un error = 0.0178 y un error actual = 0.1075
    #Con un optimizador de 0.03, la red aprende hasta llegar a la epoca 42, donde se queda con un error = 0.064984
    #y un error actual = 0.094
