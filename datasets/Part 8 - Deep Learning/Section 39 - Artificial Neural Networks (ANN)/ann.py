#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 10:38:56 2019

@author: juangabriel
"""

# Redes Neuronales Artificales

# Instalar Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Instalar Tensorflow y Keras
# conda install -c conda-forge keras

# Parte 1 - Pre procesado de datos


# Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el data set
dataset = pd.read_csv('C:/Users/Admin/Documents/GitHub/machinelearning-az/datasets/Part 8 - Deep Learning/Section 39 - Artificial Neural Networks (ANN)/Churn_Modelling.csv')

X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Codificar datos categóricos
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Dividir el data set en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Escalado de variables
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# Parte 2 - Construir la RNA

# Importar Keras y librerías adicionales
import keras
from keras.models import Sequential
from keras.layers import Dense

# Inicializar la RNA
classifier = Sequential()

# Añadir las capas de entrada y primera capa oculta
'''units es el número de nodos que habrá en la siguiente capa (en este caso la oculta)
el units, lo calculo (regla no escrita), calculando la media entre los nodos de entrada y de salida
mientras que imput_dim es el número de nodos que habrá en la capa actual, en este 
caso la capa de entrada. Es decir que tendremos 11 datos de entrada que serán 
transformados a 6 en la siguiente capa.
El kernel_initializer es como se eligen los pesos iniciales, el uniform es una iniciación aleatoria
activation es la función de activación de la capa oculta; relu es el rectificado lineal unitario'''

classifier.add(Dense(units = 6, kernel_initializer = "uniform",  
                     activation = "relu", input_dim = 11)) 


'''Al añadir una segunda capa, no hace falta que le indique el input_dim, porque la
segunda capa oculta ya sabe que se tiene que enganchar a la primera capa que ya sabe
que es de 6'''

# Añadir la segunda capa oculta
classifier.add(Dense(units = 6, kernel_initializer = "uniform",  activation = "relu"))


'''La capa de salida, le tenemos que indicar el número de nodos de salida en units. Como la salida es 
binaria, me basta con un solo nodo de salida.
Le indico la función de activación, que en este caso le pongo que es sigmoide ya que quiero
que sea algo similar a la probabilidad ya que va a tomar valores entre 0 y 1. Si se quiere
clasificar en tres categorías, tendríamos que cambiar en primer lugar el número de nodos de la 
capa de salida (un nodo por clase), y por otro lado, que la función sigmoide aplicada a 3 
categorias no sería la más adecuada, realmente habría que reemplazarla por un escalon o 
por un relu, para ver cual debería activarse.
El kernel_initializer lo puedo dejar en uniform '''

# Añadir la capa de salida
classifier.add(Dense(units = 1, kernel_initializer = "uniform",  activation = "sigmoid"))


'''El optimizador es el algoritmo que se emplea para encontrar el conjunto optimo 
de pesos dentro de la red neuronal, ya que las hemos establecido en aleatorias. Elegiremos 
uno que nos permita llegar al optimo. Por defecto aparece el adam y es el que utilizaremos.
loss, es la función de perdidas, y se corresponde a aquella función que debe ser minimizada,
es la que nos permitia comparar entre la predicción y el dato real.
metrics es una lista de todas las medidas que quiero que me devuelva la red neural para evaluarla 
'''

# Compilar la RNA
classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])



'''El batch_size es para indicarle al algoritmo si tienen que ajustar los pesos observación a observación
o si tiene que ajustarse despues de un conjunto de observaciones (batch). Este parámetro es 
para reducir la posibilidad de caer en un mínimo local.
Epochs son el número de iteraciones globales que tienen que pasar sobre todo el dataset. Por lo
general cuantas más pasadas más aprenderá, pero si nos pasamos en ellas significara muchas 
posibilidades de sufrir overfitting.
Se puede jugar con estos parámetros para ver que tal se comporta.
'''
# Ajustamos la RNA al Conjunto de Entrenamiento
classifier.fit(X_train, y_train,  batch_size = 10, epochs = 100)


# Parte 3 - Evaluar el modelo y calcular predicciones finales

# Predicción de los resultados con el Conjunto de Testing
y_pred  = classifier.predict(X_test) #Los resultados nos los da en valores continuos entre 0 y 1 = una probabilidad
y_pred = (y_pred>0.5) #Para traducir la probabilidad en una categoría
# Elaborar una matriz de confusión
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
