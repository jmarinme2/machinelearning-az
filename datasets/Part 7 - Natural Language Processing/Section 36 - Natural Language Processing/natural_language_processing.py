#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 17:13:51 2019

@author: juangabriel
"""

# Natural Language Processing

# Importar librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el dataset
dataset = pd.read_csv("C:/Users/Admin/Documents/GitHub/machinelearning-az/datasets/Part 7 - Natural Language Processing/Section 36 - Natural Language Processing/Restaurant_Reviews.tsv", delimiter = "\t", quoting = 3)

# Limpieza de texto
import re
import nltk
nltk.download('stopwords') #descargamos la lista de palabras de storwords que son las palabras irrelevantes 
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer #Libreria para poder hacer el stemming
corpus = []
for i in range(0, 1000):
    #La expresion que viene ahora indica substituir cualquier cosa que no sea una letra por un espacio en blanco.
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i]) #De esta forma eliminamos todas las cosas que no sean letras. El ^al principio indica las cosas que noquiero eliminar
    review = review.lower() #Trasnformo todas las letras a minusculas
    review = review.split() #Fracciono las frases por los espacios en blanco y me quedo con una lista de palabras o caracteres
    ps = PorterStemmer() #creo el objeto para hacer la stemizacion
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))] #Eliminamos las conjugaciones o declinaciones de las palabras que hayan sobrevivido a mi bulce de eliminacion de palabras no utiles o stopwords
    review = ' '.join(review) #Vuelvo a unir las palabras que están en la lista
    corpus.append(review)
 
# Crear el Bag of Words
from sklearn.feature_extraction.text import CountVectorizer #Transformará las palabras en vectores de frecuencias
cv = CountVectorizer(max_features = 1500) #COn el max_features, es un parámetro que nos permite reducir el número máximo de columnas de la matriz y así nos quedamos con las más frecuentes. Esto reduce la dimensionalidad del problema.
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values


# Dividir el data set en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# Ajustar el clasificador en el Conjunto de Entrenamiento
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicción de los resultados con el Conjunto de Testing
y_pred  = classifier.predict(X_test)

# Elaborar una matriz de confusión
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

(55+91)/200

#Con árboles de decision
from sklearn import tree 
classifier = tree.DecisionTreeClassifier()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

