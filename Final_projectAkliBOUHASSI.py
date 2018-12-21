#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 15:46:50 2018

@author: Akli
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# load dataframe from CSV file
data = pd.read_csv('data/tobacco-lab_data_Tobacco3482.csv')


#preparation des donnees
nbr = data.shape[0]
for i in range (nbr):
    a = data.get_value(i, 'img_path')
    data.set_value(i, 'img_path', 'data/Tobacco3482-OCR/'+a)
    data.set_value(i, 'img_path', data.get_value(i, 'img_path').split('.jpg')[0]+'.txt')
    data.set_value(i, 'img_path',open(data.get_value(i, 'img_path'), "r").read())

data.columns = ['text','label']



# separation des donnees en donnees d'apprentissage et donnees de test
(X_train,X_test,y_train,y_test) = train_test_split(data['text'],data['label'],test_size=0.2)


#transformation des document en vecteur
vectorizer = CountVectorizer(max_features=2000)
vectorizer.fit(X_train)
X_train_counts = vectorizer.transform(X_train)
X_test_counts = vectorizer.transform(X_test)


# apprentissage de model de regression logistique
clf = LogisticRegression(C = 1)
clf.fit(X_train_counts, y_train)
print(clf.score(X_train_counts, y_train))
print(clf.score(X_test_counts, y_test))
y_pred_counts = clf.predict(X_test_counts)


#Affichag des resultats en details
confusion_counts = confusion_matrix(y_test,y_pred_counts)
print('Matrice de confusion(CountVectorizer):\n',confusion_counts)
report_counts = classification_report(y_test,y_pred_counts)
print(report_counts)