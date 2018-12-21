#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 15:46:50 2018

@author: Akli
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# load dataframe from CSV file
data = pd.read_csv('data/tobacco-lab_data_Tobacco3482.csv')

#on verifie qu'il y a pas de valeur manquante
data.count()

#affichage des donnees 
print(data.head(5))

#preparation des donnees
nbr = data.shape[0]
for i in range (nbr):
    a = data.get_value(i, 'img_path')
    data.set_value(i, 'img_path', 'data/Tobacco3482-OCR/'+a)
    data.set_value(i, 'img_path', data.get_value(i, 'img_path').split('.jpg')[0]+'.txt')
    data.set_value(i, 'img_path',open(data.get_value(i, 'img_path'), "r").read())

data.columns = ['text','label']


# Plot the statistics of category
plt.figure(1,figsize = (7,7))
plt.title("plot of the class frequencies")
sns.countplot(data=data,y='label')

counts =data["label"].value_counts()
plt.figure(2,figsize = (8,8))
plt.title("Camembert plot of the class frequencies")
plt.pie(counts,labels = counts.index,shadow = True)
plt.show()

# Print examples of the articles
print(data.head())
print(data.iloc[15].text)

# Split the dataset, create X (features) and y (target), print the size
(X_train,X_test,y_train,y_test) = train_test_split(data['text'],data['label'],test_size=0.2)

vectorizer = CountVectorizer(max_features=2000)
vectorizer.fit(X_train)
X_train_counts = vectorizer.transform(X_train)
X_test_counts = vectorizer.transform(X_test)

# le model
clf = LogisticRegression(C = 1)
clf.fit(X_train_counts, y_train)
print(clf.score(X_train_counts, y_train))
print(clf.score(X_test_counts, y_test))
y_pred_counts = clf.predict(X_test_counts)

confusion_counts = confusion_matrix(y_test,y_pred_counts)
print('Matrice de confusion(CountVectorizer):\n',confusion_counts)
report_counts = classification_report(y_test,y_pred_counts)
print(report_counts)