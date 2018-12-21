# Classification de document
Réalisé par: BOUHASSI Akli<br>
Université de Rouen / Master 2 SD-2018/2019

# Introduction:
Dans ce projet, on dispose d'un dataset 'Tobacco' qui est un ensemble de 3482 documents texts et photos repartie sur 10 classes. On va étudier que la partie text de ce dataset.

# Analyse des données:
1. On charge les données et on vérifie qu'il y a pas de valeur manquante
2. On affiche nos données pour avoir une première idée. 

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>img_path</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Advertisement/0000136188.jpg</td>
      <td>Advertisement</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Advertisement/0000435350.jpg</td>
      <td>Advertisement</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Advertisement/0000556056.jpg</td>
      <td>Advertisement</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Advertisement/0030048095.jpg</td>
      <td>Advertisement</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Advertisement/0030048989.jpg</td>
      <td>Advertisement</td>
    </tr>
  </tbody>
</table>
</div>


3. on cherche a prédire les classes a partir des fichiers text. On remarque qu'on a les liens vers les fichiers photos et pas text. Par contre ils ont le même nom sauf l'extension donc il suffit de changer '.jpg' par '.txt' pour pouvoir lire les fichiers text. Ensuite, on remplace le lien vers les text par le text qui correspond.

4. On change le nom des colomns en text et label

5. On affiche les répartitions des classes
![png](stat.png)stat.png

On remarque que les données ne sont pas vraiment déséquilibrés malgres que les classes memo email letter et form représentent plus de la moitié mais on peut dire qu'on aura meilleur classification pour les classes les plus représentées

# Préparation des données 
Pour appliquer des algorithmes d'apprentissage automatique aux textes, les documents doivent être transformés en vecteurs. Le moyen le plus simple de transformer un document en vecteur est Le Bag of Word BoG.

Dans un premier temps on utilise CountVectorizer dont le principe est de créer un dictionnaire de mot contenant dans la base d'apprentissage puis compter le nombre de fois qu'un mot apparaisse dans un document pour créer le vecteur qui représente chaque document. Puis on garde que les 2000 mot qui apparaisse le plus (max_features=2000) pour cet exemple

On applique l'algorithme de Naïve Bayes dans un premier temps.
on obtient les résultats suivant:


Une autre méthode pour vectorizer nos données est TFIDF qui prend en compte le nombre de documents dans lesquels un mot donné apparaît. Un mot qui apparaît dans de nombreux documents aura moins de poids.
Avec TFIDF on obtient une précision plus petite donc CountVectorizer est plus adapté a ce problème.

Maintenant On cherche a déterminer les meilleurs paramètres avec gridsearcv on obtient alpha = 1

# Deep Learning
J'ai appliqué réseau récurrent (LSTM) plus la convolution sur les données avec 15 epochs. j'ai obtenu les résultats suivants:



# Conclusion
L'algorithme qui a marche le mieux sur ces données est la régression logistique avec une précision de 77% et un recall de 77%
 
par contre on aurait pu avoir de meilleur résultat avec un CNN si on avait plus de données 
et si on avait appliquer aussi sur les images.
On sait bien que les réseaux de neurones donnent de meilleur résultats sur les données images et texte.
