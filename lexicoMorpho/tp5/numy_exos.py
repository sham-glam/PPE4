import sys
import re
import math
import nltk
import numpy as np
from nltk.collocations import *
from nltk.tokenize import word_tokenize
import os
import glob
import numpy as np


'''
1. Cr´eez un vecteur a avec les enters de 0 `a 9.
2. Cr´eez une matrice 3×3 avec la valeur True partout.
3. R´ecup´erez tous les nombres impairs du vecteur a.
4. Remplacez tous les nombres impairs dans a par -1.
5. Transformez une matrice de dimensions variables en un vecteur de valeurs,
avec la fonction reshape et sans la fonction flatten
6. Transformez un vecteur de n dimensions en une matrice 2xn/2.
7. Si a et b sont des vecteurs de mˆeme nombre de dimensions, donnez les ´el´ements qu’ils ont en commun, et leurs indices.
8. R´ecup´erez tous les ´el´ements de a entre 5 et 10.
9. Echangez les deux premi`eres colonnes dans une matrice.

BACK TO CORPUS:

10. Dans la matrice de co-occurrences du corpus reuters, supprimez les colonnes
qui correspondent aux mots avec 0 occurrences dans le corpus.

11. Donnez la liste des 50 mots avec le tf-idf le plus ´elev´e dans le corpus
reuters. Affichez le score tf-idf aussi.

12. Repr´esentez le vocabulaire du corpus acl.txt par une matrice terme-contexte
: les lignes et les colonnes correspondent aux mˆemes mots (wordlist.txt),
et les cases aux valeurs de co-occurrences dans une fenˆetre de 5 mots.

13. Compl´etez le programme de mani`ere `a demander deux mots `a l’utilisateur,
et retourner la similarit´e cosinus entre les deux mots.

'''



def main():
  # exo1
  arr1 = np.array(range(0, 10))
  print(arr1)
  
  # exo2
  arr2 = np.ones((3, 3)) >0
  print(arr2)
  
  # exo3
  pair_array = arr1[arr1 % 2 == 1]
  print(pair_array)
  
  print(arr1 %2 ==1) # gives True or False
  
  # exo4 
  arr1[arr1 % 2 == 1] = -1
  print(arr1)
  
  # exo5  - flatten with -1 // without .flatten
  arr3= np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
  reshaped_arr = arr3.reshape(-1)
  print(reshaped_arr)
  
  ### solution
  # -> x.reshape(1,x.shape[0]*x.shape[1]) ## multiply dims to make 1 single 2x3 dims = 6
  
   
  # exo6  - Transformez un vecteur de n dimensions en une matrice 2xn. ?? 
  arr4= np.ones((4))
  print(arr4)
  
  arr4 = arr4.reshape(2, -1) # 2 rows and n columns
  print(arr4)
  
  ### solution 
  x=np.arange(10)
  x.reshape(2, int(x.shape[0]/2)) #shape with dimos of num of rows x.shape[0]

  # exo7 - Si a et b sont des vecteurs de mˆeme nombre de dimensions, donnez les ´el´ements qu’ils ont en commun, 
  #   et leurs indices.
  a = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
  
  b = np.array([[1, 3, 5, 7, 9], [2, 4, 6, 8, 10]])
  
  # print(np.where(a==b)) # difference
  # print(np.argwhere(a==b)) # gives the indices
  
  ### solution 
  a[a==b]
  np.where(a==b)
  
  # exo8 -R´ecup´erez tous les ´el´ements de a entre 5 et 10.
  a = np.array([[1, 2, 3, 4, 5], 
                  [6, 7, 8, 9, 10], 
                  [11, 12, 13, 14, 15], 
                  [16, 17, 18, 19, 20]])
  
  print(a[a>=5])
  print(np.argwhere(a>=5))
  print('Smaller than 5 and bigger than 10:')
  result = np.argwhere(a[(a >= 5) & (a <= 10)])
  print(result)

  
  # exo9 - Echangez les deux premi`eres colonnes dans une matrice.
  # print(a.shape)
  
  print(f'\nbefore exchange\n{a}')
  # a[:,:1]= a[:,:0]
  a[:,[0, 1]] = a[:,[1, 0]] # exchange the first two columns 0 becomes 1 and 1 becomes 0
  print("\n\nexchanged\n" , a)
  
  new_array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
  print(new_array.shape)
  
if __name__ == '__main__':
  main()
