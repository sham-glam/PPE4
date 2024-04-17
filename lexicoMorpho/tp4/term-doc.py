import sys
import re
import math
import nltk
import numpy as np
from nltk.collocations import *
from nltk.tokenize import word_tokenize
import os
import glob

# nltk.download('reuters')
# nltk.download('punkt')


'''

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


def max_occurence(matrix, doc):
  occ = matrix[doc]
  file = occ+"txt"
  with open(file, 'r') as file:
    words = file.read().split()
    
  
  arg_max = np.argmax(matrix, axis=1)
  
  return arg_max

def create_corpus(word, matrix):
    txt_files = glob.glob("sample.corpus/*.txt")
    num_files = 0
    for file in txt_files:
      with open(file, 'r') as file:
        content = file.read()
        content = word_tokenize(content)
        for term in content:
            if term in word:
                matrix[num_files, word.index(term)] += 1
      num_files += 1
        
    
    return matrix
    

## main ##
def main():
  docs = []
  matrix = np.zeros((400, 5000))
  print(matrix.shape)

  #  fichier avec les termes
  f = open("terms.lst", "r")  
   
  word = f.read().split() # stockage des termes 
  ## imp : la position dans cette liste correspond à la colonne de la matrice qui représente le terme donné
  print(word[20])
  # lire le contenu du fichier, utiliser word_tokenize (nltk) pour obtenir les mots
  matrix = create_corpus(word, matrix )
  print(matrix)
  print(len(matrix[0]))
  
  # Exercice 2
  #  le nombre maximum d’occurrences d’un mot dans un document observé dans le corpus
  # print(matrix[0]) # affiche les occurences du document 1
  
  max_value = np.amax(matrix) # 36
  print(max_value)

  #le nombre maximum d’occurrences d’un mot pr´ecis dans un document du corpus - eg. sales
  col = word.index('sales')
  print(f'col = {col}') # 68 find for each row and col-th item - the max value
  max_occ = np.amax(matrix[:,col]) # 8
  # matrix[:, column_number] -> parcourt tout le matrice  et prend la valeur max pour la colonne mentionn;e
  print(f'max_occ_of_particular_word = {max_occ}') # 
  
  
  ## ex 2.3 - pour un mot pr´ecis, le document dans lequel il a le plus d’occurrences
  doc_max = np.argmax(matrix[:,col]) # 54
  print(f'doc_max_occurence_of_term = {doc_max}')
  
  
  ## 2.4 - pour un mot pr´ecis, le nombre de documents dans lesquels il apparaˆıt
  # sum all rows where maxtrix[,col] > 0
  doc_count = np.count_nonzero(matrix[:,col]) # 
  print(f'doc_count_with_term = {doc_count}')
  
  list_presence = np.argwhere(matrix[:,col] > 0)
  # print(f'list_presence = {list_presence}') 
  print(len(list_presence)) # 35
  print("type: ", type(list_presence)) # <class 'numpy.ndarray'>
  
  
  # np.argwhere(a==np.max(a)) # find the index of the max value in the matrix
  
  
  # 3 -le nombre d’occurrences est remplac´e par 1 si le mot apparaˆıt dans le document, et par 0 s’il n’apparaˆıt pas.
  matrix_dup = np.where(matrix > 0, 1, 0) # np.where(condition, x, y) -> if condition is true, x, else y
  # alternative
  (matrix > 0).astype(int) # astype 
  #np.sum(matrix[:,2]) # sum all values in the column 2

  
  print(matrix_dup)
  # np.argwhere(a[:,2]==np.amax(a[:,2])) # find the index of the max value in the matrix
  
  
  # 4 -  le nombre d’occurrences est remplacé par le poids tf-idf du terme dans le document.
  # tf-idf = tf * idf -> tf = term frequency, idf = inverse document frequency
  
  ## external link : https://www.learndatasci.com/glossary/tf-idf-term-frequency-inverse-document-frequency/
  
  tf= matrix[:,col] / np.sum(matrix[:,col])
  idf  = np.log(400/doc_count) # 400 = total number of documents
  
  tf_idf = tf * idf
  # print(f'tf_idf = {tf_idf}')  # array
  
  
  # exo 10 :
  #   10. Dans la matrice de co-occurrences du corpus reuters, supprimez les colonnes 
  # qui correspondent aux mots avec 0 occurrences dans le corpus.

  # matrix = np.delete(matrix, np.where(np.sum(matrix, axis=0) == 0), axis=1)
  m = np.delete(matrix, np.where(np.sum(matrix, axis=0) == 0), axis=1)
  empty_columns = np.where(np.sum(matrix, axis=0) == 0) # pas vide == 0 pour vide 
  print(empty_columns, len(empty_columns))  
  
  ## In 2D array, axis=0 operates column-wise ||  axis=1 operates row-wise || axis=-1 represents the last axis
  print(m.shape) # 400 x 3661 or 3685
   
  # pour tf_idf 
  '''
  np.sum(matrix, axis=0) >0 # 0 for empty columns
  
  
  ''' 
  
  
  ## exo 11 - Donnez la liste des 50 mots avec le tf-idf le plus ´elev´e dans le corpus reuters. Affichez le score tf-idf aussi.
  print(f'tf_idf = >50 {tf_idf>50}')
  
if __name__ == '__main__':
  main()

