import sys
import re
import math
import nltk
from nltk.collocations import *
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
 


#nltk.download('punkt')

# Ecrivez la fonction 'main' qui ouvre le fichier donné en argument en ligne de commande 
# (sys.argv[1]) et calcule les fréquences suivantes :
# 1) la fréquence de chaque mot type dans le corpus : wordfreq
# 2) la fréquence de chaque bigram type dans le corpus : bifreq
# 3) la fréquence totale des mots : N
# 4) la fréquence totale de bigrams : B
# On considère que les séparateurs de mot sont l'espace et la fin de ligne. 

def normalize(word):
  for c in word:
    if c in string.punctuation:
      word = word.replace(c, " "+c + " ")
  # print(word)
  return word
  
      


  
 ### your code here ###
def get_occurences(file):
  wordfreq = {}
  with open ('acl/acl.txt', 'r') as f:
      content = f.read()
      print(len(content))
      # content = normalize(content)      
      content = content.split()
      print(len(content))
      print(content[:10])
      for word in content:
        word  = normalize(word)
        if word not in wordfreq.keys():
          wordfreq[word] = 1
        else:
          wordfreq[word] +=1 
          
      print(len(wordfreq))      
      print(list(sorted(wordfreq.items(), key=lambda item: item[1],  reverse=True))[:10])
      
      return wordfreq
     
      
          
      # bigram 
      for i in range (0, len(content)):
        if i < len(content)-1:
          bi = (content[i], content[i+1])
          bifreq.append(bi)
          
      
      # print(bifreq[:20])


    # exo 3 
      # 100 mots les plus fréquent
      most_frequent = (list(sorted(wordfreq.items(), key=lambda item: item[1],  reverse=True))[:100])
      # print(most_frequent)
      
      nltk.download('stopwords')
      stops = stopwords.words('english')
      # print(stops)
      acl_stops = []
      acl_stop_tup =[]
      for word in most_frequent:
        if word[0].lower() in stops:
          acl_stops.append(word[0])
          acl_stop_tup.append(word)

      # print(acl_stops)
      # print(acl_stop_tup)
      
      
      # exo 4 - redo exo1 - without stops
      word_freq = {}
      bi_freq = []
      clean = []
      for word in content:
        if word.lower() not in acl_stops and word not in string.punctuation:
          clean.append(word)
          if word not in word_freq.keys():
            word_freq[word] = 1
          else:
            word_freq[word] +=1
        
      print(len(word_freq))
      # print(list(sorted(word_freq.items(), key=lambda item: item[1],  reverse=True))[:70])


      for i in range (0, len(clean)):
        if i < len(clean)-1:
          bi = (clean[i], clean[i+1])
          bi_freq.append(bi)
      
      print(len(bi_freq))
      # print(bi_freq[:20])
          
          
def main():
  wordfreq = {} # dictionnaire contenant les fréquences des mots
  bifreq = []
  N =  get_occurences('acl/acl.txt') # fréquence totale des mots
  # print(N)
  B = 0


if __name__ == '__main__':
  main()

