import math
import string
import regex 
import string
import nltk
from nltk.corpus import stopwords
from pprint import pprint

# 4467 655


def lecture(filePath):
    with open(filePath, 'r') as f:
        content = f.readlines()
    return content

# normalize the text
def normalize(content_lines):
    all=[]
    for line in content_lines:
        line = regex.sub(r'\p{P}', lambda m: f' {m.group(0)} ', line)
        line = regex.sub(r'\p{P}', ' \1', line)
        line = regex.sub(r'\x01', ' . ', line)
        line = regex.sub(r'\s+', ' ', line)
        line = line.lower()
        if line != " ":
            words = line.split()
            all.extend(words)
    return all


def count_tokens(corpus, removePunctuation=True, removeStopWords=True):
    N = {}
    for word in corpus:
        if removePunctuation: 
            if word in string.punctuation:
                continue
        if removeStopWords:
            if word in stopwords.words('english'):
                continue
        if word not in N.keys():
            N[word] = 1
        N[word] += 1
        
    for token in N.keys():
        if N[token] < 5:
            del N[token]
        
    return N



def make_bigrams(corpus, removePunctuation=False, removeStopWords=False):
    B = {}
    for i in range(0, len(corpus)):
        if i < len(corpus)-1:
            if removePunctuation:
                if corpus[i] in string.punctuation or corpus[i+1] in string.punctuation:
                    continue
            if removeStopWords:
                if corpus[i] in stopwords.words('english') or corpus[i+1] in stopwords.words('english'):
                    continue
            bi = (corpus[i], corpus[i+1])
            if bi not in B.keys():
                B[bi] = 1
            else:
                B[bi] +=1
    
    for bigram in B:
        if B[bigram] < 5:
            del B[bigram]
    return B


def write_toFile(N, file):
    with open(file, 'w') as f:
        freqList = list(sorted(N.items(), key=lambda item: item[1], reverse=True))
        for line in freqList:
            f.write(f'{line[0]}: {line[1]}\n')
            
            
            
## math log2            
def make_bigrams_log(B, N):
    #  math.log(p_xy / (p_x * p_y), 2) # p_xy c'est ton freq_bigramme
    B_log = {}
    for bigram in B.keys():
        if B[bigram] > 5:
            freq_bigram = B[bigram] # / length(B)
            freq_mot1 = N[bigram[0]]
            freq_mot2 = N[bigram[1]]
            # B_log[bigram] = (freq_bigram/len(B)) / ((freq_mot1/len(N))* (freq_mot2/len(N)))
            # 
            B_log[bigram] = math.log2((freq_bigram/len(B)) / ((freq_mot1/len(N))* (freq_mot2/len(N))))
    return B_log
            
            
## main
def main():
    content_lines = lecture('acl/acl.txt')  # listes de listes
    print(len(content_lines))
    corpus = normalize(content_lines)
    print(f'length of corpus = {len(corpus)}')
    # print(corpus[:200])
    
    # occurences 
    N = count_tokens(corpus, removePunctuation=True, removeStopWords=True) # choisir sans ou avec ponctuation et/ou stopwords
    print(f'length of tokens = {len(N)}')
    pprint(list(sorted(N.items(), key=lambda item: item[1], reverse=True))[:10])
    
    # écriture dans un fichier
    write_toFile(N, file='acl/acl_freq.txt')
    
    
    # bigrammes
    B = make_bigrams(corpus, removePunctuation=True, removeStopWords=True)
    print(f'bigram length = {len(B)}\n\n')
    pprint(list(sorted(B.items(), key=lambda item: item[1],  reverse=True))[:10])
    print("\n\n")
    
    # écriture dans un fichier
    write_toFile(B, file='acl/acl_bigrams.txt')
    
    # 1) la fréquence de chaque mot type dans le corpus   # faire pdf 6 pour le bigramme
    # log2 = math.log(freq_bigramme/B ) / ( (freq_mot1/N) * (freq_mot2/N) ) 
    ## compare results with Notion -> affichage par ordre décroissant 
   
    B_log = make_bigrams_log(B, N)
    print(f'log of bigram length = {len(B_log)}\n\n')
    pprint(list(sorted(B_log.items(), key=lambda item: item[1], reverse=True))[:50])



if __name__ == '__main__':
  main()

