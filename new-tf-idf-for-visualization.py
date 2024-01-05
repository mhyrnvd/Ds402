import math
import string
from collections import Counter
import time 
import numpy as np

def tf(documents,words):
    num_words = len(words)
    p_words = Counter(documents.split())
    vectors = np.zeros(num_words)
    for word in p_words.keys():
        tf_word = p_words[word] / num_words
        vectors[words[word]] = tf_word
    return vectors

def idf(word_doc_appearances, words, num_paragraphs):
    vector = np.zeros(len(words))
    for w in words.keys():
        vector[words[w]] = math.log10(num_paragraphs / word_doc_appearances[w])
    return vector

data = []
all_documents = []

for z in range(50001):  
            
    with open(f'data/document_{z}.txt', 'r',encoding='utf-8', errors='ignore') as file:
        text = file.read().lower()  

        text = text.translate(str.maketrans('', '', string.punctuation))
    for x in text.splitlines():
        all_documents.append(x)
    print(z)

prepositions = ['aboard', 'about', 'above', 'across', 'after', 'against', 'along', 'amid', 'among', 'around', 'as',
            'at', 'before', 'behind', 'below', 'beneath', 'beside', 'between', 'beyond', 'but', 'by', 'concerning',
            'considering', 'despite', 'down', 'during', 'except', 'following', 'for', 'from', 'in', 'inside', 'into',
            'like', 'near', 'of', 'off', 'on', 'onto', 'out', 'outside', 'over', 'past', 'regarding', 'round',
            'since', 'through', 'throughout', 'to', 'toward', 'under', 'underneath', 'until', 'unto', 'up',
            'upon', 'with', 'within', 'without','the','a','or','which','where','per','and','are','most','its','an','that','be','is','may','it']
i = 0
words = {}

arr = []

print("len_all:",len(all_documents))
all_words = ' '.join(all_documents).split()

word_count = Counter(all_words)

num_paragraphs = len(all_documents)

for word in word_count.keys():
    words[word] = i
    i += 1


word_doc_count = {word: set() for word in words.keys()}


for idx, document in enumerate(all_documents):
    words_in_doc = set(document.split())
    for word in words_in_doc:
        if word in words:
            word_doc_count[word].add(idx)
    print("idx:",idx)

word_doc_appearances = {word: len(docs) for word, docs in word_doc_count.items()}

s = time.time()
dic = {}
for z in range(50001):  
    e = time.time()
    with open(f'data/document_{z}.txt', 'r',encoding='utf-8', errors='ignore') as file:
        text = file.read().lower()  

        text = text.translate(str.maketrans('', '', string.punctuation))

    documents = text.splitlines()
    
    doc = ' '.join(documents)
    tf_vector=tf(doc,words)
    e = time.time()
    print('tf_time:',e-s)
    
    idf_vector=idf(word_doc_appearances,words,num_paragraphs)
    e = time.time()
    print('idf_time:',e-s)

    tf_array = tf_vector
    idf_array = idf_vector

    total_vector = tf_array * idf_array
    
    with open(f'doc{z}.txt', 'a', encoding='utf-8', errors='ignore') as f:
        for item in total_vector:
            f.write(str(item)+ ' ')
        f.write('\n')  
    print("z:",z)
    
e = time.time()

print(e-s)