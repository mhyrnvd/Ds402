import math
import string
from collections import Counter
import time 
from collections import Counter
import numpy as np
import math
import string
from collections import Counter
from difflib import SequenceMatcher
import json
import pickle

import numpy as np

def dot_product(vec1, vec2):
    return np.dot(vec1, vec2)

def magnitude(vector):
    return np.linalg.norm(vector)

def cosine_similarity(vec1, vec2):
    dot_prod = dot_product(vec1, vec2)
    mag_vec1 = magnitude(vec1)
    mag_vec2 = magnitude(vec2)
    
    if mag_vec1 != 0 and mag_vec2 != 0:
        return dot_prod / (mag_vec1 * mag_vec2)
    else:
        return 0

def most_similar_word(word, word_dict):
    max_ratio = 0
    similar_word = ''

    for dict_word in word_dict.keys():
        seq_matcher = SequenceMatcher(None, word, dict_word)
        similarity_ratio = seq_matcher.ratio()

        if similarity_ratio > max_ratio:
            max_ratio = similarity_ratio
            similar_word = dict_word

    if max_ratio > 0.8:
        return similar_word
    else:
        return ''

def tf(documents,words, prepositions):
    num_words = len(words)
    p_words = Counter(documents.split())
    vectors = np.zeros(num_words)
    for word in p_words.keys():
        if word not in prepositions:
            tf_word = p_words[word] / num_words
            vectors[words[word]] = tf_word
    return vectors

def idf(doc,word_doc_appearances, words, num_paragraphs, prepositions):
    vector = np.zeros(len(words))
    p_words = Counter(doc.split())
    for w in p_words.keys():
        if w not in prepositions:
            vector[words[w]] = math.log10(num_paragraphs / word_doc_appearances[w])
    return vector



data = []
all_documents = []


prepositions = ['aboard', 'about', 'above', 'across', 'after', 'against', 'along', 'amid', 'among', 'around', 'as',
            'at', 'before', 'behind', 'below', 'beneath', 'beside', 'between', 'beyond', 'but', 'by', 'concerning',
            'considering', 'despite', 'down', 'during', 'except', 'following', 'for', 'from', 'in', 'inside', 'into',
            'like', 'near', 'of', 'off', 'on', 'onto', 'out', 'outside', 'over', 'past', 'regarding', 'round',
            'since', 'through', 'throughout', 'to', 'toward', 'under', 'underneath', 'until', 'unto', 'up',
            'upon', 'with', 'within', 'without','the','a','or','which','where','per','and','are','most','its','an','that','be','is','may','it']
i = 0
words = {}

arr = []

with open('all_documents.pkl', 'rb') as file:
    all_documents = pickle.load(file)

print("len_all:",len(all_documents))
all_words = ' '.join(all_documents).split()

with open('dic.pkl', 'rb') as file:
    words = pickle.load(file)

word_count = Counter(all_words)

num_paragraphs = len(all_documents)



with open('word_doc_appearances.pkl', 'rb') as file:
    word_doc_appearances = pickle.load(file)

with open('data.json', 'r') as file:
    data = json.load(file)

s = time.time()
for p in range(0,50001):
    with open(f'data/document_{p}.txt', 'r',encoding='utf-8', errors='ignore') as file:
        t = file.read().lower()  

        t = t.translate(str.maketrans('', '', string.punctuation))
    results= []
    lis = data[p]["candidate_documents_id"]

    query = data[p]["query"]

    w = Counter(t.split())
    query_without_punctuation = query.translate(str.maketrans('', '', string.punctuation))
    query_without_punctuation=query_without_punctuation.lower()
    
    query = ''
    for query_word in query_without_punctuation.split():
        if query_word not in prepositions:
            query += most_similar_word(query_word, w) + " "


    word_count = Counter(query.split())

    tf_array = tf(query,words,prepositions)

    idf_array = idf(query,word_doc_appearances,words,num_paragraphs,prepositions)

    total_query_vector = tf_array * idf_array


    for z in lis:
        with open(f'doc{z}.txt','r') as f:
            content = f.read()
        s2 = time.time()
        tf_idf = np.fromiter(map(float, content.split()), dtype=float)

        s3 = time.time()
        results.append((cosine_similarity(total_query_vector, tf_idf),z))

    sorted_list = sorted(results, key=lambda x: x[0])

    selected_doc=sorted_list[-1][1]
    print(selected_doc)

    with open(f'data/document_{selected_doc}.txt', 'r',encoding='utf-8', errors='ignore') as file:
        text = file.read().lower() 

        text = text.translate(str.maketrans('', '', string.punctuation)) 

    documents = text.splitlines()

    paragraph_similarity = []

    for line in documents:
    
        tf_paragraph=tf(line,words,prepositions)
        idf_paragraph=idf(line,word_doc_appearances,words,num_paragraphs,prepositions)

        total_vector = tf_paragraph * idf_paragraph

        paragraph_similarity.append(cosine_similarity(total_query_vector,total_vector))

    print(paragraph_similarity)
    print(paragraph_similarity.index(max(paragraph_similarity)))
    print()
