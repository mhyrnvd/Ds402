import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
from collections import Counter
import math
import numpy as np
from difflib import SequenceMatcher

# Download NLTK data (if not already downloaded)
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

def most_similar_word(word, word_dict):
    max_ratio = 0
    similar_word = ''

    for dict_word in word_dict.keys():
        seq_matcher = SequenceMatcher(None, word, dict_word)
        similarity_ratio = seq_matcher.ratio()

        if similarity_ratio > max_ratio:
            max_ratio = similarity_ratio
            similar_word = dict_word

    if max_ratio > 0.7:
        return similar_word
    else:
        return ''

def tf(documents,words):
    num_words = len(words)
    p_words = Counter(documents.split())
    vectors = np.zeros(num_words)
    for word in p_words.keys():

        tf_word = p_words[word] / num_words
        vectors[words[word]] = tf_word
    return vectors

def idf(doc,word_doc_appearances, words, num_paragraphs):
    vector = np.zeros(len(words))
    p_words = Counter(doc.split())
    for w in p_words.keys():
        vector[words[w]] = math.log10(num_paragraphs / (word_doc_appearances[w] + 1))
    return vector

for z in range(0,50001):  
    all_documents = []
    with open(f'data/document_{z}.txt', 'r',encoding='utf-8', errors='ignore') as file:
        text = file.read().lower()  

        text = text.translate(str.maketrans('', '', string.punctuation))
    for x in text.splitlines():
        all_documents.append(x)
    all_words = ' '.join(all_documents)

    tokens = word_tokenize(all_words)

    # Initialize the WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()

    # Lemmatize each token
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Get English stopwords
    stop_words = set(stopwords.words('english'))

    # Remove stopwords
    filtered_tokens = [token for token in lemmatized_tokens if token.lower() not in stop_words]


    word_count = Counter(filtered_tokens)

    num_paragraphs = len(all_documents)

    words = {}
    i = 0
    for word in word_count.keys():
        words[word] = i
        i += 1


    word_doc_count = {word: set() for word in words.keys()}

    
    for idx, document in enumerate(all_documents):
        words_in_doc = set(document.split())
        for word in words_in_doc:
            w = most_similar_word(word,words)
            if w:
                word_doc_count[w].add(idx)

    word_doc_appearances = {word: len(docs) for word, docs in word_doc_count.items()}

    tf_vector = tf(' '.join(filtered_tokens),words)
    idf_vector = idf(' '.join(filtered_tokens),word_doc_appearances,words,num_paragraphs)

    tf_idf = tf_vector * idf_vector

    print(tf_idf)
