import nltk
import os
from nltk import bigrams
import itertools
from nltk.stem import WordNetLemmatizer 
from nltk.collocations import *
from nltk.corpus import stopwords
#nltk.download('stopwords')
from nltk.tokenize import word_tokenize

def generate_co_occurence_matrix():
    path='/home/julien/Documents/stage/anchor/datasets/rt-polaritydata/rt-polaritydata'
    text_cooccurrence = ""
    """
    f_names = ['rt-polarity (copie).neg', 'rt-polarity (copie).pos']
    i = 0
    lemmatizer = WordNetLemmatizer()
    tokenizer = nltk.RegexpTokenizer(r"\w+")
    for (l, f) in enumerate(f_names):
        for line in open(os.path.join(path, f), 'rb'):
            i = i + 1
            try:
                line.decode('utf8')
            except:
                continue
            #if (i % 10) == 0:
            coocurrence_line = line.strip().split() 
            for word in coocurrence_line:
                word = word.decode("utf-8")
                word = word_tokenize(word)[0]
                word = [word if not word in stopwords.words() else ","][0]
                if tokenizer.tokenize(word) != tokenizer.tokenize(","):
                    word = tokenizer.tokenize(word)[0]
                    text_cooccurrence = text_cooccurrence + " " + lemmatizer.lemmatize(word)
    """
    with open(path + "/cooccurrence.data", "r") as f:

        text_cooccurrence = f.readline()       
    return text_cooccurrence
    #print(data, labels, text_cooccurrence)

def generate_n_best_co_occurrence(text):
    #text = "I do not like green eggs and ham, I do not like them Sam I am!"
    tokens = nltk.wordpunct_tokenize(text)
    finder = BigramCollocationFinder.from_words(tokens)
    bigram_measures = nltk.collocations.BigramAssocMeasures()
    #scored = finder.score_ngrams(bigram_measures.raw_freq)
    finder.apply_freq_filter(3)
    #print(sorted(bigram for bigram, score in scored))  # doctest: +NORMALIZE_WHITESPACE
    #print(finder.ngram_fd[('the', 'most')])
    #print(finder.nbest(bigram_measures.pmi, 1000))
    n_best_co_occurrence = finder.nbest(bigram_measures.pmi, 100000000)
    return n_best_co_occurrence

def generate_bi_grams_words(target, n_best_co_occurrence, number_words=20):
    lemmatizer = WordNetLemmatizer()
    tokenizer = nltk.RegexpTokenizer(r"\w+")
    target = target.lower()
    word = [target if not target in stopwords.words() else ","][0]
    word = tokenizer.tokenize(word)
    if len(word) == 0:
        targets = [""]
        #weights = [1]
        return targets#, weights
    elif len(word) == 1:
        word = word[0]
    else:
        word = word[1]
    target = lemmatizer.lemmatize(word)
    targets = []
    #weights = []
    number = 0
    for element in n_best_co_occurrence:
        if number == 20:
            break
        if str(target) in element:
            number = number + 1
            if str(target) in element[0]:
                targets.append(element[1])
            else:
                targets.append(element[0])
    """
    for i in range(len(targets)):
        weights.append(1/number)
    if len(targets) == 0:
        targets = [""]
        weights = [1]"""
    targets = set(targets)
    targets = [i for i in targets]
    return targets

"""
text = generate_co_occurence_matrix()
n_best_co_occurrence = generate_n_best_co_occurrence(text)
generate_bi_grams_words("love", n_best_co_occurrence)
"""
