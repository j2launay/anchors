"""Co-occurrence matrix generation for pertinent negative computation.

This module builds word co-occurrence statistics from a text corpus
(polarity dataset) and provides functions to find frequently
co-occurring word pairs for use in pertinent negative generation.
"""
import itertools
import os

import nltk
from nltk import bigrams
from nltk.collocations import BigramCollocationFinder
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


def generate_co_occurence_matrix(generateNew=False, dataset_path=None):
    """Generate or load a word co-occurrence representation from text.

    Processes text from the polarity dataset to build co-occurrence
    statistics. Can either generate fresh statistics or load
    pre-computed ones from disk.

    Args:
        generateNew: If True, computes a new co-occurrence matrix from
            the raw text files. If False, loads the pre-computed one.
        dataset_path: Path to the datasets directory. If None, defaults
            to '../datasets' relative to this file's location.

    Returns:
        str: A space-separated string of lemmatized words representing
        the co-occurrence corpus.
    """
    if dataset_path is None:
        dataset_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), '..', 'datasets')

    text_cooccurrence = ""
    if generateNew:
        f_names = ['rt-polarity (copie).neg', 'rt-polarity (copie).pos']
        lemmatizer = WordNetLemmatizer()
        tokenizer = nltk.RegexpTokenizer(r"\w+")
        for (l, f) in enumerate(f_names):
            for line in open(os.path.join(dataset_path, f), 'rb'):
                try:
                    line.decode('utf8')
                except UnicodeDecodeError:
                    continue
                # Split the line by words
                coocurrence_line = line.strip().split()
                for word in coocurrence_line:
                    word = word.decode("utf-8")
                    word = word_tokenize(word)[0]
                    word = [word if not word in stopwords.words() else ","][0]
                    if tokenizer.tokenize(word) != tokenizer.tokenize(","):
                        word = tokenizer.tokenize(word)[0]
                        text_cooccurrence = text_cooccurrence + " " + lemmatizer.lemmatize(word)
    else:
        cooccurrence_file = os.path.join(dataset_path, "cooccurrence.data")
        with open(cooccurrence_file, "r") as f:
            text_cooccurrence = f.readline()
    return text_cooccurrence


def generate_n_best_co_occurrence(text, n=100000000):
    """Find the n most significant word bigram collocations.

    Uses Pointwise Mutual Information (PMI) to rank bigram
    collocations found in the input text.

    Args:
        text: A string of space-separated words (the co-occurrence corpus).
        n: Maximum number of top collocations to return.

    Returns:
        List of (word1, word2) tuples representing the top collocations,
        sorted by PMI score.
    """
    tokens = nltk.wordpunct_tokenize(text)
    finder = BigramCollocationFinder.from_words(tokens)
    bigram_measures = nltk.collocations.BigramAssocMeasures()
    finder.apply_freq_filter(3)
    n_best_co_occurrence = finder.nbest(bigram_measures.pmi, n)
    return n_best_co_occurrence


def generate_bi_grams_words(target, n_best_co_occurrence, number_words=20):
    """Find words that frequently co-occur with a target word.

    Searches the pre-computed co-occurrence collocations for pairs
    containing the target word and returns the co-occurring words.

    Args:
        target: The target word to find co-occurrences for.
        n_best_co_occurrence: List of (word1, word2) collocations from
            generate_n_best_co_occurrence().
        number_words: Maximum number of co-occurring words to return.

    Returns:
        List of words that frequently co-occur with the target word.
    """
    # We start by using classical natural language processing steps
    lemmatizer = WordNetLemmatizer()
    tokenizer = nltk.RegexpTokenizer(r"\w+")
    target = target.lower()
    word = [target if not target in stopwords.words() else ","][0]
    word = tokenizer.tokenize(word)
    # If the word is a stopword
    if len(word) == 0:
        targets = []
        return targets
    elif len(word) == 1:
        word = word[0]
    else:
        word = word[1]
    target = lemmatizer.lemmatize(word)
    targets = []
    number = 0
    for element in n_best_co_occurrence:
        if number == number_words:
            break
        if str(target) in element:
            number = number + 1
            if str(target) in element[0]:
                targets.append(element[1])
            else:
                targets.append(element[0])
    targets = set(targets)
    targets = [i for i in targets]
    return targets
