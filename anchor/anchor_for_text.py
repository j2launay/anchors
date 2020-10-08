#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 14:53:51 2020

@author: jdelauna
"""

from __future__ import print_function
import os
import os.path
import numpy as np
import sklearn
import sklearn.model_selection
import sklearn.linear_model
import sklearn.ensemble
from sklearn import tree
from sklearn.svm import LinearSVC
from sklearn.tree.export import export_text
from sklearn.neural_network import MLPClassifier
import pandas as pd
import spacy
import sys
import csv
#import predictSentiment as ps
import load_tweets
import emoji
import baseGraph
import generate_dataset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import anchor_text
from time import time
import warnings
sys.path.append('lime-tuner/LIME_TUNER_LIBRARY/')
import utilities
from tuner_library import tuner
from lime.lime_text import LimeTextExplainer

warnings.filterwarnings("ignore")
os.environ['SPACY_WARNING_IGNORE'] = 'W008'

def safe_str(obj):
    try: 
        return str(obj)
    except UnicodeEncodeError:
        return obj.encode('ascii', 'ignore').decode('ascii')
    return ""

# Function to predict class for a text "texts"        
def predict_lr(texts):
    if len(texts) > 1:
        return model.predict(vectorizer.transform(texts))
    return int(model.predict(vectorizer.transform(texts))[0])

# Function to predict class for a text with Bert model 
def predict_bert(texts):
    if len(texts) > 100:
        return predicts_bert(texts)
    model = tuner(texts,'lime-tuner/distilBert_model.pkl', class_names)
    predict_result = utilities.JsonToArray(model.get_Prediction())[0]
    return np.argmax(predict_result)

# Function to predict classes for multiple text with a Bert model
def predicts_bert(texts):
    predict_texts = []
    for idx, text in enumerate(texts):
        print("instance number #", str(idx), text)
        model = tuner(text,'lime-tuner/distilBert_model.pkl', class_names)
        predict_result = utilities.JsonToArray(model.get_Prediction())[0]
        predict_texts.append(np.argmax(predict_result))
    return predict_texts

# Function to display all the interesting information with an anchor (i.e: precision, coverage, example where it applies or not, etc...)
def print_text_explanation(predict, text, exp):        
    print(text)
    pred = class_names[predict([text])]
    alternative =  class_names[1 - predict([text])]
    # get the explanation for a particular sigma (all the explanation fields are filled)
    print('Prediction: %s' % pred)
    print('Anchor: %s' % (' AND '.join(exp.names())))
    print('Precision: %.2f' % exp.precision())
    print('Coverage: %.2f' % exp.coverage())
    print()
    print('Examples where anchor applies and model predicts %s:' % pred)
    print()
    print('\n'.join([mask_word_boolean[0] for mask_word_boolean in exp.examples(only_same_prediction=True)]))
    print()
    print('Examples where anchor applies and model predicts %s:' % alternative)
    print()
    print('\n'.join([mask_word_boolean[0] for mask_word_boolean in exp.examples(only_different_prediction=True)]))

    print('Partial anchor: %s' % (' AND '.join(exp.names(0))))
    print('Precision: %.2f' % exp.precision(0))
    print('Coverage: %.2f' % exp.coverage(0))
    print()
    print('Examples where anchor applies and model predicts %s:' % pred)
    print()
    print('\n'.join([mask_word_boolean[0] for mask_word_boolean in exp.examples(partial_index=0, only_same_prediction=True)]))
    print()
    print('Examples where anchor applies and model predicts %s:' % alternative)
    print()
    print('\n'.join([mask_word_boolean[0] for mask_word_boolean in exp.examples(partial_index=0, only_different_prediction=True)]))

# Parameter representing the dataset. There is three possibilities : 'polarity', 'tweet' and 'bert'
dataset = 'polarity'
#Name: 0, Length: 120, dtype: object
size_batch_bert = 40
if dataset == 'polarity':
    # dataset from http://www.cs.cornell.edu/people/pabo/movie-review-data/
    def load_polarity(path='/Users/jdelauna/Documents/anchors/datasets/rt-polaritydata/rt-polaritydata'):
        data = []
        labels = []
        f_names = ['rt-polarity.neg', 'rt-polarity.pos']
        for (l, f) in enumerate(f_names):
            for line in open(os.path.join(path, f), 'rb'):
                try:
                    line.decode('utf8')
                except:
                    continue
                data.append(line.strip())
                labels.append(l)
        return data, labels

    # Separate data into train, validation and test sets
    data, labels = load_polarity()
    class_names = ['negatif', 'positif']
    size_data = len(labels)
    train, test, train_labels, test_labels = sklearn.model_selection.train_test_split(data, labels, test_size=.9, random_state=42)
    train, val, train_labels, val_labels = sklearn.model_selection.train_test_split(train, train_labels, test_size=.1, random_state=42)
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)
    val_labels = np.array(val_labels)

elif dataset == 'tweet':
    # If spanish change load_tweets to load_tweets_es() else only load_tweets()
    class_names = load_tweets.transform_emoji()
    train, train_labels = load_tweets.load_tweets()
    train, test, train_labels, test_labels = sklearn.model_selection.train_test_split(train, train_labels, test_size=.9, random_state=42)
    train, val, train_labels, val_labels = sklearn.model_selection.train_test_split(train, train_labels, test_size=.2, random_state=42)
    size_data = len(train_labels) + len(test_labels)

elif 'bert' in dataset:
    class_names = ['negative',  'positive']
    df = pd.read_csv('https://github.com/clairett/pytorch-sentiment-classification/raw/master/data/SST2/train.tsv', delimiter='\t', header=None)
    # Number of sentance examples on which the bert model is trained on the train set (limited since the model takes time)
    batch_1 = df[:size_batch_bert]
    train_vectors = batch_1[0]
    train_bert = train_vectors
    train_labels = batch_1[1]
    df = pd.read_csv('https://github.com/clairett/pytorch-sentiment-classification/raw/master/data/SST2/test.tsv', delimiter='\t', header=None)
    # Number of sentance examples on which the bert model is trained on the test set (limited since the model takes time)
    batch_2 = df[:int(size_batch_bert/2)]
    test_vectors = batch_2[0]
    test_bert = test_vectors
    test_labels = batch_2[1]
    size_data = len(train_labels) + len(test_labels)

# Vectorize data from train, validataion and test sets
if dataset != 'bert':
    vectorizer = CountVectorizer(min_df=1)
    vectorizer.fit(train)
    train_vectors = vectorizer.transform(train)
    test_vectors = vectorizer.transform(test)
    #val_vectors = vectorizer.transform(val)
    safe_str_bert = []
    for i in train:
        safe_str_bert.append(safe_str(i))
    train_bert = pd.DataFrame (safe_str_bert)
    batch = train_bert[:120]
    train_bert = batch[0]
    safe_str_bert = []
    for i in test:
        safe_str_bert.append(safe_str(i))
    test_bert = pd.DataFrame (safe_str_bert)
    batch = test_bert[:60]
    test_bert = batch[0]
    #test_bert = pd.DataFrame (test, columns=['sentences'])

# spanish language can only work with tweet dataset
# french language doesn't work (for the moment eheh)
language = 'english'
np.random.seed(1)

# Change use_unk_distribution to true generate mask word instead of random word to replace word
if language == 'french':
    # Need to import predictSentiment as ps
    "j'utilise le modèle français"
    nlp = spacy.load('fr_core_news_md')
    explainer = anchor_text.AnchorText(nlp, ['negative', 'positive', 'neutre'], use_unk_distribution=True)
    text = "C'est une jolie phrase . En Bretagne nous avons l'habitude de bien parler, c'est une grande qualite."
    pred = explainer.class_names[ps.predictFrench([text])]
    alternative =  explainer.class_names[1 - ps.predictFrench([text])]
    exp = explainer.explain_instance(text, ps.predictFrench, threshold=0.95, use_proba=False)

elif language == 'english':
    nlp = spacy.load('/Users/jdelauna/Documents/anchors/datasets/en_core_web_lg-2.2.5/en_core_web_lg/en_core_web_lg-2.2.5')
    #nlp = spacy.load('en_core_web_lg')
    if dataset == 'tweet':
        print("emojis possibles : ", load_tweets.transform_emoji())

elif language == 'spanish':
    nlp = spacy.load('es_core_news_md')
    text = "La magia son personas. Nada es imposible. en Teatro Rialto"

# All the possible models that are used to test and compute explanations
models_all = [tuner(train_bert[7],'lime-tuner/distilBert_model.pkl', class_names), sklearn.linear_model.LogisticRegression(), sklearn.ensemble.RandomForestClassifier(n_estimators=2, n_jobs=10),
        MultinomialNB(alpha=0.1), MLPClassifier(alpha=1, max_iter=100), 
        tree.DecisionTreeClassifier(), LinearSVC(random_state=0, tol=1e-5)]
#models_all = [sklearn.linear_model.LogisticRegression(), tree.DecisionTreeClassifier()]
models = [tuner(train_bert[7],'lime-tuner/distilBert_model.pkl', class_names)] if dataset=='bert' else models_all
# Parameter used to compute the bert model or the "classic models"
bert = dataset=='bert'
# Parameter that enables (or not) to launch "time_experiment" times the experiments 
experiment = True
mask_word_boolean = [False, True]
mask_word_boolean_text = ['Replace all words', 'Use of mask word', 'Replace with pertinent negatif', 'Replace words and pertinent']
# Minimum threshold precision for a sufficiant anchor
threshold=0.95
if experiment:
    for model in models:
        bb_model = type(model).__name__
        print("model : ", bb_model)
        filename = "graph/" + dataset + "/" + bb_model + "/" + str(threshold) + "/"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        # Compute the test and train accuracy of the black box model
        if bb_model != 'tuner':
            model.fit(train_vectors, train_labels)
            train_preds = model.predict(train_vectors)
            test_preds = model.predict(test_vectors)
        else:
            train_preds = predicts_bert(train_bert)
            test_preds = predicts_bert(test_bert)
        train_accuracy = sklearn.metrics.accuracy_score(train_labels[:len(train_preds)], train_preds)
        test_accuracy = sklearn.metrics.accuracy_score(test_labels[:len(test_preds)], test_preds)
        print('Val accuracy', test_accuracy)

        # Number of times the experiment are launch (i.e: number of text that are explained by anchor)
        time_experiment = 10
        model_time = []
        mean_model_coverage = []
        mean_model_precision = []
        mean_size_explanation = []

        for j in range (len(mask_word_boolean_text)):
            start = time()
            if j < len(mask_word_boolean):
                print("Use of mask word: ", mask_word_boolean[j])
                explainer = anchor_text.AnchorText(nlp, class_names, use_unk_distribution=mask_word_boolean[j])
            else:
                print("Use of pertinent negatif") if j == 2 else print("Use of pertinent negatif and replace words ")
                explainer = anchor_text.AnchorText(nlp, class_names, use_unk_distribution=False)
            mean_precision = 0
            mean_coverage = 0
            mean_size = 0
            counter = 0
            number_sentences = 0
            while number_sentences < time_experiment:
            #for i in range (time_experiment):
                counter += 1
                text = safe_str(test[counter]) if bb_model != 'tuner' else safe_str(test_bert[counter])
                if len (text) > 60 :
                    continue
                number_sentences += 1
                print("instance number :", number_sentences)
                print(text)
                # compute the class prediction for the text from the black box model  
                if bb_model != 'tuner':
                    pred = class_names[predict_lr([text])]
                    alternative =  class_names[1 - predict_lr([text])]
                else:
                    pred = class_names[predict_bert([text])]
                    alternative =  class_names[1 - predict_bert([text])]
                print("prediction : ", pred)
                if j < len(mask_word_boolean):
                    if bb_model != 'tuner':
                        exp = explainer.explain_instance(text, predict_lr, threshold=threshold, use_proba=True)
                    else:
                        exp = explainer.explain_instance(text, predict_bert, threshold=threshold, use_proba=True)
                else:
                    if j == 2:  
                        if bb_model != 'tuner':
                            exp = explainer.explain_instance(text, predict_lr, threshold=threshold, use_proba=True, pertinents_negatifs=True)
                        else:
                            exp = explainer.explain_instance(text, predict_bert, threshold=threshold, use_proba=True, pertinents_negatifs=True)
                    else:
                        if bb_model != 'tuner':
                            exp = explainer.explain_instance(text, predict_lr, threshold=threshold, use_proba=True, pertinents_negatifs_replace=True)
                        else:
                            exp = explainer.explain_instance(text, predict_bert, threshold=threshold, use_proba=True, pertinents_negatifs_replace=True)
                mean_coverage = mean_coverage + exp.coverage()
                mean_precision = mean_precision + exp.precision()
                mean_size = mean_size + len(exp.features())
            file_csv = (filename + "/example_result_" + str(mask_word_boolean_text[j]) + ".csv")
            generate_dataset.print_example_text(text, pred, alternative, exp, file_csv)
            end = time()
            mean_coverage = mean_coverage / time_experiment
            mean_precision = mean_precision / time_experiment
            mean_size = mean_size / time_experiment
            model_time.append(end - start)
            mean_model_coverage.append(mean_coverage)
            mean_model_precision.append(mean_precision)
            mean_size_explanation.append(mean_size)
        print("mean coverage", mean_model_coverage)
        print("mean precision", mean_model_precision)
        mean_model_coverage_precision = []
        for counter in range(len(mean_model_coverage)):
            mean_model_coverage_precision.append(2/(1/mean_model_coverage[counter]+1/mean_model_precision[counter]))

        # Generates graphs for coverage, precision, F1, time and size of the explanation
        graph_coverage = baseGraph.BaseGraph(title="Difference of coverage for mask words", y_label="coverage", 
                    model=bb_model, accuracy=test_accuracy, dataset=dataset, threshold=threshold)
        graph_coverage.show_coverage(model=mask_word_boolean_text, mean_coverage=mean_model_coverage)
        graph_precision = baseGraph.BaseGraph(title="Difference of precision for mask words", y_label="precision", 
                    model=bb_model, accuracy=test_accuracy, dataset=dataset, threshold=threshold)
        graph_precision.show_precision(model=mask_word_boolean_text, mean_precision=mean_model_precision)
        graph_coverage_precision = baseGraph.BaseGraph(title="Difference of 2/(1/coverage + 1/precision) for mask words", 
                    y_label="2/(1/coverage + 1/precision)", model=bb_model, accuracy=test_accuracy, dataset=dataset,
                    threshold=threshold)
        graph_coverage_precision.show_coverage_precision(model=mask_word_boolean_text, mean_coverage_precision=mean_model_coverage_precision)
        graph_time = baseGraph.BaseGraph(title="Results of time for mask words prediction", y_label="time", 
                    model=bb_model, accuracy=test_accuracy, dataset=dataset, threshold=threshold)
        graph_time.show_time(model=mask_word_boolean_text, mean_time=model_time)
        graph_size = baseGraph.BaseGraph(title="Results of size for mask words prediction", y_label="size", 
                    model=bb_model, accuracy=test_accuracy, dataset=dataset, threshold=threshold)
        graph_size.show_size(model=mask_word_boolean_text, mean_size=mean_size_explanation)
        # Store all the information that are essential to reproduce the experimentation
        graph_size.writer_in_csv(dataset_name=dataset, dataset_size=size_data, bb_name=bb_model, bbox_train=train_accuracy, 
                                bbox_test=test_accuracy, precision=mean_model_precision, coverage=mean_model_coverage, 
                                coverage_precision=mean_model_coverage_precision, size=mean_size_explanation, 
                                time=model_time, time_experiment=time_experiment, x=mask_word_boolean_text, threshold=threshold)
elif bert:
    class_names = ['negative',  'positive']
    df = pd.read_csv('https://github.com/clairett/pytorch-sentiment-classification/raw/master/data/SST2/train.tsv', delimiter='\t', header=None)
    # Number of sentance examples on which the bert model is trained on the train set (limited since the model takes time)
    batch_1 = df[:20]
    train_vectors = batch_1[0]
    train_labels = batch_1[1]
    df = pd.read_csv('https://github.com/clairett/pytorch-sentiment-classification/raw/master/data/SST2/test.tsv', delimiter='\t', header=None)
    # Number of sentance examples on which the bert model is trained on the test set (limited since the model takes time)
    batch_2 = df[:10]
    test_vectors = batch_2[0]
    test_labels = batch_2[1]
    
    #instances = [0, 50, 275, 350, 500, 623, 891, 1002, 1502, 1768]
    # Instances that are explained
    instances = [0, 1, 2]

    filename = "graph/bert/" + str(threshold) + "/"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    # Compute accuracy on test and train dataset
    pred = predict_bert(batch_1[0][5])
    preds = predicts_bert(train_vectors)
    train_accuracy = sklearn.metrics.accuracy_score(train_labels, preds)
    preds = predicts_bert(test_vectors)
    test_accuracy = sklearn.metrics.accuracy_score(test_labels, preds)
    print('Val accuracy', test_accuracy)

    # Compute explanation for each instance 
    explainer = anchor_text.AnchorText(nlp, class_names, use_unk_distribution=False)
    if experiment:
        for idx in instances :
            print('Computing explanations and curves for instance #', str(idx))
            text = batch_1[0][idx]
            print('text :', text)
            exp = explainer.explain_instance(text, predict_bert, threshold=threshold, use_proba=True)
            print_text_explanation(predict_bert, text, exp)
    else:
        text = "This is a good book."
        exp = explainer.explain_instance(text, predict_bert, threshold=threshold, use_proba=True)
        print_text_explanation(predicts_bert, text, exp)
        

else:
    #model = tree.DecisionTreeClassifier()
    model = models[0]
    model.fit(train_vectors, train_labels)
    mask = True
    for j in range (len(mask_word_boolean) + 2):            
        explainer = anchor_text.AnchorText(nlp, class_names, use_unk_distribution=mask)
        print("Use of mask word : ", mask)
        mask = False
        print("model: ", type(model).__name__)
        print("dataset: ", dataset)
        #text = safe_str(test[0])
        text = "They seems movies"
        print("text ", text)
        pertinents_negatifs = (j == 2)
        pertinents_negatifs_replace = (j == 3)
        print("Use of pertinent negatifs ", pertinents_negatifs)
        print("Use of pertinent negatifs and replace words ", pertinents_negatifs_replace)
        exp = explainer.explain_instance(text, predict_lr, threshold=threshold, use_proba=True, pertinents_negatifs_replace=pertinents_negatifs_replace)
        print_text_explanation(predict_lr, text, exp)
