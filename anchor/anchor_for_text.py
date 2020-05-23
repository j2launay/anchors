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
import pandas as pd
from sklearn import tree
from sklearn.svm import LinearSVC
from sklearn.tree.export import export_text
from sklearn.neural_network import MLPClassifier
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
        
def predict_lr(texts):
    return int(model.predict(vectorizer.transform(texts))[0])

def predict_bert(texts):
    model = tuner(texts,'lime-tuner/distilBert_model.pkl', class_names)
    predict_result = utilities.JsonToArray(model.get_Prediction())[0]
    return np.argmax(predict_result)

def predicts_bert(texts):
    predict_texts = []
    for idx, text in enumerate(texts):
        print("instance number #", str(idx), text)
        model = tuner(text,'lime-tuner/distilBert_model.pkl', class_names)
        predict_result = utilities.JsonToArray(model.get_Prediction())[0]
        predict_texts.append(np.argmax(predict_result))
    return predict_texts

def print_text_explanation(predict, text, exp):        
    print(text)
    pred = class_names[predict([text])]
    alternative =  class_names[1 - predict([text])]
    #get the explanation for a particular sigma (all the explanation fields are filled)
    print('Prediction: %s' % pred)
    print('Anchor: %s' % (' AND '.join(exp.names())))
    print('Precision: %.2f' % exp.precision())
    print('Coverage: %.2f' % exp.coverage())
    print()
    print('Examples where anchor applies and model predicts %s:' % pred)
    print()
    print('\n'.join([x[0] for x in exp.examples(only_same_prediction=True)]))
    print()
    print('Examples where anchor applies and model predicts %s:' % alternative)
    print()
    print('\n'.join([x[0] for x in exp.examples(only_different_prediction=True)]))

    print('Partial anchor: %s' % (' AND '.join(exp.names(0))))
    print('Precision: %.2f' % exp.precision(0))
    print('Coverage: %.2f' % exp.coverage(0))
    print()
    print('Examples where anchor applies and model predicts %s:' % pred)
    print()
    print('\n'.join([x[0] for x in exp.examples(partial_index=0, only_same_prediction=True)]))
    print()
    print('Examples where anchor applies and model predicts %s:' % alternative)
    print()
    print('\n'.join([x[0] for x in exp.examples(partial_index=0, only_different_prediction=True)]))

dataset = 'bert'
if dataset == 'polarity':
    # dataset from http://www.cs.cornell.edu/people/pabo/movie-review-data/
    def load_polarity(path='/home/julien/Documents/stage/anchor/datasets/rt-polaritydata/rt-polaritydata'):
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
    class_names = ['positif', 'negatif']
    size_data = len(labels)
    train, test, train_labels, test_labels = sklearn.model_selection.train_test_split(data, labels, test_size=.9, random_state=42)
    #train, test, train_labels, test_labels = sklearn.model_selection.train_test_split(train, train_labels, test_size=.8, random_state=42)
    train, val, train_labels, val_labels = sklearn.model_selection.train_test_split(train, train_labels, test_size=.1, random_state=42)
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)
    val_labels = np.array(val_labels)

elif dataset == 'tweet':
    # If spanish change load_tweets to load_tweets_es() else only load_tweets()
    class_names = load_tweets.transform_emoji()
    train, train_labels = load_tweets.load_tweets()
    train, test, train_labels, test_labels = sklearn.model_selection.train_test_split(train, train_labels, test_size=.1, random_state=42)
    train, val, train_labels, val_labels = sklearn.model_selection.train_test_split(train, train_labels, test_size=.2, random_state=42)
    size_data = len(train_labels) + len(test_labels)

elif dataset == 'bert':
    class_names = ['negative',  'positive']
    df = pd.read_csv('https://github.com/clairett/pytorch-sentiment-classification/raw/master/data/SST2/train.tsv', delimiter='\t', header=None)
    batch_1 = df[:40]
    train_vectors = batch_1[0]
    train = train_vectors
    train_labels = batch_1[1]
    df = pd.read_csv('https://github.com/clairett/pytorch-sentiment-classification/raw/master/data/SST2/test.tsv', delimiter='\t', header=None)
    batch_2 = df[:20]
    test_vectors = batch_2[0]
    test = test_vectors
    test_labels = batch_2[1]
    size_data = len(train_labels) + len(test_labels)

# Vectorize data from train, validataion and test sets
if dataset != 'bert':
    vectorizer = CountVectorizer(min_df=1)
    vectorizer.fit(train)
    train_vectors = vectorizer.transform(train)
    test_vectors = vectorizer.transform(test)
    #val_vectors = vectorizer.transform(val)

# spanish language can only work with tweet dataset
# french language doesn't work (for the moment eheh)
language = 'english'
np.random.seed(1)

# Change use_unk_distribution to true generate mask word instead of random word
if language == 'french':
    "j'utilise le modèle français"
    nlp = spacy.load('fr_core_news_md')
    explainer = anchor_text.AnchorText(nlp, ['negative', 'positive', 'neutre'], use_unk_distribution=True)
    text = "C'est une jolie phrase . En Bretagne nous avons l'habitude de bien parler, c'est une grande qualite."
    pred = explainer.class_names[ps.predictFrench([text])]
    alternative =  explainer.class_names[1 - ps.predictFrench([text])]
    exp = explainer.explain_instance(text, ps.predictFrench, threshold=0.95, use_proba=False)

elif language == 'english':
    nlp = spacy.load('/home/julien/Documents/stage/anchor/datasets/en_core_web_lg-2.2.5/en_core_web_lg/en_core_web_lg-2.2.5')
    if dataset == 'tweet':
        print("emojis possibles : ", load_tweets.transform_emoji())

elif language == 'spanish':
    nlp = spacy.load('es_core_news_md')
    text = "La magia son personas. Nada es imposible. en Teatro Rialto"


#models = [sklearn.linear_model.LogisticRegression(), MLPClassifier(alpha=1, max_iter=100), tree.DecisionTreeClassifier(), 
#        LinearSVC(random_state=0, tol=1e-5), sklearn.ensemble.RandomForestClassifier(n_estimators=2, n_jobs=10), MultinomialNB(alpha=0.1)]
models = [tuner(train[7],'lime-tuner/distilBert_model.pkl', class_names)] if dataset=='bert' else [MLPClassifier(alpha=1, max_iter=50)]
bert = dataset=='bert'
experiment = False
x = [False, True]
threshold=0.2
if experiment:
    for model in models:
        bb_model = type(model).__name__
        print("model : ", bb_model)
        filename = "graph/" + dataset + "/" + bb_model + "/" + str(threshold) + "/"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        if dataset != 'bert':
            model.fit(train_vectors, train_labels)
            train_preds = model.predict(train_vectors)
            test_preds = model.predict(test_vectors)
        else:
            train_preds = predicts_bert(train_vectors)
            test_preds = predicts_bert(test_vectors)
        train_accuracy = sklearn.metrics.accuracy_score(train_labels, train_preds)
        test_accuracy = sklearn.metrics.accuracy_score(test_labels, test_preds)
        print('Val accuracy', test_accuracy)

        time_experiment = 10
        model_time = []
        mean_model_coverage = []
        mean_model_precision = []
        mean_size_explanation = []

        for j in range (len(x)):
            start = time()
            explainer = anchor_text.AnchorText(nlp, class_names, use_unk_distribution=x[j])
            mean_precision = 0
            mean_coverage = 0
            mean_size = 0
            print("Use of mask word : ", x[j])
            for i in range (time_experiment):
                print("instance numéro :", i)
                text = safe_str(test[i])
                print(text)
                if dataset != 'bert':
                    pred = class_names[predict_lr([text])]
                    alternative =  class_names[1 - predict_lr([text])]
                else:
                    pred = class_names[predict_bert([text])]
                    alternative =  class_names[1 - predict_bert([text])]
                print("prediction : ", pred)
                if dataset != 'bert':
                    exp = explainer.explain_instance(text, predict_lr, threshold=threshold, use_proba=True)
                else:
                    exp = explainer.explain_instance(text, predict_bert, threshold=threshold, use_proba=True)
                mean_coverage = mean_coverage + exp.coverage()
                mean_precision = mean_precision + exp.precision()
                mean_size = mean_size + len(exp.features())
            file_csv = (filename + "/example_result_" + str(x[j]) + ".csv")
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
        x = ['Replace all words', 'Use of mask word']
        for i in range(len(mean_model_coverage)):
            mean_model_coverage_precision.append(mean_model_coverage[i]*mean_model_precision[i])

        
        graph_coverage = baseGraph.BaseGraph(title="Difference of coverage for mask words", y_label="coverage", 
                    model=bb_model, accuracy=test_accuracy, dataset=dataset, threshold=threshold)
        graph_coverage.show_coverage(model=x, mean_coverage=mean_model_coverage)
        graph_precision = baseGraph.BaseGraph(title="Difference of precision for mask words", y_label="precision", 
                    model=bb_model, accuracy=test_accuracy, dataset=dataset, threshold=threshold)
        graph_precision.show_precision(model=x, mean_precision=mean_model_precision)
        graph_coverage_precision = baseGraph.BaseGraph(title="Difference of coverage * precision for mask words", 
                    y_label="coverage * precision", model=bb_model, accuracy=test_accuracy, dataset=dataset,
                    threshold=threshold)
        graph_coverage_precision.show_coverage_precision(model=x, mean_coverage_precision=mean_model_coverage_precision)
        graph_time = baseGraph.BaseGraph(title="Results of time for mask words prediction", y_label="time", 
                    model=bb_model, accuracy=test_accuracy, dataset=dataset, threshold=threshold)
        graph_time.show_time(model=x, mean_time=model_time)
        graph_size = baseGraph.BaseGraph(title="Results of size for mask words prediction", y_label="size", 
                    model=bb_model, accuracy=test_accuracy, dataset=dataset, threshold=threshold)
        graph_size.show_size(model=x, mean_size=mean_size_explanation)
        graph_size.writer_in_csv(dataset_name=dataset, dataset_size=size_data, bb_name=bb_model, bbox_train=train_accuracy, 
                                bbox_test=test_accuracy, precision=mean_model_precision, coverage=mean_model_coverage, 
                                coverage_precision=mean_model_coverage_precision, size=mean_size_explanation, 
                                time=model_time, time_experiment=time_experiment, x=x, threshold=threshold)
elif bert:
    class_names = ['negative',  'positive']
    df = pd.read_csv('https://github.com/clairett/pytorch-sentiment-classification/raw/master/data/SST2/train.tsv', delimiter='\t', header=None)
    batch_1 = df[:20]
    train_vectors = batch_1[0]
    train_labels = batch_1[1]
    df = pd.read_csv('https://github.com/clairett/pytorch-sentiment-classification/raw/master/data/SST2/test.tsv', delimiter='\t', header=None)
    batch_2 = df[:10]
    test_vectors = batch_2[0]
    test_labels = batch_2[1]
    
    #instances = [0, 50, 275, 350, 500, 623, 891, 1002, 1502, 1768]
    instances = [5, 6, 7]

    filename = "graph/bert/" + str(threshold) + "/"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    preds = predicts_bert(train_vectors)
    train_accuracy = sklearn.metrics.accuracy_score(train_labels, preds)
    preds = predicts_bert(test_vectors)
    test_accuracy = sklearn.metrics.accuracy_score(test_labels, preds)
    print('Val accuracy', test_accuracy)

    explainer = anchor_text.AnchorText(nlp, class_names, use_unk_distribution=False)
    for idx in instances :
        print('Computing explanations and curves for instance #', str(idx))
        text = batch_1[0][idx]
        exp = explainer.explain_instance(text, predict_bert, threshold=threshold, use_proba=True)
        print_text_explanation(predict_bert, text, exp)

else:
    #model = tree.DecisionTreeClassifier()
    model = models[0]
    model.fit(train_vectors, train_labels)

    for j in range (len(x)):
        explainer = anchor_text.AnchorText(nlp, class_names, use_unk_distribution=x[j])
        print("Use of mask word : ", x[j])
        print("model: ", type(model).__name__)
        #text = safe_str(test[0])
        text = "This is good movie"
        print("text ", text)
        pertinents_negatifs = (not x[j])
        exp = explainer.explain_instance(text, predict_lr, threshold=threshold, use_proba=True, pertinents_negatifs=pertinents_negatifs)
        print_text_explanation(predict_lr, text, exp)
