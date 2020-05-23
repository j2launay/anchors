#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 14:30:36 2020

@author: jdelauna
"""

from __future__ import print_function
import numpy as np
np.random.seed(1)
import sklearn
import sklearn.ensemble
from sklearn.neural_network import MLPClassifier
import sklearn.linear_model
from sklearn import tree
from sklearn.svm import LinearSVC
from sklearn.tree.export import export_text
from sklearn.naive_bayes import MultinomialNB
import utils
import csv
import anchor_tabular
from generate_dataset import generate, compute_coverage, compute_precision
from sklearn.datasets import make_blobs, make_moons, make_circles
import predictSentiment as ps
import baseGraph
import copy
import os
from matplotlib import pyplot as plt
import time

# make sure you have adult/adult.data inside dataset_folder
dataset_folder = '/home/julien/Documents/stage/anchor/datasets/'
# for the moment; three possible datasets : adult (income revenue prediction)
#                                           titanic (survival prediction)
#                                           generate (2D generate dataset for visualisation)
dataset_name = 'generate_blobs'
if 'generate' in dataset_name:
    if 'moons' in dataset_name:
        X, Y = make_moons(n_samples=10000, noise=0.1)
    elif 'circles' in dataset_name:
        X, Y = make_circles(n_samples=10000, noise=0.05)
    else:
        dataset_name = 'generate_blobs'
        X, Y = make_blobs(n_samples=10000, centers=2, n_features=2)
    fichier_write = open("../datasets/generate/generate.data", "w+")
    for x, y in zip(X, Y):
        text = ("%s%s %s\n" % (str(x).replace('[ ', '').replace('[', '').replace(' [', '').replace(' ]', '').
                               replace(']', '').replace(' ', ', ').replace(', , ,', ',').replace(', ,', ','), ',', y))
        text = str(text).replace(', ,', ',')
        fichier_write.write(text)
    fichier_write.close()

dataset = utils.load_dataset(dataset_name, balance=True, discretize=False, dataset_folder=dataset_folder)
train, labels_train = dataset.train, dataset.labels_train
print("Names of the different features: ", dataset.feature_names)
print()
experiment = True
threshold=0.75
if experiment :
    models=[tree.DecisionTreeClassifier(), MLPClassifier(alpha=1, max_iter=100), LinearSVC(random_state=0, tol=1e-5), sklearn.linear_model.LogisticRegression(), 
                    sklearn.ensemble.RandomForestClassifier(n_estimators=30, n_jobs=5)]
    #models=[tree.DecisionTreeClassifier()]
    time_experiment = 50
    x = ['kmeans', 'decile', 'quartile', 'entropy', 'MDLP']

    for c in models:
        bb_model = type(c).__name__
        print('model: ', bb_model)
        filename = "graph/" + dataset_name + "/" + bb_model + "/" + str(threshold) + "/"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        c=tree.DecisionTreeClassifier()
        c.fit(train, labels_train)
        black_box_labels = c.predict(train)

        model_time = []
        mean_model_coverage = []
        mean_model_precision = []
        mean_size_explanation = []
        model_size_discretization = []
        feature_discretize = []
        for j in range (len(x)):
            start = time.time()
            mean_precision = 0
            mean_coverage = 0
            mean_size = 0
            print("Discretization: ", x[j])
            explainer = anchor_tabular.AnchorTabularExplainer(dataset.class_names, dataset.feature_names, dataset.train, 
                                                            copy.copy(dataset.categorical_names), black_box_labels=black_box_labels,
                                                            discretizer=x[j], filename=filename)
            for i in range (time_experiment):
                print("instance numéro :", i)

                predict_fn = lambda x: c.predict(x)
                bbox_train = sklearn.metrics.accuracy_score(dataset.labels_train, predict_fn(dataset.train))
                bbox_test = sklearn.metrics.accuracy_score(dataset.labels_test, predict_fn(dataset.test))
                print('Train', sklearn.metrics.accuracy_score(dataset.labels_train, predict_fn(dataset.train)))
                print('Test', sklearn.metrics.accuracy_score(dataset.labels_test, predict_fn(dataset.test)))

                idx = i
                #idx = 1
                #idx = 15
                np.random.seed(1)
                print("instance: ", dataset.test[idx].reshape(1, -1)[0])
                #print('Prediction: ', explainer.class_names[predict_fn(dataset.test[idx].reshape(1, -1))[0]])
                exp = explainer.explain_instance(dataset.test[idx], c.predict, threshold=threshold, 
                                                delta=0.1, tau=0.15, batch_size=100,
                                                max_anchor_size=None,
                                                stop_on_first=False,
                                                desired_label=None,
                                                beam_size=4)
                print('Anchor: %s' % (' AND '.join(exp.names())))                    

                coverage_tab = None
                precision_tab = None
                if 'generate' in dataset_name:
                    anchors=exp.names()
                    _, _, pick_anchors_informations = generate(dataset.test[idx].reshape(1, -1)[0][0], dataset.test[idx].reshape(1, -1)[0][1], anchors, 
                                                            blackbox=c, X= X, Y = Y)
                    coverage_tab = compute_coverage(dataset, exp, anchors, pick_anchors_informations)
                    precision_tab = compute_precision(dataset, dataset.labels_test, predict_fn(dataset.test[idx].reshape(1, -1))[0],
                                                    exp, anchors, pick_anchors_informations)

                """
                print('Anchor: %s' % (' AND '.join(exp.names())))
                print('Precision: %.2f' % exp.precision())
                print('Coverage: %.2f' % exp.coverage())
                """
                
                # Get test examples where the anchor applies
                fit_anchor = np.where(np.all(dataset.test[:, exp.features()] == dataset.test[idx][exp.features()], axis=1))[0]
                partial_coverage = (fit_anchor.shape[0] / float(dataset.test.shape[0]))
                partial_precision = (np.mean(predict_fn(dataset.test[fit_anchor]) == predict_fn(dataset.test[idx].reshape(1, -1))))
                #print('Anchor test coverage: %.2f' % partial_coverage)
                if coverage_tab != None:
                    mean_coverage += coverage_tab
                else:
                    mean_coverage += partial_coverage
                if precision_tab != None:
                    mean_precision += precision_tab
                else:
                    mean_precision += partial_precision
                #print('Anchor test precision: %.2f' % partial_precision)
                mean_size += len(exp.features())
                
            end = time.time()
            mean_coverage = mean_coverage / time_experiment
            mean_precision = mean_precision / time_experiment
            mean_size = mean_size / time_experiment
            model_time.append(end - start)
            mean_model_coverage.append(mean_coverage)
            mean_model_precision.append(mean_precision)
            mean_size_explanation.append(mean_size)
            features = []
            for feature in explainer.disc.bins_size:
                features.append(feature)
                model_size_discretization.append(explainer.disc.bins_size[feature])
            if feature_discretize == []:
                for feature in features:
                    feature_discretize.append(feature)
            if 'generate' in dataset_name:
                graph_data_generate = generate(dataset.test[idx].reshape(1, -1)[0][0], dataset.test[idx].reshape(1, -1)[0][1], anchors, blackbox=c, 
                                                experiment=False, X= X, Y = Y)
                graph_data_generate.savefig(filename + "/graph_data_" + x[j] + ".png")
        mean_model_coverage_precision = []
        plt.show(block=False)
        plt.pause(1)
        plt.close('all')
        print("bins :", model_size_discretization)
        print("nb feature :", feature_discretize)
        for i in range(len(mean_model_coverage)):
            mean_model_coverage_precision.append(mean_model_coverage[i]*mean_model_precision[i])

        graph_coverage = baseGraph.BaseGraph(title="Results of discretization methods for coverage", y_label="coverage", 
                            model=bb_model, accuracy=sklearn.metrics.accuracy_score(dataset.labels_test, predict_fn(dataset.test)), 
                            dataset=dataset_name, threshold=threshold)
        graph_coverage.show_coverage(model=x, mean_coverage=mean_model_coverage)
        graph_precision = baseGraph.BaseGraph(title="Results of discretization methods for precision", y_label="precision", 
                            model=bb_model, accuracy=sklearn.metrics.accuracy_score(dataset.labels_test, predict_fn(dataset.test)), 
                            dataset=dataset_name, threshold=threshold)
        graph_precision.show_precision(model=x, mean_precision=mean_model_precision)
        graph_coverage_precision = baseGraph.BaseGraph(title="Results of discretization methods for coverage * precision", 
                            y_label="coverage * precision", model=bb_model, threshold=threshold, 
                            accuracy=sklearn.metrics.accuracy_score(dataset.labels_test, predict_fn(dataset.test)), dataset=dataset_name)
        graph_coverage_precision.show_coverage_precision(model=x, mean_coverage_precision=mean_model_coverage_precision)
        graph_time = baseGraph.BaseGraph(title="Results of time for discretization methods", y_label="time", 
                            model=bb_model, accuracy=sklearn.metrics.accuracy_score(dataset.labels_test, predict_fn(dataset.test)), 
                            dataset=dataset_name, threshold=threshold)
        graph_time.show_time(model=x, mean_time=model_time)
        graph_size = baseGraph.BaseGraph(title="Results of size for discretization methods", y_label="size", 
                            model=bb_model, accuracy=sklearn.metrics.accuracy_score(dataset.labels_test, predict_fn(dataset.test)), 
                            dataset=dataset_name, threshold=threshold)
        graph_size.show_size(model=x, mean_size=mean_size_explanation)
        for i in range(len(feature_discretize)):
            size_discretization_feature = []
            for j in range(len(model_size_discretization)):
                if (j + i) % len(feature_discretize) == 0:
                    size_discretization_feature.append(model_size_discretization[j])
            graph_size_discretization = baseGraph.BaseGraph(title="Results of size of the discretization for feature " + str(feature_discretize[i]), 
                            y_label="size of bins", model=bb_model, accuracy=sklearn.metrics.accuracy_score(dataset.labels_test, predict_fn(dataset.test)), 
                            dataset=dataset_name, threshold=threshold)
            graph_size_discretization.show_size_discretization(model=x, size_discretization=size_discretization_feature, 
                            feature_discretization=feature_discretize[i])
        graph_size.writer_in_csv(dataset_name=dataset_name, dataset_size=len(dataset.data), nb_feature=len(dataset.feature_names), bb_name=bb_model, 
                bbox_train=bbox_train, bbox_test=bbox_test, precision=mean_model_precision, coverage=mean_model_coverage, 
                coverage_precision=mean_model_coverage_precision, size=mean_size_explanation, time=model_time,
                time_experiment=time_experiment, x=x, threshold=threshold)
else:
    c=tree.DecisionTreeClassifier()
    c.fit(train, labels_train)
    filename = "graph/" + dataset_name + "/" + type(c).__name__ + "/" + str(threshold) + "/"
    black_box_labels = c.predict(train)
    i = 0
    explainer = anchor_tabular.AnchorTabularExplainer(dataset.class_names, dataset.feature_names, dataset.train, dataset.categorical_names,
                                                                black_box_labels=black_box_labels,
                                                                discretizer="kmeans", filename=filename)
    predict_fn = lambda x: c.predict(x)
    bbox_train = sklearn.metrics.accuracy_score(dataset.labels_train, predict_fn(dataset.train))
    bbox_test = sklearn.metrics.accuracy_score(dataset.labels_test, predict_fn(dataset.test))
    print('Train', sklearn.metrics.accuracy_score(dataset.labels_train, predict_fn(dataset.train)))
    print('Test', sklearn.metrics.accuracy_score(dataset.labels_test, predict_fn(dataset.test)))

    idx = i
    np.random.seed(1)
    print("instance: ", dataset.test[idx].reshape(1, -1)[0])
    print('Prediction: ', explainer.class_names[predict_fn(dataset.test[idx].reshape(1, -1))[0]])
    exp = explainer.explain_instance(dataset.test[idx], c.predict, threshold=threshold, 
                                    delta=0.1, tau=0.15, batch_size=100,
                                    max_anchor_size=None,
                                    stop_on_first=False,
                                    desired_label=None,
                                    beam_size=4)
    print('Anchor: %s' % (' AND '.join(exp.names())))
    print('Precision: %.2f' % exp.precision())
    print('Coverage: %.2f' % exp.coverage())    
    #graph_data_generate = generate(dataset.test[idx].reshape(1, -1)[0][0], dataset.test[idx].reshape(1, -1)[0][1], exp.names(), blackbox=c, 
    #                                            experiment=False, X= X, Y = Y)