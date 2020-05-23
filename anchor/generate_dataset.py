#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 17:54:12 2020

@author: jdelauna
"""

from sklearn.datasets import make_blobs, make_moons, make_circles
from matplotlib import pyplot
from pandas import DataFrame
import numpy as np
import re
import csv
from scipy.special import expit
from sklearn import linear_model

# generate 2d classification dataset
def generate(target_instance_x, target_instance_y, anchors, blackbox, n_samples=100, centers=2, n_features=2, experiment=True, X=None, Y=None):
    """
    Function to generate a 2 dimensional dataset
    Function plots the samples separated by different colors and the target instance represented by a star
    A square that takes the form of the anchor is drawn upon these graphic
    args:
        target_instance_x : Value of the target instance for the first feature
        target_instance_y : Value of the target instance for the second feature
        anchors : anchors return by the interpretable method
        n_samples : number of isntances to generate
        X : value for both features for each element of the sample
        Y : value of the label for each element of the sample
    """
    
    def pick_anchors_informations(anchors=anchors, x_min=-10, width=20, y_min=-10, height=20, compute=False):
        """
        Function to store information about the anchors and return the position and size to draw the anchors
        Anchors is of the form : "2 < x <= 7, -5 > y" or any rule
        """
        regex = re.compile(r"([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")
        if len(anchors) == 0:
            return x_min, y_min, width, height
        elif len(anchors) == 1:
            if "x" in anchors[0]:
                x_bounds = regex.findall(anchors[0])
                x_min = min(float(x) for x in x_bounds)
                if len(x_bounds) == 1:
                    width = (-width if "<" in anchors[0] else width)
                else:
                    width = max(float(x) for x in x_bounds) - min(float(y) for y in x_bounds)
            else:
                y_bounds = regex.findall(anchors[0])
                y_min = min(float(y) for y in y_bounds)
                if len(y_bounds) == 1:
                    height = (-height if "<" in anchors[0] else height)
                else:
                    height = max(float(y) for y in y_bounds) - min(float(x) for x in y_bounds)

        else:
            if "x" in anchors[0]:
                x_bounds = regex.findall(anchors[0])
                x_min = min(float(x) for x in x_bounds)
                if len(x_bounds) == 1:
                    width = (-width if "<" in anchors[0] else width)
                else:
                    width = max(float(x) for x in x_bounds) - min(float(y) for y in x_bounds)
                y_bounds = regex.findall(anchors[1])
                y_min = min(float(y) for y in y_bounds)
                if len(y_bounds) == 1:
                    height = (-height if "<" in anchors[1] else height)
                else:
                    height = max(float(y) for y in y_bounds) - min(float(x) for x in y_bounds)
            else:
                y_bounds = regex.findall(anchors[0])
                y_min = min(float(y) for y in y_bounds)
                if len(y_bounds) == 1:
                    height = (-height if "<" in anchors[0] else height)
                else:
                    height = max(float(y) for y in y_bounds) - min(float(x) for x in y_bounds)
                x_bounds = regex.findall(anchors[1])
                x_min = min(float(x) for x in x_bounds)
                if len(x_bounds) == 1:
                    width = (-width if "<" in anchors[1] else width)
                    print(width)
                else:
                    width = max(float(x) for x in x_bounds) - min(float(y) for y in x_bounds)
        return x_min, y_min, width, height

    if not experiment:
        #X, Y = make_blobs(n_samples=n_samples, centers=centers, n_features=n_features)
        # scatter plot, dots colored by class value
        df = DataFrame(dict(x=X[:, 0], y=X[:, 1], label=Y))
        # If you want a third color and have three classes
        colors = {0: 'red', 1: 'blue', 2: 'green'}
        fig, ax = pyplot.subplots()
        x_min = min([x[0] for x in X])
        x_max = max([x[0] for x in X])
        y_min = min([y[1] for y in X])
        y_max = max([y[1] for y in X])
        def draw_black_box(ax, X, Y, x_min, y_min, x_max, y_max, blackbox) :
            X_test = np.linspace(x_min-5, x_max+5, 100) 
            Y_test = np.linspace(y_min-5, y_max+5, 100)
            X_prime = np.transpose([np.tile(X_test, len(Y_test)), 
                                np.repeat(Y_test, len(X_test))])

            X_test, Y_test = np.meshgrid(X_test, Y_test)
            if type(blackbox) == linear_model.LogisticRegression :
                loss = expit(np.matmul(X_prime, np.transpose(blackbox.coef_)) + blackbox.intercept_).ravel()
            else:
                loss = blackbox.predict(X_prime)    
            loss =  np.array(np.split(loss, 100))
            #ax.plot(newx, newy,  color='gray', linewidth=3)
            ax.pcolormesh(X_test, Y_test, loss, cmap='gray')
            ax.axis([x_min-5, x_max+5, y_min-5, y_max+5])
            return blackbox
            #loss_parts = np.split(loss, 2)
            #ax.plot(loss_parts[0], loss_parts[1], color='gray', linewidth=3)

        ax = pyplot.subplot(111)
        text = "Anchors: " + str(anchors)
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        # place a text box in upper left in axes coords
        ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=props)
        
        draw_black_box(ax, X, Y, x_min, y_min, x_max, y_max, blackbox)

        grouped = df.groupby('label')
        for key, group in grouped:
            group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])

        # Function to draw the target instance (Modify the markersize to modify the size of the star)
        ax.plot(target_instance_x, target_instance_y, marker="*", color='purple', markersize=20)

        def draw_rectangle(ax, x_min_anchors, y_min_anchors, width, height):
            # Draw the rectangle upon the graphics
            if y_min_anchors != y_min-4:
                ax.plot([x_min_anchors, x_min_anchors + width], [y_min_anchors, y_min_anchors],'r-')
                ax.plot([x_min_anchors + width, x_min_anchors], [y_min_anchors + height, y_min_anchors + height], 'y-')
            else:
                ax.plot([x_min_anchors, x_min_anchors + width], [y_min_anchors, y_min_anchors],'r-', linestyle='dashed')
                ax.plot([x_min_anchors + width, x_min_anchors], [y_min_anchors + height, y_min_anchors + height], 'y-', linestyle='dashed')
            if x_min_anchors != x_min-4 :
                ax.plot([x_min_anchors, x_min_anchors], [y_min_anchors, y_min_anchors + height], 'g-')
                ax.plot([x_min_anchors + width, x_min_anchors + width], [y_min_anchors, y_min_anchors + height], 'b-')
            else:
                ax.plot([x_min_anchors, x_min_anchors], [y_min_anchors, y_min_anchors + height], 'g-', linestyle='dashed')
                ax.plot([x_min_anchors + width, x_min_anchors + width], [y_min_anchors, y_min_anchors + height], 'b-', linestyle='dashed')

        x_min_anchors, y_min_anchors, width, height = pick_anchors_informations(x_min=x_min-4, y_min=y_min-4, width=x_max-x_min+8, height=y_max-y_min+8)
        draw_rectangle(ax, x_min_anchors, y_min_anchors, width, height)
        #pyplot.savefig("my_graph.png")
        #pyplot.show()
        return pyplot
    return X, Y, pick_anchors_informations

def print_example_text(text, pred, alternative, exp, file_csv):
    with open(file_csv, 'w', newline='') as f:
        file_csv = csv.writer(f)
        file_csv.writerow([str('text: ' + text)])
        file_csv.writerow([str('Prediction: ' + pred)])
        file_csv.writerow([str('Anchor: ' + str(exp.names()))])
        file_csv.writerow([str('Precision: ' + str(np.round(exp.precision())))])
        file_csv.writerow([str('Coverage: ' + str(np.round(exp.coverage())))])
        file_csv.writerow([])
        file_csv.writerow([str('Examples where anchor applies and model predicts :' + pred)])
        file_csv.writerow([])
        file_csv.writerow([str('\n'.join([x[0] for x in exp.examples(only_same_prediction=True)]))])
        file_csv.writerow([])
        file_csv.writerow([str('Examples where anchor applies and model predicts :' + alternative)])
        file_csv.writerow([])
        file_csv.writerow([str('\n'.join([x[0] for x in exp.examples(only_different_prediction=True)]))])

def compare_value_to_anchor(min, range, z):
    if range == -20:
        return bool(z < min)
    elif range == 20:
        return bool(z > min)
    else:
        return bool(z >= min and z <= min + range) 

def compare_values_to_anchor(xmin, ymin, x_range, y_range, v, w):
    partial_boolean_tab = []
    partial_boolean_tab.append(compare_value_to_anchor(xmin, x_range, v))
    partial_boolean_tab.append(compare_value_to_anchor(ymin, y_range, w))
    if False in partial_boolean_tab:
        return False
    else:
        return True

def compute_coverage_tab(dataset, exp, anchors, pick_anchors_informations):
    coverage_tab = []
    x_min, y_min, width, height = pick_anchors_informations(anchors=anchors, compute=True)
    if len(exp.features()) == 1:
        if exp.features()[0] == 0:
            for element in dataset.test[:, exp.features()]:
                coverage_tab.append(compare_value_to_anchor(x_min, width, element))
        else:
            for element in dataset.test[:, exp.features()]:
                coverage_tab.append(compare_value_to_anchor(y_min, height, element))
    else:
        for element in dataset.test[:, exp.features()]:
            coverage_tab.append(compare_values_to_anchor(x_min, y_min, width, height, element[0], element[1]))
    return coverage_tab

def compute_coverage(dataset, exp, anchors, pick_anchors_informations):
    coverage_tab = compute_coverage_tab(dataset, exp, anchors, pick_anchors_informations)
    coverage = 0
    for compute, instance in enumerate(coverage_tab):
        if instance == True:
            coverage = coverage + 1
    return coverage/compute

def compute_precision(dataset, labels, target_labels, exp, anchors, pick_anchors_informations):
    coverage_tab = compute_coverage_tab(dataset, exp, anchors, pick_anchors_informations)
    coverage = 0
    precision = 0
    for compute, instance in enumerate(coverage_tab):
        if instance == True:
            if labels[compute] == target_labels:
                precision = precision + 1
            coverage = coverage + 1
    if coverage != 0:
        return precision/coverage
    return None 

#generate(0, 0, ("-2 < x <= 2", "-5 < y <= 5"))
