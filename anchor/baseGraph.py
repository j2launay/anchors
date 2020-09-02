#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import csv
from matplotlib.font_manager import FontProperties

class BaseGraph:

    def __init__(self, title, y_label, dataset, model, accuracy, threshold, x_label="model", width=0.33):
        """
        Initialise a base for a graph
        args:
            title: title of the graph
            y_label: label of the graph on axe y
            dataset: name of the dataset
            model: name of the black box model
            accuracy: accuracy of the black box model
            threshold: minimum threshold for the anchor
        """
        self.title = title 
        self.x_label = x_label
        self.y_label = y_label
        self.width = width
        self.dataset = dataset
        self.model = model
        self.threshold = str(threshold)
        self.color = (0.65098041296005249, 0.80784314870834351, 0.89019608497619629, 1.0)
        # initialise a filename inside "graph" to store the graph generated
        self.filename = "graph/" + self.dataset + "/" + self.model + "/" + self.threshold
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)
        plt.title(self.title)
        plt.grid(True)
        fig = plt.gcf()
        fig.set_size_inches(15, 10)
        self.fontP = FontProperties()
        self.fontP.set_size('small')
        self.accuracy = accuracy

    def add_legend_bottom(self, title, mean, model):
        """
        Generate a legend at the bottom of the graph
        args:
            title: type of measure (i.e:'coverage = ' or 'precision = ')
            mean: mean results for each discretization model (i.e: (0.7, 0.5) for a coverage of 0.7 for MDLP and 0.5 for Decile)
            model: Different discretization model that are measure
        """
        ax = plt.subplot(111)
        text = "Accuracy of the black box: " + str(self.accuracy)
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=props)
        for i in range(len(model)):
            text = (model[i] + title + str(np.round(mean[i], 2)))
            line, = ax.plot(model[i], i, label=text)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height*0.9])

        # Put a legend below current axis
        ax.legend(loc='upper center', bbox_to_anchor=(0.8, -0.05), fancybox=True, shadow=True, prop=self.fontP)

    """
    For following function, model represents the list of discretization methods that are used 
    mean_'things' corresponds to the mean 'things' results for each discretization methods
    """

    def show_coverage(self, model, mean_coverage):
        plt.bar(model, mean_coverage, self.width, color=self.color)
        axes = plt.axes()
        axes.set_ylim([0, 1])
        self.add_legend_bottom(title=" coverage = ", mean=mean_coverage, model=model)
        plt.savefig(self.filename + "/graph_coverage.png")
        plt.show(block=False)
        plt.pause(1)
        plt.close('all')

    def show_precision(self, model, mean_precision):
        plt.bar(model, mean_precision, self.width, color=self.color)
        axes = plt.axes()
        axes.set_ylim([0, 1.2])
        self.add_legend_bottom(title=" precision = ", mean=mean_precision, model=model)
        plt.savefig(self.filename + "/graph_precision.png")
        plt.show(block=False)
        plt.pause(1)
        plt.close('all')

    def show_coverage_precision(self, model, mean_coverage_precision):
        plt.bar(model, mean_coverage_precision, self.width, color=self.color)
        axes = plt.axes()
        axes.set_ylim([0, 1])
        self.add_legend_bottom(title=": 2/(1/coverage + 1/precision) = ", mean=mean_coverage_precision, model=model)
        plt.savefig(self.filename + "/graph_coverage_precision.png")
        plt.show(block=False)
        plt.pause(1)
        plt.close('all')

    def show_time(self, model, mean_time):
        plt.bar(model, mean_time, self.width, color=self.color)
        self.add_legend_bottom(title=" time = ", mean=mean_time, model=model)
        plt.savefig(self.filename + "/graph_time.png")
        plt.show(block=False)
        plt.pause(1)
        plt.close('all')

    def show_size(self, model, mean_size):
        plt.bar(model, mean_size, self.width, color=self.color)
        self.add_legend_bottom(title=" size = ", mean=mean_size, model=model)
        plt.savefig(self.filename + "/graph_size.png")
        plt.show(block=False)
        plt.pause(1)
        plt.close('all')
    
    def show_size_discretization(self, model, size_discretization, feature_discretization):
        """
        Function that shows the number of bins for a feature created by a discretization method.
        """
        plt.bar(model, size_discretization, self.width, color=self.color)
        self.add_legend_bottom(title=" number of bins = ", mean=size_discretization, model=model)
        plt.savefig(self.filename + "/" + feature_discretization + "_graph_number_bins.png")
        plt.show(block=False)
        plt.close('all')

    def output_csv(self, writer, mean_models, models_discretization):
        for mean, model in zip(mean_models, models_discretization):
            writer.writerow(["", model + ": ", str(np.round(mean, 2))])
    
    def writer_in_csv(self, dataset_name, dataset_size, bb_name, bbox_train, bbox_test, precision,
                    coverage, coverage_precision, size, time, time_experiment, threshold, x, nb_feature=None):
        """
        Write in a csv file all the information that are necessary to reproduce the experiment and compute the results easily
        args:
            dataset_name: self explain
            dataset_size: self explain
            bb_name: name of the black box model
            bbox_train: Accuracy score of the black box on the training set 
            bbox_test: Accuracy score of the black box on the testing set
            precision: Mean precision computed for all discretization methods
            coverage: Mean coverage computed for all discretization methods
            coverage_precision: Mean F1 score computed for all discretization methods
            size: Mean size of the explanation generated by anchor for all discretization methods
            time: Mean time to generate all the explanation by discretization method
            time_experiment: Number of explanations generated by each discretization method
            threshold: Minimum precision threshold for the anchor
            x: What is evaluated (i.e: [MDLP, kmeans, quartile] or [Mask words, Replace words, Pertinent Negatif])
            nb_feature: Number of feature in the dataset
        """
        with open(self.filename + "/tab_resume.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['black box type', 'black box train accuracy', 'black box test accuracy', 'anchor precision threshold'])
            writer.writerow([bb_name, bbox_train, bbox_test, threshold])
            writer.writerow([])
            writer.writerow(['anchors precision'])
            writer.writerow([self.output_csv(writer, precision, x)])
            writer.writerow(['anchors coverage'])
            writer.writerow([self.output_csv(writer, coverage, x)])
            writer.writerow(['anchors prec*coverage'])  
            writer.writerow([self.output_csv(writer, coverage_precision, x)])
            writer.writerow(['anchors size'])
            writer.writerow([self.output_csv(writer, size, x)])
            writer.writerow(['time to compute anchors'])
            writer.writerow([self.output_csv(writer, time, x)])
            if nb_feature == None:
                writer.writerow(['dataset name', 'dataset size', 'Number of instances running'])
                writer.writerow([dataset_name, dataset_size, time_experiment])
            else:
                writer.writerow(['dataset name', 'dataset size', 'dataset nb feature', 'Number of instances running'])
                writer.writerow([dataset_name, dataset_size, nb_feature, time_experiment])