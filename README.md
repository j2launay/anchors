# anchors
This repository has code for the internship "When are anchors based explanations not a good idea?"  

An anchor explanation is a rule that sufficiently “anchors” the
prediction locally – such that changes to the rest of the feature
values of the instance do not matter. In other words, for instances on which the anchor holds, the prediction is (almost)
always the same.

At the moment, we support explaining individual predictions for text classifiers or classifiers that act on tables (numpy arrays of numerical or categorical data).

The anchor method is able to explain any black box classifier, with two or more classes. All we require is that the classifier implements a function that takes in raw text or a numpy array and outputs a prediction (integer)

During this internship we aim at improving the anchor method of Marco Tulio Ribeiro by extending the way it discretizes tabular data and adding the possibility to return pertinent negatif for textual data.

## Installation
The Anchor package is on pypi. Simply clone the repository and run:

    python setup.py install

If you want to use `AnchorTextExplainer`, you have to run the following:

    python -m spacy download en_core_web_lg

#### Examples
See notebooks folder for tutorials. Note that from version 0.0.1.0, it only works on python 3.

- [Tabular data](https://github.com/marcotcr/anchor/blob/master/notebooks/Anchor%20on%20tabular%20data.ipynb)
- [Text data](https://github.com/marcotcr/anchor/blob/master/notebooks/Anchor%20for%20text.ipynb) 
