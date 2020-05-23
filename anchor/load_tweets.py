import csv
import numpy as np
import pandas as pd
import emoji
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score, f1_score, jaccard_score, classification_report
from sklearn.feature_extraction.text import CountVectorizer


def load_tweets():
    X = []
    with open("/home/julien/Documents/stage/anchor/datasets/premoji/data/train/crawler/data/tweets.txt.text", newline='', encoding='utf8') as file_data:
        for row in file_data:
            X.append(row.replace("\n",""))
                
    # Training emojis recovery
    y = []
    with open("/home/julien/Documents/stage/anchor/datasets/premoji/data/train/crawler/data/tweets.txt.labels", newline='', encoding='utf8') as file_data:
        for row in file_data:
            y.append(row.replace("\n",""))
    X = np.array(X)
    y = np.array(y)

    """
    # Testing tweets recovery
    test=[]
    with open("/home/julien/Documents/stage/anchor/datasets/premoji/data/test/us_test.text", newline='', encoding='utf8') as test_data:
        file = test_data.readlines()
        for row in file:
            test.append(row.replace("\n",""))
    test = np.asarray(test)

    # Testing emojis recovery
    test_label = []
    with open("/home/julien/Documents/stage/anchor/datasets/premoji/data/test/us_test.labels", newline='', encoding='utf8') as test_data_label:
        file = test_data_label.readlines()
        for row in file:
            test_label.append(row.replace("\n",""))
    test_label = np.asarray(test_label)
    test_label = test_label.reshape(-1,1)
    return X, y, test, test_label
    """
    return X, y

def load_tweets_es():
    X = []
    with open("/home/julien/Documents/stage/anchor/datasets/premoji/data/trial/es_trial.text", newline='', encoding='utf8') as file_data:
        for row in file_data:
            X.append(row.replace("\n",""))
                
    # Training emojis recovery
    y = []
    with open("/home/julien/Documents/stage/anchor/datasets/premoji/data/trial/es_trial.labels", newline='', encoding='utf8') as file_data:
        for row in file_data:
            y.append(row.replace("\n",""))
    X = np.array(X)
    y = np.array(y)

    """
    # Testing tweets recovery
    test=[]
    with open("/home/julien/Documents/stage/anchor/datasets/premoji/data/test/es_test.text", newline='', encoding='utf8') as test_data:
        file = test_data.readlines()
        for row in file:
            test.append(row.replace("\n",""))
    test = np.asarray(test)

    # Testing emojis recovery
    test_label = []
    with open("/home/julien/Documents/stage/anchor/datasets/premoji/data/test/es_test.labels", newline='', encoding='utf8') as test_data_label:
        file = test_data_label.readlines()
        for row in file:
            test_label.append(row.replace("\n",""))
    test_label = np.asarray(test_label)
    test_label = test_label.reshape(-1,1)
    return X, y, test, test_label
    """
    return X, y

def transform_emoji():
    list_emoji = [emoji.emojize(':red_heart:'), emoji.emojize(':smiling_face_with_heart-eyes:'), emoji.emojize(':face_with_tears_of_joy:'), 
                emoji.emojize(':two_hearts:'), emoji.emojize(':fire:'), emoji.emojize(':smiling_face_with_smiling_eyes:'), 
                emoji.emojize(':smiling_face_with_sunglasses:'), emoji.emojize(':sparkles:'), emoji.emojize(':blue_heart:'), 
                emoji.emojize(':face_blowing_a_kiss:'), emoji.emojize(':camera:'), emoji.emojize(":United_States:"),
                emoji.emojize(':sun:'), emoji.emojize(':purple_heart:'), emoji.emojize(':winking_face:'), emoji.emojize(':hundred_points:'), 
                emoji.emojize(':beaming_face_with_smiling_eyes:'), emoji.emojize(':Christmas_tree:'), emoji.emojize(':camera_with_flash:'), 
                emoji.emojize(':winking_face_with_tongue:')]
    return list_emoji

def transform_emoji_es():
    list_emoji = [emoji.emojize(':red_heart:'), emoji.emojize(':smiling_face_with_heart-eyes:'), emoji.emojize(':face_with_tears_of_joy:'), 
                emoji.emojize(':two_hearts:'), emoji.emojize(':fire:'), emoji.emojize(':smiling_face_with_smiling_eyes:'), 
                emoji.emojize(':smiling_face_with_sunglasses:'), emoji.emojize(':sparkles:'), emoji.emojize(':blue_heart:'), 
                emoji.emojize(':face_blowing_a_kiss:'), emoji.emojize(':camera:'), emoji.emojize(":United_States:"),
                emoji.emojize(':sun:'), emoji.emojize(':purple_heart:'), emoji.emojize(':winking_face:'), emoji.emojize(':hundred_points:'), 
                emoji.emojize(':beaming_face_with_smiling_eyes:'), emoji.emojize(':Christmas_tree:'), emoji.emojize(':camera_with_flash:'), 
                emoji.emojize(':winking_face_with_tongue:')]
    return list_emoji
    