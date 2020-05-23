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
import spacy
import sys
from sklearn.feature_extraction.text import CountVectorizer
from anchor import anchor_text


#-------------- Model definition ---------------#
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ['SPACY_WARNING_IGNORE'] = 'W008'

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc_txt = nn.Linear(768, 256)
        self.fc_classif = nn.Linear(256, 2)


    def forward(self, input_tensor):
        text = sentence_emb(input_tensor, bert)
        text = F.relu(self.fc_txt(text))
        text = self.fc_classif(text)
        # return F.log_softmax(x)
        return nn.Softmax(dim = 1)(text)

#-------------- Text embedding ---------------#
from transformers import BertTokenizer, BertModel

#Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# Load pre-trained model (weights)
bert = BertModel.from_pretrained('bert-base-uncased')
# Put the model in "evaluation" mode, meaning feed-forward operation.
bert.eval()

def sentence_emb(input_tensor, model):
    #marked_text = "[CLS] " + sentence + " [SEP]"
    # Create tensor containing os.environ['SPACY_WARNING_IGNORE'] = 'W008'tokenized sentence
    input_tensor = input_tensor.long()
    input_tensor = input_tensor.unsqueeze(0)
    # tokens = tokens.long()
    # attention_masks = attention_masks.long()
    with torch.no_grad():
        output = model(input_tensor[:,0,:], attention_mask = input_tensor[:,1,:])
    # The last hidden-state is the first element of the output tuple and we take the first value of the batch and the first token encoding ([CLS])
    return output[0][:,0]

#-------------- Data preparation ---------------#
import matplotlib.pyplot as plt
import pandas as pd

path = "/home/julien/Documents/stage/anchor/datasets/antoine/"
# Create an instance of our network
net = Net()
# Load weights
net.load_state_dict(torch.load(path + "output/models/textonly_2020_03_09-14_29_17.pth"))

def predict_antoine(text):
    text = text[:512]
    res_tokenizer = tokenizer.encode_plus(text, pad_to_max_length=True)
    tensor_txt_inputs = torch.Tensor(res_tokenizer['input_ids']) # transform to torch tensor
    tensor_attention_masks = torch.Tensor(res_tokenizer['attention_mask'])
    input_tensor = torch.stack([tensor_txt_inputs, tensor_attention_masks], dim = 0)
    output = net(input_tensor)
    predicted = torch.argmax(output.data, 1)
    return predicted

def get_proba(texts):
    res = np.empty([len(texts), 2])
    i = 0
    with torch.no_grad():
        for text in texts:
            res_tokenizer = tokenizer.encode_plus(text, pad_to_max_length=True)
            tensor_txt_inputs = torch.Tensor(res_tokenizer['input_ids']) # transform to torch tensor
            tensor_attention_masks = torch.Tensor(res_tokenizer['attention_mask'])
            input_tensor = torch.stack([tensor_txt_inputs, tensor_attention_masks], dim = 1)
            output = net(input_tensor)
            #predicted = torch.argmax(output.data, 1)
            res[i] = output.numpy()[0]
            i = ++i
    print(res)
    return res
            
    
nlp = spacy.load('/home/julien/Documents/stage/anchor/datasets/en_core_web_lg-2.2.5/en_core_web_lg/en_core_web_lg-2.2.5')
#nlp = spacy.load('/udd/jdelauna/Documents/anchor/datasets/fr_core_news_sm-2.2.5/fr_core_news_sm/fr_core_news_sm-2.2.5')

explainer = anchor_text.AnchorText(nlp, ['true news', 'fake news'], use_unk_distribution=True)

np.random.seed(1)
#text = 'This is a good book . I will learn a lot in this book . Maybe one day I will be an expert in such a domain .'
#text = "DUBAI, April 19 (Reuters) - Singapore-based Lloyd’s of London insurer, Global Specialty Brokers (GSB), said on Monday.It had suspended flights to Hong Kong from Qatar Airways."
#text = "DUBAI , April 19 (Reuters) - Singapore - based Lloyd of London insurer , Global Specialty Brokers (GSB) , said on Monday . China destroys France ."

#text = "WARSAW (Reuters) - Three new cases of the new coronavirus have been diagnosed in Poland - one man in a critical condition, and two suspected cases - the Health Ministry said on Friday.In January, a 35-year-old Iraqi man died in Poland after suffering severe respiratory infection, possibly caused by the novel coronavirus, also known as NCoV.Authorities are still trying to determine the extent of any relationship between the man, who was admitted to hospital with respiratory illness and died last month, and other possible victims in the country.As in other parts of the world, some foreign universities and medical schools have cancelled conferences or seminars due to NCoV cases in different countries, as has Poland’s health ministry.There have been no reported cases of the novel coronavirus in Poland.NCoV is a virus from the same family as the SARS virus which killed around 800 people worldwide in 2002 and 2003. Scientists believe it may have circulated before the world had developed the ability to detect it through human-to-human transmission."
text = "DUBAI, April 19 (Reuters) - Singapore-based Lloyd’s of London insurer, Global Specialty Brokers (GSB), said on Monday it had suspended flights to Hong Kong from Qatar Airways after low demand since a state crackdown on fundraising by activist investors.“Since the implementation of Hong Kong regulations, low demand for our services from Qatar Airways has led us to suspend its operations,” GSB said in a statement.Hong Kong has tightened regulations on shareholder activism, including curbs on companies bringing in external directors, and launched a review of such matters after a wave of activist campaigns last year.The rules also require companies to publish a list of companies that have been conducting a financial or administrative audit for up to three years, citing concerns over the preparation of the financial statements of such companies.GSB offered alternative services to Qatar Airways, such as trading claims, claims management and reinsurance, through Lloyd’s of London in Hong Kong. It did not reveal how many passengers it had earned from services for Qatar Airways.GSB had offered “well over” 20 flights per month to Hong Kong from Doha since 2015, but with limited demand, the insurer said.The Qatar Airways spokesman said its policy is to not comment on media reports."
text = text.encode('utf-8')
text = str(text)

#text = "We are going to extend this new method and prevent China from attacking France ."
#text = "This is a good book ."
pred = explainer.class_names[predict_antoine(text)]
alternative =  explainer.class_names[1 - predict_antoine(text)]
print('Prediction: %s' % pred)
exp = explainer.explain_instance(text, predict_antoine, threshold=0.95, use_proba=True)
print('Anchor: %s' % (' AND '.join(exp.names())))
print('Precision: %.2f' % exp.precision())
print()
print('Examples where anchor applies and model predicts %s:' % pred)
print()
print('\n'.join([x[0] for x in exp.examples(only_same_prediction=True)]))
print()
print('Examples where anchor applies and model predicts %s:' % alternative)
print()
print('\n'.join([x[0] for x in exp.examples(only_different_prediction=True)]))
