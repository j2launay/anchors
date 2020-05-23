#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 17:37:47 2020

@author: jdelauna
"""
fichier = open("../datasets/titanic/titanic.data", "w")
with open("../datasets/titanic/new_test_train.csv") as f:
    boolean = True
    for line in f:      
        if boolean:
            boolean = False
        else:
            fichier.write(line.replace(' ', '').replace(',', ', '))
    
        
fichier.close()
