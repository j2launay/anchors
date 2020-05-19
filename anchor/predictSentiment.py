import pandas as pd
import pickle
import os
import sklearn

# Load Machine Learning Models
print("Load Machine Learning Models")
dirname = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(dirname,'models','svm.pkl'), 'rb') as f1:
    svc = pickle.load(f1)
with open(os.path.join(dirname,'models','vect.pkl'), 'rb') as f2:
    vect = pickle.load(f2)

"""
predicts the sentiment expressed by a tweet '+', '-' or '='
"""

def transform_output(signe):
    if signe == "+":
        return 0
    else:
        return 1
    """
    elif signe == "-":
        return 1
    else:
        return 2
    """
    
def predictFrench(tweet):
    instance = pd.DataFrame(vect.transform(tweet).todense(),columns=vect.get_feature_names())
    return transform_output(svc.predict(instance)[0])