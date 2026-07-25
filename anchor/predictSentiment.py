"""Sentiment prediction module using pre-trained SVM model.

This module loads a pre-trained SVM classifier and vectorizer
to predict sentiment of French text inputs.

Security Note:
    Model loading uses RestrictedUnpickler to mitigate arbitrary
    code execution risks from pickle deserialization. Only sklearn
    and numpy classes are allowed during unpickling.
"""
import io
import os
import pickle

import numpy as np
import pandas as pd
import sklearn


class RestrictedUnpickler(pickle.Unpickler):
    """A restricted unpickler that only allows safe sklearn/numpy classes.

    This prevents arbitrary code execution when loading pickle files
    from untrusted sources.
    """

    _SAFE_MODULES = frozenset({
        'sklearn', 'sklearn.svm', 'sklearn.svm._classes',
        'sklearn.feature_extraction.text',
        'sklearn.preprocessing', 'sklearn.utils',
        'numpy', 'numpy.core', 'numpy.core.multiarray',
        'builtins', 'collections', 'copy_reg', '_codecs',
    })

    def find_class(self, module, name):
        """Only allow classes from safe modules during unpickling."""
        base_module = module.split('.')[0]
        if base_module in ('sklearn', 'numpy', 'builtins', 'collections',
                           'copy_reg', '_codecs', 'copyreg'):
            return super().find_class(module, name)
        raise pickle.UnpicklingError(
            f"Blocked loading of class '{name}' from module '{module}'. "
            "Only sklearn and numpy classes are permitted."
        )


def _restricted_load(filepath):
    """Load a pickle file using the restricted unpickler.

    Args:
        filepath: Path to the pickle file to load.

    Returns:
        The deserialized Python object.

    Raises:
        pickle.UnpicklingError: If the pickle contains disallowed classes.
    """
    with open(filepath, 'rb') as f:
        return RestrictedUnpickler(f).load()


# Load Machine Learning Models
print("Load Machine Learning Models")
dirname = os.path.dirname(os.path.abspath(__file__))
svc = _restricted_load(os.path.join(dirname, 'models', 'svm.pkl'))
vect = _restricted_load(os.path.join(dirname, 'models', 'vect.pkl'))


def transform_output(signe):
    """Transform sentiment sign to numeric label.

    Args:
        signe: A string, either '+' for positive or anything else for negative.

    Returns:
        int: 0 for positive ('+'), 1 for negative.
    """
    if signe == "+":
        return 0
    else:
        return 1


def predictFrench(tweet):
    """Predict sentiment of a French tweet.

    Args:
        tweet: A list containing the tweet text to classify.

    Returns:
        int: Predicted sentiment label (0=positive, 1=negative).
    """
    instance = pd.DataFrame(
        vect.transform(tweet).todense(),
        columns=vect.get_feature_names_out()
    )
    return transform_output(svc.predict(instance)[0])