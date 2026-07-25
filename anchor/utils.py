"""Utility functions for the Anchors explainability library.

This module provides helper functions for data loading, text perturbation,
and pertinent negative generation used by the anchor explanation methods.
"""
# from __future__ import print_function
import copy
import os
import sys

import numpy as np
import sklearn
import lime
import lime.lime_tabular

os.environ['SPACY_WARNING_IGNORE'] = 'W008'
import co_occurrence_matrix as co_occ  # noqa: E402

if (sys.version_info > (3, 0)):
    def unicode(s, errors=None):
        return s  # str(s)


class Bunch(object):
    """A simple container object that exposes dict keys as attributes.

    Args:
        adict: A dictionary whose keys will become object attributes.
    """
    def __init__(self, adict):
        self.__dict__.update(adict)


def map_array_values(array, value_map):
    """Map values in an array according to a value mapping dictionary.

    Args:
        array: numpy array to transform.
        value_map: dict mapping source values to target values.

    Returns:
        numpy array with mapped values.
    """
    ret = array.copy()
    for src, target in value_map.items():
        ret[ret == src] = target
    return ret


def replace_binary_values(array, values):
    """Replace binary '0'/'1' string values with descriptive labels.

    Args:
        array: numpy array containing '0' and '1' string values.
        values: list of two strings [label_for_0, label_for_1].

    Returns:
        numpy array with replaced values.
    """
    return map_array_values(array, {'0': values[0], '1': values[1]})


def load_dataset(dataset_name, balance=False, discretize=True, dataset_folder='./'):
    """Load a named dataset for anchor explanation experiments.

    Supports the following datasets: 'adult', 'titanic', 'generate',
    'diabetes', 'default', 'recidivism', 'lending'.

    Args:
        dataset_name: Name of the dataset to load.
        balance: If True, balance class distribution by undersampling.
        discretize: If True, discretize continuous features.
        dataset_folder: Path to the folder containing dataset files.

    Returns:
        A Bunch object containing train/test splits, feature names,
        categorical feature information, and class names.

    Raises:
        NameError: If dataset_name is not recognized (implicit via missing return).
    """
    if dataset_name == 'adult':
        feature_names = ["Age", "Workclass", "fnlwgt", "Education",
                         "Education-Num", "Marital Status", "Occupation",
                         "Relationship", "Race", "Sex", "Capital Gain",
                         "Capital Loss", "Hours per week", "Country", 'Income']
        features_to_use = [0, 1, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        categorical_features = [1, 3, 5, 6, 7, 8, 9, 10, 11, 13]
        education_map = {
            '10th': 'Dropout', '11th': 'Dropout', '12th': 'Dropout', '1st-4th':
            'Dropout', '5th-6th': 'Dropout', '7th-8th': 'Dropout', '9th':
            'Dropout', 'Preschool': 'Dropout', 'HS-grad': 'High School grad',
            'Some-college': 'High School grad', 'Masters': 'Masters',
            'Prof-school': 'Prof-School', 'Assoc-acdm': 'Associates',
            'Assoc-voc': 'Associates',
        }
        occupation_map = {
            "Adm-clerical": "Admin", "Armed-Forces": "Military",
            "Craft-repair": "Blue-Collar", "Exec-managerial": "White-Collar",
            "Farming-fishing": "Blue-Collar", "Handlers-cleaners":
            "Blue-Collar", "Machine-op-inspct": "Blue-Collar", "Other-service":
            "Service", "Priv-house-serv": "Service", "Prof-specialty":
            "Professional", "Protective-serv": "Other", "Sales":
            "Sales", "Tech-support": "Other", "Transport-moving":
            "Blue-Collar",
        }
        country_map = {
            'Cambodia': 'SE-Asia', 'Canada': 'British-Commonwealth', 'China':
            'China', 'Columbia': 'South-America', 'Cuba': 'Other',
            'Dominican-Republic': 'Latin-America', 'Ecuador': 'South-America',
            'El-Salvador': 'South-America', 'England': 'British-Commonwealth',
            'France': 'Euro_1', 'Germany': 'Euro_1', 'Greece': 'Euro_2',
            'Guatemala': 'Latin-America', 'Haiti': 'Latin-America',
            'Holand-Netherlands': 'Euro_1', 'Honduras': 'Latin-America',
            'Hong': 'China', 'Hungary': 'Euro_2', 'India':
            'British-Commonwealth', 'Iran': 'Other', 'Ireland':
            'British-Commonwealth', 'Italy': 'Euro_1', 'Jamaica':
            'Latin-America', 'Japan': 'Other', 'Laos': 'SE-Asia', 'Mexico':
            'Latin-America', 'Nicaragua': 'Latin-America',
            'Outlying-US(Guam-USVI-etc)': 'Latin-America', 'Peru':
            'South-America', 'Philippines': 'SE-Asia', 'Poland': 'Euro_2',
            'Portugal': 'Euro_2', 'Puerto-Rico': 'Latin-America', 'Scotland':
            'British-Commonwealth', 'South': 'Euro_2', 'Taiwan': 'China',
            'Thailand': 'SE-Asia', 'Trinadad&Tobago': 'Latin-America',
            'United-States': 'United-States', 'Vietnam': 'SE-Asia'
        }
        married_map = {
            'Never-married': 'Never-Married', 'Married-AF-spouse': 'Married',
            'Married-civ-spouse': 'Married', 'Married-spouse-absent':
            'Separated', 'Separated': 'Separated', 'Divorced':
            'Separated', 'Widowed': 'Widowed'
        }
        label_map = {'<=50K': 'Less than $50,000', '>50K': 'More than $50,000'}

        def cap_gains_fn(x):
            x = x.astype(float)
            d = np.digitize(x, [0, np.median(x[x > 0]), float('inf')],
                            right=True).astype('|S128')
            return map_array_values(d, {'0': 'None', '1': 'Low', '2': 'High'})

        transformations = {
            3: lambda x: map_array_values(x, education_map),
            5: lambda x: map_array_values(x, married_map),
            6: lambda x: map_array_values(x, occupation_map),
            10: cap_gains_fn,
            11: cap_gains_fn,
            13: lambda x: map_array_values(x, country_map),
            14: lambda x: map_array_values(x, label_map),
        }
        dataset = load_csv_dataset(
            os.path.join(dataset_folder, 'adult/adult.data'), -1, ', ',
            feature_names=feature_names, features_to_use=features_to_use,
            categorical_features=categorical_features, discretize=discretize,
            balance=balance, feature_transformations=transformations)

    elif dataset_name == 'titanic':
        feature_names = ["PassengerId", "Pclass",  "First Name", "Last Name", "Sex",
                "Age", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked", "Survived"]
        features_to_use = [1, 4, 5, 6, 7, 11]
        categorical_features = [1, 4, 6, 7, 11]

        sex_map = {'0.00': 'Female', 1: 'Male'}
        pclass_map = {1: '1st', 2: '2nd', 3: '3rd'}
        transformations = {
            4: lambda x: map_array_values(x, sex_map),
            1: lambda x: map_array_values(x, pclass_map)
        }
        dataset = load_csv_dataset(
            os.path.join(dataset_folder, 'titanic/titanic.data'), -1, ', ',
            feature_names=feature_names, features_to_use=features_to_use,
            categorical_features=categorical_features,
            discretize=discretize,
            balance=balance, feature_transformations=transformations)
        dataset.class_names = ['Survived', 'Died']

    elif 'generate' in dataset_name:
        feature_names = ["x", "y", "class"]

        dataset = load_csv_dataset(
            os.path.join(dataset_folder, 'generate/generate.data'), -1, ', ',
            feature_names=feature_names, discretize=discretize, balance=balance)
        dataset.class_names = ['class 0', 'class 1']

    elif dataset_name == 'diabetes':
        categorical_features = [2, 3, 4, 5, 6, 7, 8, 10, 11, 18, 19, 20, 22,
                                23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
                                35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
                                47, 48]
        label_map = {'<30': 'YES', '>30': 'YES'}
        transformations = {
            49: lambda x: map_array_values(x, label_map),
        }
        dataset = load_csv_dataset(
            os.path.join(dataset_folder, 'diabetes/diabetic_data.csv'), -1, ',',
            features_to_use=range(2, 49),
            categorical_features=categorical_features, discretize=discretize,
            balance=balance, feature_transformations=transformations)
    elif dataset_name == 'default':
        categorical_features = [2, 3, 4, 6, 7, 8, 9, 10, 11]
        dataset = load_csv_dataset(
                os.path.join(dataset_folder, 'default/default.csv'), -1, ',',
                features_to_use=range(1, 24),
                categorical_features=categorical_features, discretize=discretize,
                balance=balance)
    elif dataset_name == 'recidivism':
        features_to_use = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        feature_names = ['Race', 'Alcohol', 'Junky', 'Supervised Release',
                         'Married', 'Felony', 'WorkRelease',
                         'Crime against Property', 'Crime against Person',
                         'Gender', 'Priors', 'YearsSchool', 'PrisonViolations',
                         'Age', 'MonthsServed', '', 'Recidivism']

        def violations_fn(x):
            x = x.astype(float)
            d = np.digitize(x, [0, 5, float('inf')],
                            right=True).astype('|S128')
            return map_array_values(d, {'0': 'NO', '1': '1 to 5', '2': 'More than 5'})

        def priors_fn(x):
            x = x.astype(float)
            d = np.digitize(x, [-1, 0, 5, float('inf')],
                            right=True).astype('|S128')
            return map_array_values(d, {'0': 'UNKNOWN', '1': 'NO', '2': '1 to 5', '3': 'More than 5'})

        transformations = {
            0: lambda x: replace_binary_values(x, ['Black', 'White']),
            1: lambda x: replace_binary_values(x, ['No', 'Yes']),
            2: lambda x: replace_binary_values(x, ['No', 'Yes']),
            3: lambda x: replace_binary_values(x, ['No', 'Yes']),
            4: lambda x: replace_binary_values(x, ['No', 'Married']),
            5: lambda x: replace_binary_values(x, ['No', 'Yes']),
            6: lambda x: replace_binary_values(x, ['No', 'Yes']),
            7: lambda x: replace_binary_values(x, ['No', 'Yes']),
            8: lambda x: replace_binary_values(x, ['No', 'Yes']),
            9: lambda x: replace_binary_values(x, ['Female', 'Male']),
            10: lambda x: priors_fn(x),
            12: lambda x: violations_fn(x),
            13: lambda x: (x.astype(float) / 12).astype(int),
            16: lambda x: replace_binary_values(x, ['No more crimes',
                                                    'Re-arrested'])
        }

        dataset = load_csv_dataset(
            os.path.join(dataset_folder, 'recidivism/Data_1980.csv'), 16,
            feature_names=feature_names, discretize=discretize,
            features_to_use=features_to_use, balance=balance,
            feature_transformations=transformations, skip_first=True)

    elif dataset_name == 'lending':
        def filter_fn(data):
            to_remove = ['Does not meet the credit policy. Status:Charged Off',
               'Does not meet the credit policy. Status:Fully Paid',
               'In Grace Period', '-999', 'Current']
            for x in to_remove:
                data = data[data[:, 16] != x]
            return data
        bad_statuses = set(["Late (16-30 days)", "Late (31-120 days)", "Default", "Charged Off"])
        transformations = {
            16:  lambda x: np.array([y in bad_statuses for y in x]).astype(int),
            19:  lambda x: np.array([len(y) for y in x]).astype(int),
            6:  lambda x: np.array([y.strip('%') if y else -1 for y in x]).astype(float),
            35:  lambda x: np.array([y.strip('%') if y else -1 for y in x]).astype(float),
        }
        features_to_use = [2, 12, 13, 19, 29, 35, 51, 52, 109]
        categorical_features = [12, 109]
        dataset = load_csv_dataset(
            os.path.join(dataset_folder, 'lendingclub/LoanStats3a_securev1.csv'),
            16, ',',  features_to_use=features_to_use,
            feature_transformations=transformations, fill_na='-999',
            categorical_features=categorical_features, discretize=discretize,
            filter_fn=filter_fn, balance=True)
        dataset.class_names = ['Good Loan', 'Bad Loan']
    return dataset



def load_csv_dataset(data, target_idx, delimiter=',',
                     feature_names=None, categorical_features=None,
                     features_to_use=None, feature_transformations=None,
                     discretize=False, balance=False, fill_na='-1', filter_fn=None, skip_first=False):
    """Load a CSV dataset and prepare it for anchor explanations.

    Args:
        data: Path to the CSV file or a numpy array.
        target_idx: Index of the target/label column. Negative values
            count from the end.
        delimiter: Column delimiter character.
        feature_names: List of feature names. If None, uses first row.
        categorical_features: List of indices of categorical features.
            If None, features with <20 unique values are treated as categorical.
        features_to_use: List of feature indices to keep. If None, uses all
            except target.
        feature_transformations: Dict mapping feature index to transformation
            function.
        discretize: If True, discretize continuous features using quartiles.
        balance: If True, balance classes by undersampling.
        fill_na: String to fill NA values with.
        filter_fn: Optional function to filter rows.
        skip_first: If True, skip the first data row.

    Returns:
        A Bunch object with attributes: train, test, validation, labels_train,
        labels_test, labels_validation, feature_names, categorical_features,
        categorical_names, class_names, etc.
    """
    if feature_transformations is None:
        feature_transformations = {}
    try:
        data = np.genfromtxt(data, delimiter=delimiter, dtype='|S128')
    except Exception:
        import pandas
        data = pandas.read_csv(data,
                               header=None,
                               delimiter=delimiter,
                               na_filter=True,
                               dtype=str).fillna(fill_na).values
    if target_idx < 0:
        target_idx = data.shape[1] + target_idx
    ret = Bunch({})
    if feature_names is None:
        feature_names = list(data[0])
        data = data[1:]
    else:
        feature_names = copy.deepcopy(feature_names)
    if skip_first:
        data = data[1:]
    if filter_fn is not None:
        data = filter_fn(data)
    for feature, fun in feature_transformations.items():
        data[:, feature] = fun(data[:, feature])
    labels = data[:, target_idx]
    le = sklearn.preprocessing.LabelEncoder()
    le.fit(labels)
    ret.labels = le.transform(labels)
    labels = ret.labels
    ret.class_names = list(le.classes_)
    ret.class_target = feature_names[target_idx]
    if features_to_use is not None:
        data = data[:, features_to_use]
        feature_names = ([x for i, x in enumerate(feature_names)
                          if i in features_to_use])
        if categorical_features is not None:
            categorical_features = ([features_to_use.index(x)
                                     for x in categorical_features])
    else:
        data = np.delete(data, target_idx, 1)
        feature_names.pop(target_idx)
        if categorical_features:
            categorical_features = ([x if x < target_idx else x - 1
                                     for x in categorical_features])

    if categorical_features is None:
        categorical_features = []
        for f in range(data.shape[1]):
            if len(np.unique(data[:, f])) < 20:
                categorical_features.append(f)
    categorical_names = {}
    for feature in categorical_features:
        le = sklearn.preprocessing.LabelEncoder()
        le.fit(data[:, feature])
        data[:, feature] = le.transform(data[:, feature])
        categorical_names[feature] = le.classes_
    data = data.astype(float)
    ordinal_features = []
    if discretize:
        disc = lime.lime_tabular.QuartileDiscretizer(data,
                                                     categorical_features,
                                                     feature_names)
        data = disc.discretize(data)
        ordinal_features = [x for x in range(data.shape[1])
                            if x not in categorical_features]
        categorical_features = range(data.shape[1])
        categorical_names.update(disc.names)
    for x in categorical_names:
        categorical_names[x] = [y.decode() if type(y) == np.bytes_ else y for y in categorical_names[x]]
    ret.ordinal_features = ordinal_features
    ret.categorical_features = categorical_features
    ret.categorical_names = categorical_names
    ret.feature_names = feature_names
    np.random.seed(1)
    if balance:
        idxs = np.array([], dtype='int')
        min_labels = np.min(np.bincount(labels))
        for label in np.unique(labels):
            idx = np.random.choice(np.where(labels == label)[0], min_labels)
            idxs = np.hstack((idxs, idx))
        data = data[idxs]
        labels = labels[idxs]
        ret.data = data
        ret.labels = labels

    splits = sklearn.model_selection.ShuffleSplit(n_splits=1,
                                                  test_size=.2,
                                                  random_state=1)
    train_idx, test_idx = [x for x in splits.split(data)][0]
    ret.train = data[train_idx]
    ret.labels_train = ret.labels[train_idx]
    cv_splits = sklearn.model_selection.ShuffleSplit(n_splits=1,
                                                     test_size=.5,
                                                     random_state=1)
    cv_idx, ntest_idx = [x for x in cv_splits.split(test_idx)][0]
    cv_idx = test_idx[cv_idx]
    test_idx = test_idx[ntest_idx]
    ret.validation = data[cv_idx]
    ret.labels_validation = ret.labels[cv_idx]
    ret.test = data[test_idx]
    ret.labels_test = ret.labels[test_idx]
    ret.test_idx = test_idx
    ret.validation_idx = cv_idx
    ret.train_idx = train_idx

    ret.data = data
    return ret


class Neighbors:
    """Finds word neighbors using spaCy word vectors.

    This class caches neighbor lookups for efficiency during
    text perturbation.

    Args:
        nlp_obj: A spaCy language model with word vectors.
    """

    def __init__(self, nlp_obj):
        self.nlp = nlp_obj
        self.to_check = [w for w in self.nlp.vocab if w.prob >= -15]
        self.n = {}

    def neighbors(self, word):
        """Find the 500 most similar words to the given word.

        Args:
            word: The target word to find neighbors for.

        Returns:
            A list of (spacy_token, similarity_score) tuples,
            sorted by similarity in descending order.
        """
        word = unicode(word)
        orig_word = word
        if word not in self.n:
            if word not in self.nlp.vocab:
                self.n[word] = []
            else:
                word = self.nlp.vocab[unicode(word)]
                queries = [w for w in self.to_check
                            if w.is_lower == word.is_lower]
                if word.prob < -15:
                    queries += [word]
                by_similarity = sorted(
                    queries, key=lambda w: word.similarity(w), reverse=True)
                self.n[orig_word] = [(self.nlp(w.orth_)[0], word.similarity(w))
                                     for w in by_similarity[:500]]
        return self.n[orig_word]


def perturb_sentence(text, present, n, neighbors, proba_change=0.5,
                     top_n=50, forbidden=None, forbidden_tags=None,
                     forbidden_words=None,
                     pos=None, use_proba=True,
                     temperature=.4):
    """Perturb a sentence by replacing words with similar alternatives.

    Generates n perturbed versions of the input text by replacing words
    (that are not in the 'present' set) with similar words found via
    word embeddings.

    Args:
        text: The input text to perturb.
        present: List of word indices that must remain unchanged.
        n: Number of perturbed samples to generate.
        neighbors: A Neighbors instance for finding similar words.
        proba_change: Probability of changing each word (when use_proba=False).
        top_n: Maximum number of neighbor candidates to consider.
        forbidden: Set of forbidden lemmas (not to be replaced).
        forbidden_tags: Set of forbidden POS tags.
        forbidden_words: Set of forbidden word forms.
        pos: Set of POS tags eligible for replacement.
        use_proba: If True, sample replacements weighted by similarity.
        temperature: Temperature for similarity-based sampling.

    Returns:
        A tuple of (raw_data, data) where raw_data is a list of perturbed
        text strings and data is a binary matrix indicating word presence.
    """
    if forbidden is None:
        forbidden = []
    if forbidden_tags is None:
        forbidden_tags = ['PRP$']
    if forbidden_words is None:
        forbidden_words = ['be']
    if pos is None:
        pos = ['NOUN', 'VERB', 'ADJ', 'ADV', 'ADP', 'DET']

    tokens = neighbors.nlp(unicode(text))
    forbidden = set(forbidden)
    forbidden_tags = set(forbidden_tags)
    forbidden_words = set(forbidden_words)
    pos = set(pos)
    raw = np.zeros((n, len(tokens)), '|S80')
    data = np.ones((n, len(tokens)))
    raw[:] = [x.text for x in tokens]
    for i, t in enumerate(tokens):
        if i in present:
            continue
        if (t.text not in forbidden_words and t.pos_ in pos and
                t.lemma_ not in forbidden and t.tag_ not in forbidden_tags):
            # Returns words that have the same tag (i.e: Nouns, adj, etc...)
            # among the 500 words that are most similar to the word in entry
            r_neighbors = [
                (unicode(x[0].text.encode('utf-8'), errors='ignore'), x[1])
                for x in neighbors.neighbors(t.text)
                if x[0].tag_ == t.tag_][:top_n]
            if not r_neighbors:
                continue
            t_neighbors = [x[0] for x in r_neighbors]
            weights = np.array([x[1] for x in r_neighbors])
            if use_proba:
                weights = weights ** (1. / temperature)
                weights = weights / sum(weights)
                raw[:, i] = np.random.choice(t_neighbors, n,  p=weights,
                                             replace=True)
                data[:, i] = raw[:, i] == t.text.encode()
            else:
                n_changed = np.random.binomial(n, proba_change)
                changed = np.random.choice(n, n_changed, replace=False)
                if t.text in t_neighbors:
                    idx = t_neighbors.index(t.text)
                    weights[idx] = 0
                weights = weights / sum(weights)
                raw[changed, i] = np.random.choice(t_neighbors, n_changed, p=weights)
                data[changed, i] = 0
    if (sys.version_info > (3, 0)):
        raw = [' '.join([y.decode() for y in x]) for x in raw]
    else:
        raw = [' '.join(x) for x in raw]
    return raw, data


def return_pertinent_sentences(pertinent, raw_data, m):
    """Generate sentences with pertinent negative words added.

    Constructs perturbed sentences where words are either present or
    absent based on the pertinent matrix.

    Args:
        pertinent: Binary matrix (m x len(raw_data)) indicating word presence.
        raw_data: List of word byte-strings.
        m: Number of sentences to generate.

    Returns:
        List of reconstructed sentence strings.
    """
    pertinent_sentences = np.zeros((m, len(raw_data)), '|S80')
    for i, t in enumerate(raw_data):
        for j in range(m):
            if pertinent[j][i] == 1:
                pertinent_sentences[j][i] = raw_data[i]
            else:
                pertinent_sentences[j][i] = ""
    if (sys.version_info > (3, 0)):
        raw = []
        for x in pertinent_sentences:
            text = " "
            for y in x:
                if y.decode():
                    text += " " + ' '.join([y.decode()])
            raw.append(text)
    else:
        raw = [' '.join(x) for x in pertinent_sentences]
    return raw


def generate_false_pertinent(text, present, m, neighbors, n_best_co_occurrence, proba_change=0.5,
                     forbidden=None, forbidden_tags=None,
                     forbidden_words=None,
                     pos=None, use_proba=True, generate_sentence=False):
    """Generate pertinent negatives using co-occurrence statistics.

    Creates a matrix of sentences augmented with 'false pertinent' words
    that frequently co-occur with words in the original text.

    Args:
        text: The input text to augment.
        present: List of word indices that must be present.
        m: Number of samples to generate.
        neighbors: A Neighbors instance for NLP processing.
        n_best_co_occurrence: Pre-computed co-occurrence statistics.
        proba_change: Probability of changing each word.
        forbidden: List of forbidden lemmas.
        forbidden_tags: List of forbidden POS tags.
        forbidden_words: List of forbidden word forms.
        pos: List of POS tags eligible for modification.
        use_proba: If True, use probability-weighted sampling.
        generate_sentence: If True, return a single augmented sentence string
            instead of the full perturbation matrix.

    Returns:
        If generate_sentence is True: a string with all pertinent negative words.
        Otherwise: a tuple of (pertinent_matrix, raw_sentences, word_array).
    """
    if forbidden is None:
        forbidden = []
    if forbidden_tags is None:
        forbidden_tags = ['PRP$']
    if forbidden_words is None:
        forbidden_words = ['be']
    if pos is None:
        pos = ['NOUN', 'VERB', 'ADJ', 'ADV', 'ADP', 'DET']

    # Use of classical natural language processing
    tokens = neighbors.nlp(unicode(text))
    forbidden = set(forbidden)
    forbidden_tags = set(forbidden_tags)
    forbidden_words = set(forbidden_words)
    pos = set(pos)
    sentence = []
    for x in tokens:
        sentence.append(x.text)
    pertinent = np.zeros(m)
    array_false_pertinent = []
    for i, t in enumerate(sentence):
        array_false_pertinent.append(t.encode('ascii'))
        # gets the most frequent words associated with the target word t
        targets = co_occ.generate_bi_grams_words(t, n_best_co_occurrence)
        # Put to 1 for all sentence generated at the position of the word from the target sentence
        pertinent = np.c_[pertinent, np.ones(m)]
        if targets != []:
            # Add randomly a 1 in the matrix for (only) one of the most co occurent words
            size_pertinents = len(targets)
            matrix_raw_false_pertinent = np.zeros((m, size_pertinents))
            for j, p in enumerate(targets):
                array_false_pertinent.append(p.encode('ascii'))
            k = 0
            for i in range(m):
                matrix_raw_false_pertinent[i][k] = 1
                k += 1
                k = k % size_pertinents
            np.random.shuffle(matrix_raw_false_pertinent)
            pertinent = np.c_[pertinent, matrix_raw_false_pertinent]
    if generate_sentence:
        # generates a sentence composed of all the pertinent negatifs words inside the target sentence
        sentence_false_pertinent = ""
        for word in array_false_pertinent:
            sentence_false_pertinent += " " + word.decode()
        return sentence_false_pertinent
    pertinent = np.delete(pertinent, 0, 1)
    raw = return_pertinent_sentences(pertinent, array_false_pertinent, m)
    return pertinent, raw, array_false_pertinent


def return_pertinent_sentences_replace(pertinent, raw_data, m, raw, array_replace_words):
    """Generate sentences with pertinent negatives using word replacement.

    Similar to return_pertinent_sentences but also replaces words at
    specific positions with perturbed alternatives.

    Args:
        pertinent: Binary matrix indicating word presence.
        raw_data: List of word byte-strings.
        m: Number of sentences to generate.
        raw: Matrix of replacement words.
        array_replace_words: Indices of words to replace.

    Returns:
        List of reconstructed sentence strings.
    """
    pertinent_sentences = np.zeros((m, len(raw_data)), '|S80')
    counter = 0
    for i, t in enumerate(raw_data):
        if i in array_replace_words:
            pertinent_sentences[:, i] = raw[:, counter]
            counter += 1
        else:
            for j in range(m):
                if pertinent[j][i] == 1:
                    pertinent_sentences[j][i] = t
                else:
                    pertinent_sentences[j][i] = ""
    if (sys.version_info > (3, 0)):
        raw = []
        for x in pertinent_sentences:
            text = " "
            for y in x:
                if y.decode():
                    text += " " + ' '.join([y.decode()])
            raw.append(text)
    else:
        raw = [' '.join(x) for x in pertinent_sentences]
    return raw


def generate_false_pertinents_replace(text, present, m, neighbors, n_best_co_occurrence, proba_change=0.5,
                     forbidden=None, forbidden_tags=None,
                     forbidden_words=None, top_n=50, temperature=.4,
                     pos=None, use_proba=True, generate_sentence=False):
    """Generate pertinent negatives with word replacement strategy.

    Combines co-occurrence-based pertinent negative generation with
    word embedding-based replacement of existing words.

    Args:
        text: The input text to augment.
        present: List of word indices that must be present.
        m: Number of samples to generate.
        neighbors: A Neighbors instance for NLP processing.
        n_best_co_occurrence: Pre-computed co-occurrence statistics.
        proba_change: Probability of changing each word.
        forbidden: List of forbidden lemmas.
        forbidden_tags: List of forbidden POS tags.
        forbidden_words: List of forbidden word forms.
        top_n: Maximum number of neighbor candidates.
        temperature: Temperature for similarity-based sampling.
        pos: List of POS tags eligible for modification.
        use_proba: If True, use probability-weighted sampling.
        generate_sentence: If True, return a single augmented sentence string.

    Returns:
        If generate_sentence is True: a string with all pertinent negative words.
        Otherwise: a tuple of (pertinent_matrix, raw_sentences, word_array).
    """
    if forbidden is None:
        forbidden = []
    if forbidden_tags is None:
        forbidden_tags = ['PRP$']
    if forbidden_words is None:
        forbidden_words = ['be']
    if pos is None:
        pos = ['NOUN', 'VERB', 'ADJ', 'ADV', 'ADP', 'DET']

    # Use of classical natural language processing
    tokens = neighbors.nlp(unicode(text))
    forbidden = set(forbidden)
    forbidden_tags = set(forbidden_tags)
    forbidden_words = set(forbidden_words)
    pos = set(pos)
    sentence = []
    for x in tokens:
        sentence.append(x.text)
    pertinent = np.zeros(m)
    array_false_pertinent = []
    raw = np.zeros((m, len(tokens)), '|S80')
    data = np.ones((m, len(tokens)))
    raw[:] = [x.text for x in tokens]
    array_replace_words = []
    counter = 0
    for i, t in enumerate(tokens):
        if (t.text not in forbidden_words and t.pos_ in pos and
                t.lemma_ not in forbidden and t.tag_ not in forbidden_tags):
            # Returns words that have the same tag (i.e: Nouns, adj, etc...)
            # among the 500 words that are most similar to the word in entry
            r_neighbors = [
                (unicode(x[0].text.encode('utf-8'), errors='ignore'), x[1])
                for x in neighbors.neighbors(t.text)
                if x[0].tag_ == t.tag_][:top_n]
            if not r_neighbors:
                continue
            t_neighbors = [x[0] for x in r_neighbors]
            weights = np.array([x[1] for x in r_neighbors])
            if use_proba:
                weights = weights ** (1. / temperature)
                weights = weights / sum(weights)
                raw[:, i] = np.random.choice(t_neighbors, m,  p=weights,
                                             replace=True)
                data[:, i] = raw[:, i] == t.text.encode()
            else:
                n_changed = np.random.binomial(m, proba_change)
                changed = np.random.choice(m, n_changed, replace=False)
                if t.text in t_neighbors:
                    idx = t_neighbors.index(t.text)
                    weights[idx] = 0
                weights = weights / sum(weights)
                raw[changed, i] = np.random.choice(t_neighbors, n_changed, p=weights)
                data[changed, i] = 0
        array_false_pertinent.append(t.text.encode('ascii'))
        # gets the most frequent words associated with the target word t
        targets = co_occ.generate_bi_grams_words(t.text, n_best_co_occurrence)
        # Put to 1 for all sentence generated at the position of the word from the target sentence
        pertinent = np.c_[pertinent, data[:, i]]
        array_replace_words.append(counter)
        counter += len(targets) + 1
        if targets != []:
            # Add randomly a 1 in the matrix for (only) one of the most co occurent words
            size_pertinents = len(targets)
            matrix_raw_false_pertinent = np.zeros((m, size_pertinents))
            for j, p in enumerate(targets):
                array_false_pertinent.append(p.encode('ascii'))
            k = 0
            for i in range(m):
                matrix_raw_false_pertinent[i][k] = 1
                k += 1
                k = k % size_pertinents
            np.random.shuffle(matrix_raw_false_pertinent)
            pertinent = np.c_[pertinent, matrix_raw_false_pertinent]
    if generate_sentence:
        # generates a sentence composed of all the pertinent negatifs words inside the target sentence
        sentence_false_pertinent = ""
        for word in array_false_pertinent:
            sentence_false_pertinent += " " + word.decode()
        return sentence_false_pertinent
    pertinent = np.delete(pertinent, 0, 1)
    raw = return_pertinent_sentences_replace(pertinent, array_false_pertinent, m, raw, array_replace_words)
    return pertinent, raw, array_false_pertinent
