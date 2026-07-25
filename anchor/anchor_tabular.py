"""Anchor explainer for tabular (structured) data.

This module provides the AnchorTabularExplainer class for generating
anchor-based explanations for classifiers operating on tabular data
(numpy arrays of numerical or categorical features).
"""
import collections
import copy
import json
import os
import string
from io import open

import numpy as np
import sklearn
import lime
import lime.lime_tabular

import anchor_base
import anchor_explanation


def id_generator(size=15):
    """Generate a random alphanumeric string for unique HTML div IDs.

    This is useful for embedding multiple explanations in a single
    HTML document or IPython notebook without ID collisions.

    Args:
        size: Length of the generated string. Defaults to 15.

    Returns:
        A random string of uppercase letters and digits.
    """
    chars = list(string.ascii_uppercase + string.digits)
    return ''.join(np.random.choice(chars, size, replace=True))


class AnchorTabularExplainer(object):
    """Generates anchor explanations for tabular data classifiers.

    An anchor explanation is a set of if-then rules (predicates) that
    sufficiently "anchor" a prediction — changes to features not in
    the anchor do not affect the prediction.

    Args:
        class_names: List of class name strings.
        feature_names: List of feature name strings.
        train_data: Training data numpy array used for sampling
            (bootstrap perturbations).
        categorical_names: Dict mapping feature index to list of
            category name strings. Features not in this map are treated
            as continuous and will be discretized.
        black_box_labels: Labels from black box for MDLP/entropy discretizers.
        discretizer: Discretization method. One of 'quartile', 'decile',
            'MDLP', 'entropy', 'kmeans'.
        filename: Filename for KMeans discretizer persistence.
        ordinal_features: List of ordinal feature indices.
    """
    def __init__(self, class_names, feature_names, train_data, categorical_names=None,
                black_box_labels=None, discretizer='quartile', filename="", ordinal_features=None):
        if ordinal_features is None:
            ordinal_features = []
        self.encoder = collections.namedtuple('random_name',
                                              ['transform'])(lambda x: x)
        self.disc = collections.namedtuple('random_name2',
                                              ['discretize'])(lambda x: x)
        self.categorical_features = []
        if categorical_names:
            cat_names = sorted(categorical_names.keys())
            n_values = [len(categorical_names[i]) for i in cat_names]
            self.encoder = sklearn.preprocessing.OneHotEncoder(
                categories='auto')
            self.encoder.fit(train_data)
            self.categorical_features = list(cat_names)
        else:
            categorical_names = {}
        if len(ordinal_features) == 0:
            self.ordinal_features = [x for x in range(len(feature_names)) if x not in self.categorical_features]
        self.feature_names = feature_names
        self.class_names = class_names
        self.categorical_names = categorical_names

        self.min = {}
        self.max = {}
        self.std = {}
        self.train = train_data
        self.scaler = sklearn.preprocessing.StandardScaler()
        self.scaler.fit(train_data)
        if discretizer == 'quartile':
            self.disc = lime.lime_tabular.QuartileDiscretizer(train_data,
                                                         self.categorical_features,
                                                         self.feature_names)
        elif discretizer == 'decile':
            self.disc = lime.lime_tabular.DecileDiscretizer(train_data,
                                                     self.categorical_features,
                                                     self.feature_names)
        elif discretizer == 'MDLP':
            self.disc = lime.lime_tabular.MDLPDiscretizer(train_data,
                                                          self.categorical_features,
                                                          self.feature_names,
                                                          labels=black_box_labels)
        elif discretizer == 'entropy':
            self.disc = lime.lime_tabular.EntropyDiscretizer(train_data,
                                                          self.categorical_features,
                                                          self.feature_names,
                                                          labels=black_box_labels)
        elif discretizer == 'kmeans':
            self.disc = lime.lime_tabular.KMeansDiscretizer(train_data,
                                                          self.categorical_features,
                                                          self.feature_names, filename)

        else:
            raise ValueError('Discretizer must be quartile, decile, MDLP, entropy or kmeans')

        self.d_train = self.disc.discretize(self.train)
        val = self.disc.discretize(train_data)
        self.categorical_names.update(self.disc.names)
        self.ordinal_features = [x for x in range(val.shape[1])
                            if x not in self.categorical_features]
        self.categorical_features += self.ordinal_features

        for f in range(train_data.shape[1]):
            if f in self.categorical_features and f not in self.ordinal_features:
                continue
            self.min[f] = np.min(train_data[:, f])
            self.max[f] = np.max(train_data[:, f])
            self.std[f] = np.std(train_data[:, f])


    def sample_from_train(self, conditions_eq, conditions_neq, conditions_geq,
                          conditions_leq, num_samples, validation=True):
        """Sample instances from training data subject to anchor conditions.

        Generates num_samples bootstrap samples from the training data,
        enforcing equality/inequality constraints from the anchor rules.

        Args:
            conditions_eq: Dict of {feature_idx: value} equality constraints.
            conditions_neq: Dict of {feature_idx: value} inequality constraints.
            conditions_geq: Dict of {feature_idx: value} greater-or-equal constraints.
            conditions_leq: Dict of {feature_idx: value} less-or-equal constraints.
            num_samples: Number of samples to generate.
            validation: Whether to use validation sampling.

        Returns:
            numpy array of sampled instances (num_samples x n_features).
        """
        train = self.train
        d_train = self.d_train
        idx = np.random.choice(range(train.shape[0]), num_samples,
                               replace=True)
        sample = train[idx]
        d_sample = d_train[idx]
        for f in conditions_eq:
            sample[:, f] = np.repeat(conditions_eq[f], num_samples)
        for f in conditions_geq:
            idx = d_sample[:, f] <= conditions_geq[f]
            if f in conditions_leq:
                idx = (idx + (d_sample[:, f] > conditions_leq[f])).astype(bool)
            if idx.sum() == 0:
                continue
            options = d_train[:, f] > conditions_geq[f]
            if f in conditions_leq:
                options = options * (d_train[:, f] <= conditions_leq[f])
            if options.sum() == 0:
                min_ = conditions_geq.get(f, self.min[f])
                max_ = conditions_leq.get(f, self.max[f])
                to_rep = np.random.uniform(min_, max_, idx.sum())
            else:
                to_rep = np.random.choice(train[options, f], idx.sum(),
                                          replace=True)
            sample[idx, f] = to_rep
        for f in conditions_leq:
            if f in conditions_geq:
                continue
            idx = d_sample[:, f] > conditions_leq[f]
            if idx.sum() == 0:
                continue
            options = d_train[:, f] <= conditions_leq[f]
            if options.sum() == 0:
                min_ = conditions_geq.get(f, self.min[f])
                max_ = conditions_leq.get(f, self.max[f])
                to_rep = np.random.uniform(min_, max_, idx.sum())
            else:
                to_rep = np.random.choice(train[options, f], idx.sum(),
                                          replace=True)
            sample[idx, f] = to_rep
        return sample


    def transform_to_examples(self, examples, features_in_anchor=None,
                              predicted_label=None):
        """Transform raw data examples to human-readable format.

        Args:
            examples: numpy array of example instances.
            features_in_anchor: Set of feature indices in the current anchor.
            predicted_label: The predicted class label index.

        Returns:
            List of lists, each containing (feature_name, value, weight) tuples.
        """
        if features_in_anchor is None:
            features_in_anchor = []
        ret_obj = []
        if len(examples) == 0:
            return ret_obj
        weights = [int(predicted_label) if x in features_in_anchor else -1
                   for x in range(examples.shape[1])]
        examples = self.disc.discretize(examples)
        for ex in examples:
            values = [self.categorical_names[i][int(ex[i])]
                      if i in self.categorical_features
                      else ex[i] for i in range(ex.shape[0])]
            ret_obj.append(list(zip(self.feature_names, values, weights)))
        return ret_obj

    def to_explanation_map(self, exp):
        """Convert an anchor explanation dict to a visualization-ready map.

        Args:
            exp: The raw anchor explanation dictionary.

        Returns:
            Dict with keys: explanation, rawData, predictProba, labelNames,
            rawDataType, explanationType, trueClass.
        """
        instance = exp['instance']
        predicted_label = exp['prediction']
        predict_proba = np.zeros(len(self.class_names))
        predict_proba[predicted_label] = 1

        examples_obj = []
        for i, temp in enumerate(exp['examples'], start=1):
            features_in_anchor = set(exp['feature'][:i])
            ret = {}
            ret['coveredFalse'] = self.transform_to_examples(
                temp['covered_false'], features_in_anchor, predicted_label)
            ret['coveredTrue'] = self.transform_to_examples(
                temp['covered_true'], features_in_anchor, predicted_label)
            ret['uncoveredTrue'] = self.transform_to_examples(
                temp['uncovered_true'], features_in_anchor, predicted_label)
            ret['uncoveredFalse'] = self.transform_to_examples(
                temp['uncovered_false'], features_in_anchor, predicted_label)
            ret['covered'] = self.transform_to_examples(
                temp['covered'], features_in_anchor, predicted_label)
            examples_obj.append(ret)

        explanation = {'names': exp['names'],
                       'certainties': exp['precision'] if len(exp['precision']) else [exp['all_precision']],
                       'supports': exp['coverage'],
                       'allPrecision': exp['all_precision'],
                       'examples': examples_obj,
                       'onlyShowActive': False}
        weights = [-1 for x in range(instance.shape[0])]
        instance = self.disc.discretize(exp['instance'].reshape(1, -1))[0]
        values = [self.categorical_names[i][int(instance[i])]
                  if i in self.categorical_features
                  else instance[i] for i in range(instance.shape[0])]
        raw_data = list(zip(self.feature_names, values, weights))
        ret = {
            'explanation': explanation,
            'rawData': raw_data,
            'predictProba': list(predict_proba),
            'labelNames': list(map(str, self.class_names)),
            'rawDataType': 'tabular',
            'explanationType': 'anchor',
            'trueClass': False
        }
        return ret

    def as_html(self, exp, **kwargs):
        """Render an anchor explanation as a self-contained HTML page.

        The generated HTML embeds the pre-built bundle.js and renders
        an interactive explanation visualization.

        Args:
            exp: The raw anchor explanation dictionary.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            A unicode string containing the complete HTML document.
        """
        exp_map = self.to_explanation_map(exp)

        this_dir, _ = os.path.split(__file__)
        with open(os.path.join(this_dir, 'bundle.js'), encoding='utf8') as f:
            bundle = f.read()
        random_id = 'top_div' + id_generator()

        # Use json.dumps for all dynamic data to prevent XSS.
        # The random_id is generated internally and is safe.
        out = u'''<html>
        <meta http-equiv="content-type" content="text/html; charset=UTF8">
        <head><script>%s </script></head><body>''' % bundle
        out += u'''
        <div id="{random_id}" />
        <script>
            div = d3.select("#{random_id}");
            lime.RenderExplanationFrame(div,{label_names}, {predict_proba},
            {true_class}, {explanation}, {raw_data}, "tabular", {explanation_type});
        </script>'''.format(random_id=random_id,
                            label_names=json.dumps(exp_map['labelNames']),
                            predict_proba=json.dumps(exp_map['predictProba']),
                            true_class=json.dumps(exp_map['trueClass']),
                            explanation=json.dumps(exp_map['explanation']),
                            raw_data=json.dumps(exp_map['rawData']),
                            explanation_type=json.dumps(exp_map['explanationType']))
        out += u'</body></html>'
        return out


    def get_sample_fn(self, data_row, classifier_fn, desired_label=None):
        """Create a sampling function for anchor beam search.

        Constructs a closure that generates perturbed samples around
        the given data row, subject to anchor constraints.

        Args:
            data_row: The instance to explain (1-D numpy array).
            classifier_fn: Prediction function that takes a 2-D array
                and returns class predictions.
            desired_label: Target label for the explanation. If None,
                uses the model's prediction on data_row.

        Returns:
            A tuple of (sample_fn, mapping) where sample_fn is the
            sampling closure and mapping maps feature indices to
            (feature, operator, value) triples.
        """
        def predict_fn(x):
            return classifier_fn(x)
        true_label = desired_label
        if true_label is None:
            true_label = predict_fn(data_row.reshape(1, -1))[0]
        # must map present here to include categorical features (for conditions_eq),
        # and numerical features for geq and leq
        mapping = {}
        data_row = self.disc.discretize(data_row.reshape(1, -1))[0]
        for f in self.categorical_features:
            if f in self.ordinal_features:
                for v in range(len(self.categorical_names[f])):
                    idx = len(mapping)
                    if data_row[f] <= v and v != len(self.categorical_names[f]) - 1:
                        mapping[idx] = (f, 'leq', v)
                    elif data_row[f] > v:
                        mapping[idx] = (f, 'geq', v)
            else:
                idx = len(mapping)
                mapping[idx] = (f, 'eq', data_row[f])

        def sample_fn(present, num_samples, compute_labels=True, validation=True):
            conditions_eq = {}
            conditions_leq = {}
            conditions_geq = {}
            for x in present:
                f, op, v = mapping[x]
                if op == 'eq':
                    conditions_eq[f] = v
                if op == 'leq':
                    if f not in conditions_leq:
                        conditions_leq[f] = v
                    conditions_leq[f] = min(conditions_leq[f], v)
                if op == 'geq':
                    if f not in conditions_geq:
                        conditions_geq[f] = v
                    conditions_geq[f] = max(conditions_geq[f], v)
            raw_data = self.sample_from_train(
                conditions_eq, {}, conditions_geq, conditions_leq, num_samples,
                validation=validation)
            d_raw_data = self.disc.discretize(raw_data)
            data = np.zeros((num_samples, len(mapping)), int)
            for i in mapping:
                f, op, v = mapping[i]
                if op == 'eq':
                    data[:, i] = (d_raw_data[:, f] == data_row[f]).astype(int)
                if op == 'leq':
                    data[:, i] = (d_raw_data[:, f] <= v).astype(int)
                if op == 'geq':
                    data[:, i] = (d_raw_data[:, f] > v).astype(int)
            labels = []
            if compute_labels:
                labels = (predict_fn(raw_data) == true_label).astype(int)
            return raw_data, data, labels
        return sample_fn, mapping

    def explain_instance(self, data_row, classifier_fn, threshold=0.95,
                          delta=0.1, tau=0.15, batch_size=100,
                          max_anchor_size=None,
                          desired_label=None,
                          beam_size=4, **kwargs):
        """Generate an anchor explanation for a single instance.

        Args:
            data_row: The instance to explain (1-D numpy array).
            classifier_fn: Prediction function for the model.
            threshold: Minimum precision threshold for the anchor.
            delta: Confidence parameter for the beam search.
            tau: Tolerance parameter (epsilon) for the beam search.
            batch_size: Number of samples per batch during search.
            max_anchor_size: Maximum number of predicates in the anchor.
            desired_label: Target label for the explanation.
            beam_size: Width of the beam search.
            **kwargs: Additional arguments passed to anchor_beam.

        Returns:
            An AnchorExplanation object with the explanation results.
        """
        sample_fn, mapping = self.get_sample_fn(
            data_row, classifier_fn, desired_label=desired_label)
        exp = anchor_base.AnchorBaseBeam.anchor_beam(
            sample_fn, delta=delta, epsilon=tau, batch_size=batch_size,
            desired_confidence=threshold, max_anchor_size=max_anchor_size,
            **kwargs)
        self.add_names_to_exp(data_row, exp, mapping)
        exp['instance'] = data_row
        exp['prediction'] = classifier_fn(data_row.reshape(1, -1))[0]
        explanation = anchor_explanation.AnchorExplanation('tabular', exp, self.as_html)
        return explanation

    def add_names_to_exp(self, data_row, hoeffding_exp, mapping):
        """Add human-readable names to anchor feature indices.

        Converts numeric feature indices and threshold values into
        readable strings like 'Age > 30' or 'Sex = Male'.

        Args:
            data_row: The original instance being explained.
            hoeffding_exp: The raw explanation dict from beam search.
            mapping: Feature index to (feature, op, value) mapping.
        """
        idxs = hoeffding_exp['feature']
        hoeffding_exp['names'] = []
        hoeffding_exp['feature'] = [mapping[idx][0] for idx in idxs]
        ordinal_ranges = {}
        for idx in idxs:
            f, op, v = mapping[idx]
            if op == 'geq' or op == 'leq':
                if f not in ordinal_ranges:
                    ordinal_ranges[f] = [float('-inf'), float('inf')]
            if op == 'geq':
                ordinal_ranges[f][0] = max(ordinal_ranges[f][0], v)
            if op == 'leq':
                ordinal_ranges[f][1] = min(ordinal_ranges[f][1], v)
        handled = set()
        for idx in idxs:
            f, op, v = mapping[idx]
            if op == 'eq':
                fname = '%s = ' % self.feature_names[f]
                if f in self.categorical_names:
                    v = int(v)
                    if ('<' in self.categorical_names[f][v]
                            or '>' in self.categorical_names[f][v]):
                        fname = ''
                    fname = '%s%s' % (fname, self.categorical_names[f][v])
                else:
                    fname = '%s%.2f' % (fname, v)
            else:
                if f in handled:
                    continue
                geq, leq = ordinal_ranges[f]
                fname = ''
                geq_val = ''
                leq_val = ''
                if geq > float('-inf'):
                    if geq == len(self.categorical_names[f]) - 1:
                        geq = geq - 1
                    name = self.categorical_names[f][geq + 1]
                    if '<' in name:
                        geq_val = name.split()[0]
                    elif '>' in name:
                        geq_val = name.split()[-1]
                if leq < float('inf'):
                    name = self.categorical_names[f][leq]
                    if leq == 0:
                        leq_val = name.split()[-1]
                    elif '<' in name:
                        leq_val = name.split()[-1]
                if leq_val and geq_val:
                    fname = '%s < %s <= %s' % (geq_val, self.feature_names[f],
                                               leq_val)
                elif leq_val:
                    fname = '%s <= %s' % (self.feature_names[f], leq_val)
                elif geq_val:
                    fname = '%s > %s' % (self.feature_names[f], geq_val)
                handled.add(f)
            hoeffding_exp['names'].append(fname)
