"""Anchor explainer for text classifiers.

This module provides the AnchorText class for generating anchor-based
explanations for text classification models. It supports word masking,
word replacement via embeddings, and pertinent negative generation.
"""
import json
import os
import string
import sys
from io import open

import numpy as np

import anchor_base
import anchor_explanation
import co_occurrence_matrix as co_occ
import utils

os.environ['SPACY_WARNING_IGNORE'] = 'W008'

# Python3 hack
try:
    UNICODE_EXISTS = bool(type(unicode))
except NameError:
    def unicode(s):
        return s


def id_generator(size=15):
    """Generate a random alphanumeric string for unique HTML div IDs.

    Args:
        size: Length of the generated string. Defaults to 15.

    Returns:
        A random string of uppercase letters and digits.
    """
    chars = list(string.ascii_uppercase + string.digits)
    return ''.join(np.random.choice(chars, size, replace=True))


class AnchorText(object):
    """Generates anchor explanations for text classifiers.

    An anchor explanation identifies a minimal set of words in the input
    text such that, as long as these words are present, the model's
    prediction remains (almost) always the same.

    Args:
        nlp: A spaCy language model with word vectors.
        class_names: List of class name strings.
        use_unk_distribution: If True, perturb by replacing words with
            mask_string. If False, replace with similar words via embeddings.
        mask_string: Token used to mask words when use_unk_distribution=True.
    """
    def __init__(self, nlp, class_names, use_unk_distribution=True, mask_string='UNK'):
        self.nlp = nlp
        self.class_names = class_names
        self.neighbors = utils.Neighbors(self.nlp)
        self.use_unk_distribution = use_unk_distribution
        self.mask_string = mask_string
        # Generates and store the co-occurrence matrix
        co_occurence = co_occ.generate_co_occurence_matrix()
        # Stores the n best words from the co-occurrence matrix
        self.n_best_co_occurrence = co_occ.generate_n_best_co_occurrence(co_occurence)

    def get_sample_fn(self, text, classifier_fn, use_proba=False, pertinents_negatifs=False, pertinents_negatifs_replace=False):
        """Create a sampling function for anchor beam search on text.

        Constructs a closure that generates perturbed versions of the
        input text by masking or replacing words.

        Args:
            text: The input text to explain.
            classifier_fn: Prediction function that takes a list of texts
                and returns predictions.
            use_proba: If True, use probability-weighted word replacement.
            pertinents_negatifs: If True, use pertinent negative generation
                with co-occurrence augmentation.
            pertinents_negatifs_replace: If True, use pertinent negative
                generation with word replacement strategy.

        Returns:
            A tuple of (words, positions, true_label, sample_fn).
        """
        true_label = classifier_fn([text])
        if pertinents_negatifs:
            # Generates a new sentence based on the target added with the false pertinents words
            sentence_false_pertinents = utils.generate_false_pertinent(
                    text, [], 1, self.neighbors, self.n_best_co_occurrence, use_proba=use_proba, generate_sentence=True)
            text_pertinent = sentence_false_pertinents
            processed = self.nlp(unicode(text_pertinent))
            words = [x.text for x in processed]
            positions = [x.idx for x in processed]
        elif pertinents_negatifs_replace:
            sentence_false_pertinents_replace = utils.generate_false_pertinents_replace(
                    text, [], 1, self.neighbors, self.n_best_co_occurrence, use_proba=use_proba, generate_sentence=True)
            text_pertinent = sentence_false_pertinents_replace
            processed = self.nlp(unicode(text_pertinent))
            words = [x.text for x in processed]
            positions = [x.idx for x in processed]
        else:
            processed = self.nlp(unicode(text))
            words = [x.text for x in processed]
            positions = [x.idx for x in processed]

        def sample_fn(present, num_samples, compute_labels=True, pyTorch=False, pertinents_negatifs=False, pertinents_negatifs_replace=False):
            """Generate perturbed samples of the input text.

            Args:
                present: List of word indices that must remain present.
                num_samples: Number of perturbed samples to generate.
                compute_labels: If True, compute model predictions on samples.
                pyTorch: If True, use PyTorch-compatible label computation.
                pertinents_negatifs: If True, use pertinent negative generation.
                pertinents_negatifs_replace: If True, use replacement strategy.

            Returns:
                A tuple of (raw_data, data, labels) or
                (raw_data, data, labels, pertinent_words) when using
                pertinent negatives.
            """
            if self.use_unk_distribution:
                # Replace absent words with a mask
                data = np.ones((num_samples, len(words)))
                raw = np.zeros((num_samples, len(words)), '|S80')
                raw[:] = words
                for i, t in enumerate(words):
                    if i in present:
                        continue
                    n_changed = np.random.binomial(num_samples, .5)
                    changed = np.random.choice(num_samples, n_changed,
                                               replace=False)
                    raw[changed, i] = self.mask_string
                    data[changed, i] = 0
                if (sys.version_info > (3, 0)):
                    raw_data = [' '.join([y.decode() for y in x]) for x in raw]
                else:
                    raw_data = [' '.join(x) for x in raw]
            else:
                if pertinents_negatifs:
                    data, raw_data, sentence_false_pertinents = utils.generate_false_pertinent(
                        text, present, num_samples, self.neighbors, self.n_best_co_occurrence, use_proba=use_proba)
                elif pertinents_negatifs_replace:
                    data, raw_data, sentence_false_pertinents = utils.generate_false_pertinents_replace(
                        text, present, num_samples, self.neighbors, self.n_best_co_occurrence, use_proba=use_proba)
                else:
                    raw_data, data = utils.perturb_sentence(
                        text, present, num_samples, self.neighbors, top_n=100,
                        use_proba=use_proba)
            labels = []
            if true_label == 1:
                true_labels = np.ones(num_samples)
            else:
                true_labels = np.zeros(num_samples)
            compute_label = classifier_fn(raw_data)

            if compute_labels and pyTorch:
                labels = (classifier_fn(raw_data) == true_label).int()
            elif compute_labels:
                labels = compute_label == true_labels
            labels = np.array(labels)
            raw_data = np.array(raw_data).reshape(-1, 1)
            if (pertinents_negatifs or pertinents_negatifs_replace) and not self.use_unk_distribution:
                return raw_data, data, labels, sentence_false_pertinents
            return raw_data, data, labels
        return words, positions, true_label, sample_fn

    def explain_instance(self, text, classifier_fn, threshold=0.95,
                          delta=0.1, tau=0.15, batch_size=100, use_proba=False,
                          beam_size=4, pertinents_negatifs=False, pertinents_negatifs_replace=False,
                          **kwargs):
        """Generate an anchor explanation for a text instance.

        Args:
            text: The text string to explain.
            classifier_fn: Prediction function for the model.
            threshold: Minimum precision threshold for the anchor.
            delta: Confidence parameter for the beam search.
            tau: Tolerance parameter (epsilon) for the beam search.
            batch_size: Number of samples per batch during search.
            use_proba: If True, use probability-weighted word replacement.
            beam_size: Width of the beam search.
            pertinents_negatifs: If True, use pertinent negative generation.
            pertinents_negatifs_replace: If True, use replacement strategy.
            **kwargs: Additional arguments passed to anchor_beam.

        Returns:
            An AnchorExplanation object with the explanation results.
        """
        words, positions, true_label, sample_fn = self.get_sample_fn(
            text, classifier_fn, use_proba=use_proba, pertinents_negatifs=pertinents_negatifs,
            pertinents_negatifs_replace=pertinents_negatifs_replace)
        exp = anchor_base.AnchorBaseBeam.anchor_beam(
            sample_fn, delta=delta, epsilon=tau, batch_size=batch_size,
            desired_confidence=threshold, stop_on_first=True, pertinents_negatifs=pertinents_negatifs,
            pertinents_negatifs_replace=pertinents_negatifs_replace, **kwargs)
        if pertinents_negatifs:
            exp['names'] = ['not ' + words[x] for x in exp['feature']]
        elif pertinents_negatifs_replace:
            exp['names'] = ['not ' + words[x] for x in exp['feature']]
        else:
            exp['names'] = [words[x] for x in exp['feature']]
        exp['positions'] = [positions[x] for x in exp['feature']]
        exp['instance'] = text
        exp['prediction'] = true_label
        explanation = anchor_explanation.AnchorExplanation('text', exp,
                                                           self.as_html)
        return explanation

    def as_html(self, exp):
        """Render a text anchor explanation as a self-contained HTML page.

        Args:
            exp: The raw anchor explanation dictionary.

        Returns:
            A unicode string containing the complete HTML document.
        """
        predict_proba = np.zeros(len(self.class_names))
        exp['prediction'] = int(exp['prediction'])
        predict_proba[exp['prediction']] = 1
        predict_proba = list(predict_proba)

        this_dir, _ = os.path.split(__file__)
        with open(os.path.join(this_dir, 'bundle.js'), encoding='utf8') as f:
            bundle = f.read()
        random_id = 'top_div' + id_generator()

        def process_examples(examples, idx):
            """Process example instances for HTML visualization."""
            idxs = exp['feature'][:idx + 1]
            out_dict = {}
            new_names = {'covered_true': 'coveredTrue', 'covered_false': 'coveredFalse', 'covered': 'covered'}
            for name, new in new_names.items():
                ex = [x[0] for x in examples[name]]
                out = []
                for e in ex:
                    processed = self.nlp(unicode(str(e)))
                    raw_indexes = [(processed[i].text, processed[i].idx, exp['prediction']) for i in idxs]
                    out.append({'text': e, 'rawIndexes': raw_indexes})
                out_dict[new] = out
            return out_dict

        example_obj = []
        for i, examples in enumerate(exp['examples']):
            example_obj.append(process_examples(examples, i))

        explanation = {'names': exp['names'],
                       'certainties': exp['precision'] if len(exp['precision']) else [exp['all_precision']],
                       'supports': exp['coverage'],
                       'allPrecision': exp['all_precision'],
                       'examples': example_obj}
        processed = self.nlp(unicode(exp['instance']))
        raw_indexes = [(processed[i].text, processed[i].idx, exp['prediction'])
                       for i in exp['feature']]
        raw_data = {'text': exp['instance'], 'rawIndexes': raw_indexes}

        # Use json.dumps for all dynamic data to prevent XSS
        out = u'''<html>
        <meta http-equiv="content-type" content="text/html; charset=UTF8">
        <head><script>%s </script></head><body>''' % bundle
        out += u'''
        <div id="{random_id}" />
        <script>
            div = d3.select("#{random_id}");
            lime.RenderExplanationFrame(div,{label_names}, {predict_proba},
            {true_class}, {explanation}, {raw_data}, "text", "anchor");
        </script>'''.format(random_id=random_id,
                            label_names=json.dumps(self.class_names),
                            predict_proba=json.dumps(list(predict_proba)),
                            true_class=json.dumps(False),
                            explanation=json.dumps(explanation),
                            raw_data=json.dumps(raw_data))
        out += u'</body></html>'
        return out

    def show_in_notebook(self, exp, true_class=False, predict_proba_fn=None):
        """Display the explanation in an IPython/Jupyter notebook.

        Args:
            exp: The anchor explanation to display.
            true_class: Whether to show the true class.
            predict_proba_fn: Optional probability prediction function.
        """
        out = self.as_html(exp, true_class, predict_proba_fn)
        from IPython.core.display import display, HTML
        display(HTML(out))
