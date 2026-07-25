"""Anchor explanation object for presenting explanation results.

This module defines the AnchorExplanation class, which provides a
unified interface for accessing anchor explanation properties and
rendering them as HTML.
"""
from __future__ import print_function
import io
import os

os.environ['SPACY_WARNING_IGNORE'] = 'W008'


class AnchorExplanation:
    """Object returned by anchor explainers containing explanation results.

    Provides methods to access the anchor's names, features, precision,
    coverage, and examples, as well as rendering to HTML.

    Args:
        type_: The type of data explained ('text', 'tabular', or 'image').
        exp_map: Dictionary containing the raw explanation data.
        as_html: Function to render the explanation as HTML.
    """
    def __init__(self, type_, exp_map, as_html):
        self.type = type_
        self.exp_map = exp_map
        self.as_html_fn = as_html

    def names(self, partial_index=None):
        """Return human-readable names of the anchor conditions.

        Args:
            partial_index (int, optional): If provided, returns anchor
                conditions only up to this index (inclusive).
                For example, if the anchor is (A=1, B=2, C=2) and
                partial_index=1, returns ["A=1", "B=2"].

        Returns:
            List of condition name strings.
        """
        names = self.exp_map['names']
        if partial_index is not None:
            names = names[:partial_index + 1]
        return names

    def features(self, partial_index=None):
        """Return feature indices used in the anchor conditions.

        Args:
            partial_index (int, optional): If provided, returns feature
                indices only up to this index (inclusive).
                For example, if features are (1, 2, 3) and
                partial_index=1, returns [1, 2].

        Returns:
            List of integer feature indices.
        """
        features = self.exp_map['feature']
        if partial_index is not None:
            features = features[:partial_index + 1]
        return features

    def precision(self, partial_index=None):
        """Return the anchor precision.

        Precision represents the fraction of perturbed instances
        covered by the anchor that receive the same prediction.

        Args:
            partial_index (int, optional): If provided, returns precision
                at the specified partial anchor index.

        Returns:
            float: The precision value.
        """
        precision = self.exp_map['precision']
        if len(precision) == 0:
            return self.exp_map['all_precision']
        if partial_index is not None:
            return precision[partial_index]
        else:
            return precision[-1]

    def coverage(self, partial_index=None):
        """Return the anchor coverage.

        Coverage represents the fraction of instances in the dataset
        that satisfy the anchor conditions.

        Args:
            partial_index (int, optional): If provided, returns coverage
                at the specified partial anchor index.

        Returns:
            float: The coverage value.
        """
        coverage = self.exp_map['coverage']
        if len(coverage) == 0:
            return 1
        if partial_index is not None:
            return coverage[partial_index]
        else:
            return coverage[-1]

    def examples(self, only_different_prediction=False,
                 only_same_prediction=False, partial_index=None):
        """Return examples covered by the anchor.

        Args:
            only_different_prediction (bool): If True, returns only examples
                where the anchor makes a different prediction than the model.
            only_same_prediction (bool): If True, returns only examples
                where the anchor makes the same prediction as the model.
            partial_index (int, optional): If provided, returns examples
                from the partial anchor up to this index.

        Returns:
            List of example instances, or empty list if arguments conflict.
        """
        if only_different_prediction and only_same_prediction:
            print('Error: you can\'t have only_different_prediction '
                  'and only_same_prediction at the same time')
            return []
        key = 'covered'
        if only_different_prediction:
            key = 'covered_false'
        if only_same_prediction:
            key = 'covered_true'
        size = len(self.exp_map['examples'])
        idx = partial_index if partial_index is not None else size - 1
        if idx < 0 or idx > size:
            return []
        return self.exp_map['examples'][idx][key]

    def as_html(self, **kwargs):
        """Render the explanation as an HTML string.

        Args:
            **kwargs: Additional arguments passed to the rendering function.

        Returns:
            str: HTML string containing the rendered explanation.
        """
        return self.as_html_fn(self.exp_map, **kwargs)

    def show_in_notebook(self, **kwargs):
        """Display the explanation in an IPython/Jupyter notebook.

        Args:
            **kwargs: Additional arguments passed to as_html.
        """
        from IPython.core.display import display, HTML
        out = self.as_html(**kwargs)
        display(HTML(out))

    def save_to_file(self, file_path, **kwargs):
        """Save the explanation HTML to a file.

        Args:
            file_path: Path to the output file.
            **kwargs: Additional arguments passed to as_html.
        """
        out = self.as_html(**kwargs)
        with io.open(file_path, 'w', encoding='utf-8') as f:
            f.write(out)
