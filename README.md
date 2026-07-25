# Anchors — Anchor-Based Explanations for Machine Learning Models

[![License: BSD](https://img.shields.io/badge/License-BSD-blue.svg)](LICENSE)
[![Python: 3.7+](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/)

An **anchor explanation** is a rule that sufficiently "anchors" the prediction locally — such that changes to the rest of the feature values of the instance do not matter. In other words, for instances on which the anchor holds, the prediction is (almost) always the same.

This repository extends the original [anchor method by Marco Tulio Ribeiro](https://github.com/marcotcr/anchor) with:

- **Multiple discretization strategies** for tabular data (Quartile, Decile, MDLP, Entropy, KMeans)
- **Pertinent negative** generation for textual data (identifying words whose *absence* is important for the prediction)
- **Word replacement** strategies using word embeddings and co-occurrence statistics

> **Paper:** *"When Are Anchors-Based Explanations Not a Good Idea?"* — See the included [PDF](When_are_anchors_based_explanations_not_a_good_idea.pdf) for details.

## Supported Data Types

| Data Type | Explainer Class | Description |
|-----------|----------------|-------------|
| **Tabular** | `AnchorTabularExplainer` | Numerical and categorical features in numpy arrays |
| **Text** | `AnchorText` | Text classification with word-level anchors |
| **Image** | `AnchorImage` | Image classification with superpixel-level anchors |

## Installation

### Requirements

- Python 3.7+
- [spaCy](https://spacy.io/) with a language model (for text explanations)

### Install from source

```bash
git clone https://github.com/j2launay/anchors.git
cd anchors
pip install .
```

### For text explanations

Download a spaCy language model:

```bash
python -m spacy download en_core_web_lg
```

## Quick Start

### Tabular Data

```python
import numpy as np
import sklearn.ensemble
from anchor import anchor_tabular, utils

# Load dataset
dataset = utils.load_dataset('adult', balance=True, dataset_folder='./datasets')

# Train a classifier
clf = sklearn.ensemble.RandomForestClassifier(n_estimators=50, random_state=1)
clf.fit(dataset.train, dataset.labels_train)

# Create explainer
explainer = anchor_tabular.AnchorTabularExplainer(
    class_names=dataset.class_names,
    feature_names=dataset.feature_names,
    train_data=dataset.train,
    categorical_names=dataset.categorical_names
)

# Explain a prediction
idx = 0
instance = dataset.test[idx]
exp = explainer.explain_instance(instance, clf.predict, threshold=0.95)

print('Anchor:', ' AND '.join(exp.names()))
print('Precision: %.2f' % exp.precision())
print('Coverage: %.2f' % exp.coverage())
```

### Text Data

```python
import spacy
from anchor import anchor_text

nlp = spacy.load('en_core_web_lg')
explainer = anchor_text.AnchorText(nlp, ['negative', 'positive'])

text = "This is a good book."
exp = explainer.explain_instance(text, your_classifier_fn, threshold=0.95)

print('Anchor:', ' AND '.join(exp.names()))
print('Precision: %.2f' % exp.precision())
print('Coverage: %.2f' % exp.coverage())
```

## API Reference

### `AnchorExplanation`

The explanation object returned by all explainers provides:

| Method | Description |
|--------|-------------|
| `exp.names(partial_index=None)` | Human-readable anchor conditions |
| `exp.features(partial_index=None)` | Feature indices in the anchor |
| `exp.precision(partial_index=None)` | Anchor precision (fidelity) |
| `exp.coverage(partial_index=None)` | Anchor coverage (generality) |
| `exp.examples(...)` | Covered/uncovered examples |
| `exp.as_html()` | Render as interactive HTML |
| `exp.save_to_file(path)` | Save HTML to file |

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `threshold` | `0.95` | Minimum precision for a valid anchor |
| `delta` | `0.1` | Confidence parameter for beam search |
| `tau` | `0.15` | Tolerance for the LUCB algorithm |
| `beam_size` | `4` | Width of beam search |
| `batch_size` | `100` | Samples per evaluation batch |

## Pertinent Negatives

This extension introduces *pertinent negatives* — words whose **absence** from the anchor is important for the prediction. Enable them with:

```python
exp = explainer.explain_instance(
    text, classifier_fn,
    pertinents_negatifs=True  # Co-occurrence based
)
# or
exp = explainer.explain_instance(
    text, classifier_fn,
    pertinents_negatifs_replace=True  # Embedding-based replacement
)
```

## Security

- **Model loading**: Uses a restricted unpickler that only allows sklearn/numpy classes, preventing arbitrary code execution from pickle deserialization.
- **HTML output**: All dynamic data injected into HTML visualizations is sanitized via `json.dumps()` to prevent XSS.
- **No npm dependencies**: The JavaScript visualization bundle is pre-built. No npm/node runtime is needed.

See [SECURITY.md](SECURITY.md) for the full security policy.

## Project Structure

```
anchors/
├── anchor/                     # Main package
│   ├── anchor_base.py          # Core beam search algorithm (KL-LUCB)
│   ├── anchor_explanation.py   # Explanation result object
│   ├── anchor_tabular.py       # Tabular data explainer
│   ├── anchor_text.py          # Text data explainer
│   ├── anchor_image.py         # Image data explainer
│   ├── utils.py                # Data loading & text perturbation
│   ├── co_occurrence_matrix.py # Co-occurrence statistics
│   ├── generate_dataset.py     # Synthetic dataset generation
│   ├── baseGraph.py            # Experiment visualization
│   └── bundle.js               # Pre-built JS visualization
├── datasets/                   # Example datasets
├── setup.py                    # Package configuration
└── README.md
```

## Citation

If you use this work, please cite:

```bibtex
@article{delaunay2020anchors,
  title={When Are Anchors-Based Explanations Not a Good Idea?},
  author={Delaunay, Julien},
  year={2020}
}
```

## License

This project is licensed under the BSD License — see [LICENSE](LICENSE) for details.

## Acknowledgments

Based on the original [Anchor method](https://github.com/marcotcr/anchor) by Marco Tulio Ribeiro et al.
