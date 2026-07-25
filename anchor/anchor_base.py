"""Core anchor beam search algorithm.

This module implements the anchor beam search procedure that finds
anchor explanations — minimal sets of feature predicates that
"anchor" a model's prediction with high confidence.

The main class AnchorBaseBeam implements the KL-LUCB bandit algorithm
for efficient exploration of candidate anchors.
"""
from __future__ import print_function
import collections
import copy
import sys

import numpy as np
import sklearn


def matrix_subset(matrix, n_samples):
    """Return a random subset of rows from a matrix.

    Args:
        matrix: A 2-D numpy array.
        n_samples: Maximum number of rows to sample.

    Returns:
        A numpy array containing at most n_samples randomly selected rows.
    """
    if matrix.shape[0] == 0:
        return matrix
    n_samples = min(matrix.shape[0], n_samples)
    return matrix[np.random.choice(matrix.shape[0], n_samples, replace=False)]


class AnchorBaseBeam(object):
    """Anchor beam search algorithm using KL-LUCB bandits.

    This class implements the core anchor-finding algorithm. It uses
    beam search to explore combinations of feature predicates and
    KL-LUCB (Lower and Upper Confidence Bound) to efficiently evaluate
    candidate anchors with minimal samples.
    """

    def __init__(self):
        pass

    @staticmethod
    def kl_bernoulli(p, q):
        """Compute the KL divergence between two Bernoulli distributions.

        Args:
            p: Parameter of the first Bernoulli distribution.
            q: Parameter of the second Bernoulli distribution.

        Returns:
            float: KL(Bernoulli(p) || Bernoulli(q)).
        """
        p = min(0.9999999999999999, max(0.0000001, p))
        q = min(0.9999999999999999, max(0.0000001, q))
        return (p * np.log(float(p) / q) + (1 - p) *
                np.log(float(1 - p) / (1 - q)))

    @staticmethod
    def dup_bernoulli(p, level):
        """Compute the upper confidence bound for a Bernoulli mean.

        Uses binary search to find the largest q such that
        KL(p, q) <= level.

        Args:
            p: Observed Bernoulli mean.
            level: Confidence level parameter (beta / n_samples).

        Returns:
            float: Upper confidence bound on the true mean.
        """
        lm = p
        um = min(min(1, p + np.sqrt(level / 2.)), 1)
        for j in range(1, 17):
            qm = (um + lm) / 2.
            if AnchorBaseBeam.kl_bernoulli(p, qm) > level:
                um = qm
            else:
                lm = qm
        return um

    @staticmethod
    def dlow_bernoulli(p, level):
        """Compute the lower confidence bound for a Bernoulli mean.

        Uses binary search to find the smallest q such that
        KL(p, q) <= level.

        Args:
            p: Observed Bernoulli mean.
            level: Confidence level parameter (beta / n_samples).

        Returns:
            float: Lower confidence bound on the true mean.
        """
        um = p
        lm = max(min(1, p - np.sqrt(level / 2.)), 0)
        for j in range(1, 17):
            qm = (um + lm) / 2.
            if AnchorBaseBeam.kl_bernoulli(p, qm) > level:
                lm = qm
            else:
                um = qm
        return lm

    @staticmethod
    def compute_beta(n_features, t, delta):
        """Compute the exploration parameter beta for the LUCB algorithm.

        Args:
            n_features: Number of candidate features/arms.
            t: Current round number.
            delta: Confidence parameter.

        Returns:
            float: Beta value for confidence bound computation.
        """
        alpha = 1.1
        k = 405.5
        temp = np.log(k * n_features * (t ** alpha) / delta)
        return temp + np.log(temp)

    @staticmethod
    def lucb(sample_fns, initial_stats, epsilon, delta, batch_size, top_n,
             verbose=False, verbose_every=1):
        """Run the KL-LUCB algorithm to identify the top-n arms.

        Implements the Lower and Upper Confidence Bound algorithm
        to find the top_n arms (candidate anchors) with highest
        mean reward using adaptive sampling.

        Args:
            sample_fns: List of sampling functions, one per arm.
            initial_stats: Dict with 'n_samples' and 'positives' lists.
            epsilon: Tolerance parameter for stopping.
            delta: Confidence parameter.
            batch_size: Number of samples per evaluation round.
            top_n: Number of top arms to identify.
            verbose: If True, print progress information.
            verbose_every: Print every N iterations when verbose.

        Returns:
            numpy array of indices of the top_n arms.
        """
        n_features = len(sample_fns)
        n_samples = np.array(initial_stats['n_samples'])
        positives = np.array(initial_stats['positives'])
        ub = np.zeros(n_samples.shape)
        lb = np.zeros(n_samples.shape)
        for f in np.where(n_samples == 0)[0]:
            n_samples[f] += 1
            positives[f] += sample_fns[f](1)
        if n_features == top_n:
            return range(n_features)
        means = positives / n_samples
        t = 1

        def update_bounds(t):
            sorted_means = np.argsort(means)
            beta = AnchorBaseBeam.compute_beta(n_features, t, delta)
            J = sorted_means[-top_n:]
            not_J = sorted_means[:-top_n]
            for f in not_J:
                ub[f] = AnchorBaseBeam.dup_bernoulli(means[f], beta /
                                                     n_samples[f])
            for f in J:
                lb[f] = AnchorBaseBeam.dlow_bernoulli(means[f],
                                                      beta / n_samples[f])
            ut = not_J[np.argmax(ub[not_J])]
            lt = J[np.argmin(lb[J])]
            return ut, lt
        ut, lt = update_bounds(t)
        B = ub[ut] - lb[lt]
        verbose_count = 0
        while B > epsilon:
            verbose_count += 1
            if verbose and verbose_count % verbose_every == 0:
                print('Best: %d (mean:%.10f, n: %d, lb:%.4f)' %
                      (lt, means[lt], n_samples[lt], lb[lt]), end=' ')
                print('Worst: %d (mean:%.4f, n: %d, ub:%.4f)' %
                      (ut, means[ut], n_samples[ut], ub[ut]), end=' ')
                print('B = %.2f' % B)
            n_samples[ut] += batch_size
            positives[ut] += sample_fns[ut](batch_size)
            means[ut] = positives[ut] / n_samples[ut]
            n_samples[lt] += batch_size
            positives[lt] += sample_fns[lt](batch_size)
            means[lt] = positives[lt] / n_samples[lt]
            t += 1
            ut, lt = update_bounds(t)
            B = ub[ut] - lb[lt]
        sorted_means = np.argsort(means)
        return sorted_means[-top_n:]

    @staticmethod
    def make_tuples(previous_best, state):
        """Generate candidate anchor tuples by extending previous best.

        Extends the best anchors from the previous beam search round
        by adding one new feature at a time, computing coverage and
        precision statistics for each new candidate.

        Args:
            previous_best: List of tuples representing the best anchors
                from the previous round.
            state: Dict containing search state (data, labels, coverage, etc.).

        Returns:
            List of new candidate anchor tuples.
        """
        normalize_tuple = lambda x: tuple(sorted(set(x)))  # noqa
        all_features = range(state['n_features'])
        coverage_data = state['coverage_data']
        current_idx = state['current_idx']
        data = state['data'][:current_idx]
        labels = state['labels'][:current_idx]
        # if we want false pertinents at first
        if len(previous_best) == 0:
            tuples = [(x, ) for x in all_features]
            for x in tuples:
                pres = data[:, x[0]].nonzero()[0]
                state['t_idx'][x] = set(pres)
                state['t_nsamples'][x] = float(len(pres))
                state['t_positives'][x] = float(labels[pres].sum())
                state['t_order'][x].append(x[0])
                # Compute coverage based on the "presence" or absence of an index in column x from data
                state['t_coverage_idx'][x] = set(
                    coverage_data[:, x[0]].nonzero()[0])
                state['t_coverage'][x] = (
                    float(len(state['t_coverage_idx'][x])) /
                    coverage_data.shape[0])
            return tuples
        new_tuples = set()
        for f in all_features:
            for t in previous_best:
                new_t = normalize_tuple(t + (f, ))
                if len(new_t) != len(t) + 1:
                    continue
                if new_t not in new_tuples:
                    new_tuples.add(new_t)
                    state['t_order'][new_t] = copy.deepcopy(state['t_order'][t])
                    state['t_order'][new_t].append(f)
                    state['t_coverage_idx'][new_t] = (
                        state['t_coverage_idx'][t].intersection(
                            state['t_coverage_idx'][(f,)]))
                    state['t_coverage'][new_t] = (
                        float(len(state['t_coverage_idx'][new_t])) /
                        coverage_data.shape[0])
                    t_idx = np.array(list(state['t_idx'][t]))
                    t_data = state['data'][t_idx]
                    present = np.where(t_data[:, f] == 1)[0]
                    state['t_idx'][new_t] = set(t_idx[present])
                    idx_list = list(state['t_idx'][new_t])
                    state['t_nsamples'][new_t] = float(len(idx_list))
                    state['t_positives'][new_t] = np.sum(
                        state['labels'][idx_list])
        return list(new_tuples)

    @staticmethod
    def make_tuples_negatifs(previous_best, state):
        """Generate candidate tuples for pertinent negative search.

        Similar to make_tuples but designed for pertinent negative
        anchor generation. Extends previous best anchors by adding
        one new feature at a time.

        Args:
            previous_best: List of tuples from the previous round.
            state: Dict containing search state.

        Returns:
            List of new candidate anchor tuples.
        """
        normalize_tuple = lambda x: tuple(sorted(set(x)))  # noqa
        all_features = range(state['n_features'])
        coverage_data = state['coverage_data']
        current_idx = state['current_idx']
        data = state['data'][:current_idx]
        labels = state['labels'][:current_idx]
        if len(previous_best) == 0:
            tuples = [(x, ) for x in all_features]
            for x in tuples:
                pres = data[:, x[0]].nonzero()[0]
                state['t_idx'][x] = set(pres)
                state['t_nsamples'][x] = float(len(pres))
                state['t_positives'][x] = float(labels[pres].sum())
                state['t_order'][x].append(x[0])
                state['t_coverage_idx'][x] = set(
                    coverage_data[:, x[0]].nonzero()[0])
                state['t_coverage'][x] = (
                    float(len(state['t_coverage_idx'][x])) /
                    coverage_data.shape[0])
            return tuples
        new_tuples = set()
        for f in all_features:
            for t in previous_best:
                new_t = normalize_tuple(t + (f, ))
                if len(new_t) != len(t) + 1:
                    continue
                if new_t not in new_tuples:
                    new_tuples.add(new_t)
                    state['t_order'][new_t] = copy.deepcopy(state['t_order'][t])
                    state['t_order'][new_t].append(f)
                    state['t_coverage_idx'][new_t] = (
                        state['t_coverage_idx'][t].intersection(
                            state['t_coverage_idx'][(f,)]))
                    state['t_coverage'][new_t] = (
                        float(len(state['t_coverage_idx'][new_t])) /
                        coverage_data.shape[0])
                    t_idx = np.array(list(state['t_idx'][t]))
                    t_data = state['data'][t_idx]
                    present = np.where(t_data[:, f] == 1)[0]
                    state['t_idx'][new_t] = set(t_idx[present])
                    idx_list = list(state['t_idx'][new_t])
                    state['t_nsamples'][new_t] = float(len(idx_list))
                    state['t_positives'][new_t] = np.sum(
                        state['labels'][idx_list])
        return list(new_tuples)

    @staticmethod
    def get_sample_fns(sample_fn, tuples, state, negatif=False, negatif_replace=False):
        """Create sampling functions for each candidate anchor tuple.

        Wraps the base sample_fn to track statistics for each candidate
        anchor tuple during the beam search.

        Args:
            sample_fn: The base sampling function from the explainer.
            tuples: List of candidate anchor tuples.
            state: Dict containing search state.
            negatif: If True, use pertinent negative sampling.
            negatif_replace: If True, use pertinent negative with replacement.

        Returns:
            List of sampling closures, one per candidate tuple.
        """
        sample_fns = []

        def complete_sample_fn(t, n):
            if negatif:
                raw_data, data, labels, _ = sample_fn(list(t), n, pertinents_negatifs=negatif)
            elif negatif_replace:
                raw_data, data, labels, _ = sample_fn(list(t), n, pertinents_negatifs_replace=negatif_replace)
            else:
                raw_data, data, labels = sample_fn(list(t), n)
            current_idx = state['current_idx']
            idxs = range(current_idx, current_idx + n)
            state['t_idx'][t].update(idxs)
            state['t_nsamples'][t] += n
            state['t_positives'][t] += labels.sum()
            state['data'][idxs] = data
            state['raw_data'][idxs] = raw_data
            state['labels'][idxs] = labels
            state['current_idx'] += n
            if state['current_idx'] >= state['data'].shape[0] - max(1000, n):
                prealloc_size = state['prealloc_size']
                current_idx = data.shape[0]
                state['data'] = np.vstack(
                    (state['data'],
                     np.zeros((prealloc_size, data.shape[1]), data.dtype)))
                state['raw_data'] = np.vstack(
                    (state['raw_data'],
                     np.zeros((prealloc_size, raw_data.shape[1]),
                              raw_data.dtype)))
                state['labels'] = np.hstack(
                    (state['labels'],
                     np.zeros(prealloc_size, labels.dtype)))
            return labels.sum()
        for t in tuples:
            sample_fns.append(lambda n, t=t: complete_sample_fn(t, n))
        return sample_fns


    @staticmethod
    def get_initial_statistics(tuples, state):
        """Collect initial sampling statistics for candidate tuples.

        Args:
            tuples: List of candidate anchor tuples.
            state: Dict containing search state.

        Returns:
            Dict with 'n_samples' and 'positives' lists for each tuple.
        """
        stats = {
            'n_samples': [],
            'positives': []
        }
        for t in tuples:
            stats['n_samples'].append(state['t_nsamples'][t])
            stats['positives'].append(state['t_positives'][t])
        return stats

    @staticmethod
    def get_anchor_from_tuple(t, state):
        """Convert a winning tuple into a structured anchor explanation.

        Builds the complete anchor explanation dictionary including
        intermediate precision/coverage values and example instances.

        Args:
            t: The winning anchor tuple (sequence of feature indices).
            state: Dict containing search state with all sampled data.

        Returns:
            Dict containing the anchor explanation with keys: feature,
            mean, precision, coverage, examples, all_precision, num_preds.
        """
        anchor = {'feature': [], 'mean': [], 'precision': [],
                  'coverage': [], 'examples': [], 'all_precision': 0}
        anchor['num_preds'] = state['data'].shape[0]
        normalize_tuple = lambda x: tuple(sorted(set(x)))  # noqa
        current_t = tuple()
        for f in state['t_order'][t]:
            current_t = normalize_tuple(current_t + (f,))

            mean = (state['t_positives'][current_t] /
                    state['t_nsamples'][current_t])
            anchor['feature'].append(f)
            anchor['mean'].append(mean)
            anchor['precision'].append(mean)
            anchor['coverage'].append(state['t_coverage'][current_t])
            # Store the id where the current_t features are present
            raw_idx = list(state['t_idx'][current_t])
            # Store all the sentences generated that have the current_t feature presents
            raw_data = state['raw_data'][raw_idx]
            # Store all the instances that have the current feature and a positive label prediction
            covered_true = (
                state['raw_data'][raw_idx][state['labels'][raw_idx] == 1])
            covered_false = (
                state['raw_data'][raw_idx][state['labels'][raw_idx] == 0])
            exs = {}
            exs['covered'] = matrix_subset(raw_data, 10)
            exs['covered_true'] = matrix_subset(covered_true, 10)
            exs['covered_false'] = matrix_subset(covered_false, 10)
            exs['uncovered_true'] = np.array([])
            exs['uncovered_false'] = np.array([])
            anchor['examples'].append(exs)
        return anchor

    @staticmethod
    def anchor_beam(sample_fn, delta=0.05, epsilon=0.1, batch_size=10,
                    min_shared_samples=0, desired_confidence=1, beam_size=1,
                    verbose=False, epsilon_stop=0.05, min_samples_start=0,
                    max_anchor_size=None, verbose_every=1,
                    stop_on_first=False, pertinents_negatifs=False, pertinents_negatifs_replace=False, coverage_samples=10000):
        """Run the anchor beam search algorithm.

        This is the main entry point for finding anchor explanations.
        It performs beam search over feature combinations, using the
        KL-LUCB algorithm to efficiently evaluate each candidate.

        Args:
            sample_fn: Function that generates perturbed samples.
                Signature: sample_fn(present, n, compute_labels=True) ->
                (raw_data, data, labels).
            delta: Confidence parameter (probability of error).
            epsilon: Tolerance for the LUCB algorithm.
            batch_size: Number of samples per evaluation batch.
            min_shared_samples: Minimum samples shared across candidates.
            desired_confidence: Target precision threshold.
            beam_size: Width of the beam search.
            verbose: If True, print progress information.
            epsilon_stop: Tolerance for stopping condition.
            min_samples_start: Minimum initial samples.
            max_anchor_size: Maximum number of features in the anchor.
            verbose_every: Print every N iterations when verbose.
            stop_on_first: If True, stop at the first valid anchor found.
            pertinents_negatifs: If True, use pertinent negative generation.
            pertinents_negatifs_replace: If True, use replacement strategy.
            coverage_samples: Number of samples for coverage estimation.

        Returns:
            Dict containing the anchor explanation with keys: feature,
            mean, precision, coverage, examples, all_precision, num_preds.
        """
        anchor = {'feature': [], 'mean': [], 'precision': [],
                  'coverage': [], 'examples': [], 'all_precision': 0}
        # Coverage data correspond to a matrix that map instances generated by sample_fn (of size coverage_samples)
        # This map associated a point to an interval
        if pertinents_negatifs:
            _, coverage_data, _, _ = sample_fn([], coverage_samples, compute_labels=False, pertinents_negatifs=pertinents_negatifs)
        elif pertinents_negatifs_replace:
            _, coverage_data, _, _ = sample_fn([], coverage_samples, compute_labels=False, pertinents_negatifs_replace=pertinents_negatifs_replace)
        else:
            _, coverage_data, _ = sample_fn([], coverage_samples, compute_labels=False)
        if pertinents_negatifs:
            raw_data, data, labels, _ = sample_fn([], max(1, min_samples_start), pertinents_negatifs=pertinents_negatifs)
        elif pertinents_negatifs_replace:
            raw_data, data, labels, _ = sample_fn([], max(1, min_samples_start), pertinents_negatifs_replace=pertinents_negatifs_replace)
        else:
            raw_data, data, labels = sample_fn([], max(1, min_samples_start))
        mean = labels.mean()
        beta = np.log(1. / delta)
        lb = AnchorBaseBeam.dlow_bernoulli(mean, beta / data.shape[0])
        if lb > desired_confidence:
            anchor['num_preds'] = data.shape[0]
            anchor['all_precision'] = mean
            return anchor
        prealloc_size = batch_size * 10000
        current_idx = data.shape[0]
        # Stack initial data with pre-allocated zero arrays
        data = np.vstack((data, np.zeros((prealloc_size, data.shape[1]),
                                         data.dtype)))
        raw_data = np.vstack(
            (raw_data, np.zeros((prealloc_size, raw_data.shape[1]),
                                raw_data.dtype)))
        labels = np.hstack((labels, np.zeros(prealloc_size, labels.dtype)))
        n_features = data.shape[1]
        state = {'t_idx': collections.defaultdict(lambda: set()),
                 't_nsamples': collections.defaultdict(lambda: 0.),
                 't_positives': collections.defaultdict(lambda: 0.),
                 'data': data,
                 'prealloc_size': prealloc_size,
                 'raw_data': raw_data,
                 'labels': labels,
                 'current_idx': current_idx,
                 'n_features': n_features,
                 't_coverage_idx': collections.defaultdict(lambda: set()),
                 't_coverage': collections.defaultdict(lambda: 0.),
                 'coverage_data': coverage_data,
                 't_order': collections.defaultdict(lambda: list())
                 }
        current_size = 1
        best_of_size = {0: []}
        best_coverage = -1
        best_tuple = ()
        t = 1
        if max_anchor_size is None:
            max_anchor_size = n_features
        while current_size <= max_anchor_size:
            # Add to "state" informations about coverage and return tuples
            tuples = AnchorBaseBeam.make_tuples(
                best_of_size[current_size - 1], state)
            # Check for indices in tuples that have only the best coverage
            tuples = [x for x in tuples
                      if state['t_coverage'][x] > best_coverage]
            if len(tuples) == 0:
                break
            sample_fns = AnchorBaseBeam.get_sample_fns(sample_fn, tuples,
                                                       state, negatif=pertinents_negatifs,
                                                       negatif_replace=pertinents_negatifs_replace)
            initial_stats = AnchorBaseBeam.get_initial_statistics(tuples,
                                                                  state)
            chosen_tuples = AnchorBaseBeam.lucb(
                sample_fns, initial_stats, epsilon, delta, batch_size,
                min(beam_size, len(tuples)),
                verbose=verbose, verbose_every=verbose_every)
            best_of_size[current_size] = [tuples[x] for x in chosen_tuples]
            if verbose:
                print('Best of size ', current_size, ':')
            stop_this = False
            for i, t in zip(chosen_tuples, best_of_size[current_size]):
                beta = np.log(1. /
                              (delta / (1 + (beam_size - 1) * n_features)))
                mean = state['t_positives'][t] / state['t_nsamples'][t]
                lb = AnchorBaseBeam.dlow_bernoulli(
                    mean, beta / state['t_nsamples'][t])
                ub = AnchorBaseBeam.dup_bernoulli(
                    mean, beta / state['t_nsamples'][t])
                coverage = state['t_coverage'][t]
                if verbose:
                    print(i, mean, lb, ub)
                while ((mean >= desired_confidence and
                       lb < desired_confidence - epsilon_stop) or
                       (mean < desired_confidence and
                        ub >= desired_confidence + epsilon_stop)):
                    sample_fns[i](batch_size)
                    mean = state['t_positives'][t] / state['t_nsamples'][t]
                    lb = AnchorBaseBeam.dlow_bernoulli(
                        mean, beta / state['t_nsamples'][t])
                    ub = AnchorBaseBeam.dup_bernoulli(
                        mean, beta / state['t_nsamples'][t])
                if verbose:
                    print('%s mean = %.2f lb = %.2f ub = %.2f coverage: %.2f n: %d' % (t, mean, lb, ub, coverage, state['t_nsamples'][t]))
                if mean >= desired_confidence and lb > desired_confidence - epsilon_stop:
                    if verbose:
                        print('Found eligible anchor ', t, 'Coverage:',
                              coverage, 'Is best?', coverage > best_coverage)
                    if coverage > best_coverage:
                        best_coverage = coverage
                        best_tuple = t
                        if best_coverage == 1 or stop_on_first:
                            stop_this = True
            if stop_this:
                break
            current_size += 1
        if best_tuple == ():
            # Could not find an anchor, will now choose the highest precision
            # amongst the top K from every round
            if verbose:
                print('Could not find an anchor, now doing best of each size')
            tuples = []
            for i in range(0, current_size):
                tuples.extend(best_of_size[i])
            sample_fns = AnchorBaseBeam.get_sample_fns(sample_fn, tuples,
                                                       state, negatif=pertinents_negatifs,
                                                       negatif_replace=pertinents_negatifs_replace)
            initial_stats = AnchorBaseBeam.get_initial_statistics(tuples,
                                                                  state)
            chosen_tuples = AnchorBaseBeam.lucb(
                sample_fns, initial_stats, epsilon, delta, batch_size,
                1, verbose=verbose)
            best_tuple = tuples[chosen_tuples[0]]
        return AnchorBaseBeam.get_anchor_from_tuple(best_tuple, state)