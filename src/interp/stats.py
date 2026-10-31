"""Shared statistics for the interpretability analyses: subject-level bootstrap via per-subject weights (fast, exact
resampling of subjects), Spearman correlation and AUROC with 95 % CIs."""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

N_BOOT, SEED = 1000, 2026


def subject_weights(groups, n=N_BOOT, seed=SEED):
    """Yield per-row weights of n subject-level bootstrap resamples (each subject drawn with multinomial counts)."""
    codes, uniq = pd.factorize(np.asarray(groups))
    k = len(uniq)
    rng = np.random.default_rng(seed)
    for _ in range(n):
        yield rng.multinomial(k, np.full(k, 1.0 / k)).astype(float)[codes]


def spearman_ci(score, target, groups, n=N_BOOT, seed=SEED):
    rx, ry = rankdata(score), rankdata(target)
    def wcorr(w):
        mx, my = np.average(rx, weights=w), np.average(ry, weights=w)
        return np.average((rx - mx) * (ry - my), weights=w) / np.sqrt(np.average((rx - mx) ** 2, weights=w) * np.average((ry - my) ** 2, weights=w))
    point = wcorr(np.ones_like(rx))
    return point, np.percentile([wcorr(w) for w in subject_weights(groups, n, seed)], [2.5, 97.5])


def auc_ci(y, score, groups, n=N_BOOT, seed=SEED):
    y, score = np.asarray(y).astype(int), np.asarray(score, float)
    point = roc_auc_score(y, score)
    vals = []
    for w in subject_weights(groups, n, seed):
        keep = w > 0
        if y[keep].min() == y[keep].max():
            continue
        vals.append(roc_auc_score(y[keep], score[keep], sample_weight=w[keep]))
    return point, np.percentile(vals, [2.5, 97.5])
