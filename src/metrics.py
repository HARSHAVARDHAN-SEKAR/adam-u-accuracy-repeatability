"""
metrics.py -- accuracy and repeatability metrics (numpy only, no simulator).

Definitions follow ISO 9283:1998 for position:

  AP  pose accuracy       distance from the commanded position to the mean
                          (barycentre) of the attained positions
  RP  pose repeatability  l_mean + 3 * S_l, where l_j is the distance of
                          attained position j from the barycentre and S_l the
                          sample standard deviation of the l_j
  vAP multi-directional pose accuracy variation: largest distance between the
                          barycentres reached from different approach directions

ISO 9283 prescribes 30 cycles, 5 poses in a cube and 3 approach directions.
This benchmark uses the same definitions with 10 cycles and 2 approaches, so the
numbers are ISO-style, not ISO-compliant.

Alongside these, four "classic" summaries of the per-trial error are reported --
mean error, RMSE, spread (standard deviation) and success rate.  They answer a
different question: AP and RP describe *where the cluster sits and how tight it
is*, while the classic four describe *the distribution of the error magnitude*.
They are not interchangeable; see the note on mean_error and error_spread.
"""
from __future__ import annotations

import numpy as np


def barycentre(points: np.ndarray) -> np.ndarray:
    return np.asarray(points, dtype=float).mean(axis=0)


def position_accuracy(points: np.ndarray, target: np.ndarray) -> float:
    """AP: |barycentre - target|, same unit as the input."""
    return float(np.linalg.norm(barycentre(points) - np.asarray(target, dtype=float)))


def radii(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    return np.linalg.norm(points - points.mean(axis=0), axis=1)


def position_repeatability(points: np.ndarray) -> float:
    """RP = mean(l) + 3 * std(l), l = distance from the barycentre (needs n >= 2)."""
    l = radii(points)
    if l.size < 2:
        raise ValueError("repeatability needs at least two attained positions")
    return float(l.mean() + 3.0 * l.std(ddof=1))


def axis_std(points: np.ndarray) -> np.ndarray:
    """Sample standard deviation of the attained positions along x, y and z."""
    points = np.asarray(points, dtype=float)
    if points.shape[0] < 2:
        raise ValueError("a standard deviation needs at least two attained positions")
    return points.std(axis=0, ddof=1)


def radius_percentile(points: np.ndarray, q: float = 95.0) -> float:
    """q-th percentile of the distances from the barycentre: a radius that
    contains q % of the attained positions."""
    return float(np.percentile(radii(points), q))


def percentiles(values: np.ndarray, qs=(50.0, 90.0, 95.0)) -> dict:
    """Percentiles of a set of per-trial values, keyed 'p50', 'p90', ..."""
    values = np.asarray(values, dtype=float)
    return {f"p{q:g}": float(np.percentile(values, q)) for q in qs}


def multidirectional_variation(barycentres: np.ndarray) -> float:
    """vAP: largest distance between barycentres of different approaches."""
    b = np.asarray(barycentres, dtype=float)
    if b.shape[0] < 2:
        return 0.0
    diff = b[:, None, :] - b[None, :, :]
    return float(np.linalg.norm(diff, axis=2).max())


def per_trial_error(points: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Distance from the target to each attained position, one value per trial.
    This is the raw quantity the four "classic" metrics below summarise."""
    points = np.asarray(points, dtype=float)
    return np.linalg.norm(points - np.asarray(target, dtype=float), axis=1)


def mean_error(values: np.ndarray) -> float:
    """Average per-trial error. Unlike AP it does not cancel: errors pointing in
    opposite directions still add up, so mean_error >= AP always."""
    return float(np.mean(np.asarray(values, dtype=float)))


def error_spread(values: np.ndarray) -> float:
    """Sample standard deviation of the per-trial errors (ddof=1).  Spread of the
    error *magnitudes*, which is not the same as the spatial scatter RP
    measures: ten points on a sphere around the target all have the same error
    magnitude (spread 0) while being far apart in space."""
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        raise ValueError("spread needs at least two trials")
    return float(values.std(ddof=1))


def rmse(values: np.ndarray) -> float:
    """Root mean square of the per-trial errors. Penalises outliers more than
    the mean does; equals sqrt(mean_error^2 + spread^2 * (n-1)/n)."""
    values = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(values ** 2)))


def success_rate(pos_err: np.ndarray, ori_err: np.ndarray, pos_tol: float, ori_tol: float) -> float:
    ok = (np.asarray(pos_err) <= pos_tol) & (np.asarray(ori_err) <= ori_tol)
    return float(ok.mean())
