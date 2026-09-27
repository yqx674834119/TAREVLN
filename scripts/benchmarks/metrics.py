"""R2R-CE metric semantics, independent from the policy interface."""
import math
import numpy as np


def deduplicate(positions):
    result = []
    for p in positions:
        if not result or list(p) != list(result[-1]):
            result.append(p)
    return result


def exact_dtw(a, b):
    previous = np.full(len(b)+1, np.inf)
    previous[0] = 0
    for p in a:
        current = np.full(len(b)+1, np.inf)
        for j, q in enumerate(b, 1):
            current[j] = np.linalg.norm(np.asarray(p)-q) + min(previous[j], current[j-1], previous[j-1])
        previous = current
    return float(previous[-1])


def evaluate(positions, distances, stopped, initial_distance, ground_truth=None):
    success = float(stopped and distances[-1] < 3.0)
    length = float(sum(np.linalg.norm(np.asarray(b)-a) for a, b in zip(positions, positions[1:])))
    result = dict(ne_m=float(distances[-1]), sr=success,
                  osr=float(min(distances) < 3.0), path_length_m=length,
                  spl=success * initial_distance/max(initial_distance, length, 1e-12))
    if ground_truth is not None:
        actual = deduplicate(positions)
        result['ndtw_exact'] = math.exp(-exact_dtw(actual, ground_truth)/(len(ground_truth)*3.0))
        result['sdtw_exact'] = success*result['ndtw_exact']
        try:
            from fastdtw import fastdtw
            d = fastdtw(actual, ground_truth, dist=lambda a, b: np.linalg.norm(np.asarray(a)-b))[0]
            result['ndtw_fast'] = math.exp(-d/(len(ground_truth)*3.0))
            result['sdtw_fast'] = success*result['ndtw_fast']
        except ImportError:
            pass
    return result
