"""
Transformations for Variance Stabilization.
- Natural Log transformation
- Box-Cox Power transformation
"""

import numpy as np
from scipy import stats

def apply_log_transform(y):
    """Applies natural log to positive series y."""
    min_val = np.min(y)
    offset = 0.0
    if min_val <= 0:
        offset = abs(min_val) + 1.0
    y_log = np.log(y + offset)
    return y_log, offset

def apply_box_cox_transform(y):
    """
    Finds optimal lambda parameter for Box-Cox power transform.
    y(lambda) = (y^lambda - 1) / lambda if lambda != 0, else log(y).
    """
    min_val = np.min(y)
    offset = 0.0
    if min_val <= 0:
        offset = abs(min_val) + 1.0
    y_pos = y + offset
    y_bc, optimal_lambda = stats.boxcox(y_pos)
    return y_bc, optimal_lambda, offset
