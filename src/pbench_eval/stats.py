"""Statistical helper functions for scoring.

This module provides utilities for computing statistical metrics
like standard error of the mean (SEM) and formatted statistics strings.
"""

import numpy as np


def padded_sem(x: list, n: int) -> float:
    """Calculate the standard error of the mean.
    NOTE: treat missing values as zeros.

    Args:
        x: List of sample values
        n: Total number of samples (including missing). If n > len(x),
           missing values are treated as zeros.

    Returns:
        Standard error of the mean

    """
    x = [0.0 if v is None else float(v) for v in x]
    n_eff = max(int(n), len(x))

    if n_eff <= 1:
        return 0.0

    padded = np.concatenate((x, np.zeros(n_eff - len(x))))
    return float(np.std(padded, ddof=1) / (n_eff ** 0.5))


def padded_mean(x: list, n: int) -> float:
    """Calculate the mean, padding with zeros for missing values.

    Args:
        x: List of sample values
        n: Total number of samples (including missing). If n > len(x),
           missing values are treated as zeros.

    Returns:
        Mean value

    """
    x = [0.0 if v is None else float(v) for v in x]
    n_eff = max(int(n), len(x))

    if n_eff == 0:
        return 0.0

    return sum(x) / n_eff


def mean_sem_with_n(x: list, n: int) -> str:
    """Format mean +/- SEM as a string.

    Args:
        x: List of sample values (None values are treated as zeros)
        n: Total number of samples (including missing). If n > len(x),
           missing values are treated as zeros.

    Returns:
        Formatted string "mean +/- sem" with 2 decimal places

    """
    return f"{padded_mean(x, n):.2f} +/- {padded_sem(x, n):.2f}"
