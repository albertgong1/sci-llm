"""Statistical helper functions for scoring.

This module provides utilities for computing statistical metrics
like standard error of the mean (SEM) and formatted statistics strings.
"""

import numpy as np


def sem(x: list, n: int) -> float:
    """Calculate the standard error of the mean.

    Args:
        x: List of sample values
        n: Total number of samples (including missing). If n > len(x),
           missing values are treated as zeros.

    Returns:
        Standard error of the mean

    """
    return np.std(np.concatenate((x, np.zeros(n - len(x)))), ddof=1) / (n**0.5)


def mean_sem_with_n(x: list, n: int) -> str:
    """Format mean +/- SEM as a string.

    Args:
        x: List of sample values
        n: Total number of samples (including missing). If n > len(x),
           missing values are treated as zeros.

    Returns:
        Formatted string "mean +/- sem" with 2 decimal places

    """
    return f"{sum(x) / n:.2f} +/- {sem(x, n):.2f}"
