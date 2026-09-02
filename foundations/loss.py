import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        
        anti_pred = y_pred*(-1) + 1 + 1e-7
        anti_true = y_true*(-1) + 1

        y_pred += 1e-7
        result = y_true*np.log(y_pred) + anti_true*np.log(anti_pred)

        count = len(y_true)

        ans = np.round(-(np.sum(result))/count ,4)

        return ans

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        y_pred += 1e-7
        core = y_true*np.log(y_pred)
        count = len(y_true)
        return np.round(-np.sum(np.sum(core))/count, 4)
        
