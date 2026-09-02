import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)

        stbl = np.amax(z) # stablizing constant

        sigma = np.exp(z-stbl)

        total = np.sum(sigma)

        return np.round(((sigma)/total), 4)
