import numpy as np
from typing import List


class Solution:
    def rms_norm(self, x: List[float], gamma: List[float], eps: float) -> List[float]:
        # Implement RMS Normalization (similar to LayerNorm but without mean centering or beta)
        # Normalize x, then scale by gamma
        # Return result rounded to 4 decimal places as a list
        
        X = np.array(x) 

        Gma = np.array(gamma)

        rms = np.sqrt((np.mean(X**2)) +eps)

        print(rms)
        x_hat = x/rms
        print(x_hat)
        out = np.round(Gma*x_hat, 4)

        print(out)

        return list(out)