import numpy as np
from typing import Tuple, List


class Solution:
    def batch_norm(self, x: List[List[float]], gamma: List[float], beta: List[float],
                   running_mean: List[float], running_var: List[float],
                   momentum: float, eps: float, training: bool) -> Tuple[List[List[float]], List[float], List[float]]:
        # During training: normalize using batch statistics, then update running stats
        # During inference: normalize using running stats (no batch stats needed)
        # Apply affine transform: y = gamma * x_hat + beta
        # Return (y, running_mean, running_var), all rounded to 4 decimals as lists

        X = np.array(x, dtype=np.float64)
        gamma_arr = np.array(gamma, dtype=np.float64)
        beta_arr = np.array(beta, dtype=np.float64)
        r_mean = np.array(running_mean, dtype=np.float64)
        r_var = np.array(running_var, dtype=np.float64)
        
        if training:
            # 2. Compute batch statistics along batch axis 0
            mu_b = np.mean(X, axis=0)
            var_b = np.var(X, axis=0)  # ddof=0

            # 3. Normalize batch
            x_hat = (X - mu_b) / np.sqrt(var_b + eps)

            # 4. Update running statistics
            r_mean = (1 - momentum) * r_mean + momentum * mu_b
            r_var = (1 - momentum) * r_var + momentum * var_b
        else:
            # Normalize using running statistics
            x_hat = (X - r_mean) / np.sqrt(r_var + eps)

        # 5. Scale and shift (affine transform)
        Y = gamma_arr * x_hat + beta_arr

        # 6. Format returns as lists rounded to 4 decimals
        return (
            np.round(Y, 4).tolist(),
            np.round(r_mean, 4).tolist(),
            np.round(r_var, 4).tolist()
        )