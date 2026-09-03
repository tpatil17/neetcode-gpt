import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)

        x_arr = np.array(x)
        W1_arr = np.array(W1)
        b1_arr = np.array(b1)
        W2_arr = np.array(W2)
        b2_arr = np.array(b2)
        y_true_arr = np.array(y_true)

        ans = {} 

        z1 = x @ W1_arr.T + b1_arr

        a1 = np.maximum(0, z1)
        
        y_hat2= a1 @ W2_arr.T + b2_arr #pred

        Loss = np.mean((y_hat2 - y_true_arr)**2)

        n = len(y_true_arr)
        dL_dz2 = 2 * (y_hat2 - y_true_arr) / n

        # Layer 2 gradients
        dW2 = np.outer(dL_dz2, a1)
        db2 = dL_dz2

        # Hidden layer gradient (routing through W2 and ReLU)
        dL_da1 = dL_dz2 @ W2_arr
        dL_dz1 = dL_da1 * (z1 > 0)

        # Layer 1 gradients
        dW1 = np.outer(dL_dz1, x_arr)
        db1 = dL_dz1

        # 4. Format output dictionary
        return {
            'loss': round(float(Loss), 4),
            'dW1': np.round(dW1, 4).tolist(),
            'db1': np.round(db1, 4).tolist(),
            'dW2': np.round(dW2, 4).tolist(),
            'db2': np.round(db2, 4).tolist()
        }

        

