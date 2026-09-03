import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        # x: 1D input array
        # weights: list of 2D weight matrices
        # biases: list of 1D bias vectors
        # Apply ReLU after each hidden layer, no activation on output layer
        # return np.round(your_answer, 5)
        outs = []
        
        for i in range(len(weights)):
           
            if outs:
                a_i = outs[-1]
                z_i = a_i @ weights[i] + biases[i]

                if i == len(weights)-1:
                    outs.append(z_i)
                else:
                    y_hat = np.maximum(0, z_i)
                    outs.append(y_hat)
            else:
                z = x @ weights[i] + biases[i]
                outs.append(np.maximum(0, z))
        
        return np.round(outs[-1], 5)