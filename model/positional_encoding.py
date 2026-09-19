import numpy as np
from numpy.typing import NDArray

class Solution:
    def get_positional_encoding(self, seq_len: int, d_model: int) -> NDArray[np.float64]:
        # 1. Create position indices (shape: [seq_len, 1])
        pos = np.arange(seq_len)[:, np.newaxis]
        
        # 2. Create dimension indices for even columns (shape: [d_model / 2])
        i = np.arange(0, d_model, 2)
        
        # 3. Compute division term 10000^(2i / d_model)
        angle_rates = 1 / (10000 ** (i / d_model))
        
        # 4. Multiply positions with angle rates via broadcasting (shape: [seq_len, d_model / 2])
        angle_rads = pos * angle_rates
        
        # 5. Initialize the positional encoding matrix
        PE = np.zeros((seq_len, d_model), dtype=np.float64)
        
        # 6. Assign sine to even indices and cosine to odd indices
        PE[:, 0::2] = np.sin(angle_rads)
        PE[:, 1::2] = np.cos(angle_rads)
        
        # 7. Round to 5 decimal places
        return np.round(PE, 5)
