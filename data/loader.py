import torch
from torchtyping import TensorType
from typing import Tuple

class Solution:
    def create_batches(self, data: TensorType[int], context_length: int, batch_size: int) -> Tuple[TensorType[int], TensorType[int]]:
        torch.manual_seed(0)

        # Generate all random start indices at once
        max_start = len(data) - context_length
        ix = torch.randint(high=max_start, size=(batch_size,))

        # Stack the slices into 2D tensors of shape (batch_size, context_length)
        x = torch.stack([data[i : i + context_length] for i in ix])
        y = torch.stack([data[i + 1 : i + context_length + 1] for i in ix])

        return x, y
        
