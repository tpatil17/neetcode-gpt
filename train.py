import torch
import torch.nn as nn
import torch.nn.functional as F

# The GPT model is provided for you. It returns raw logits (not probabilities).
# You only need to implement the training loop below.

class Solution:
    def train(self, model: nn.Module, data: torch.Tensor, epochs: int, context_length: int, batch_size: int, lr: float) -> float:
        # Train the GPT model using AdamW and cross_entropy loss.
        # For each epoch: seed with torch.manual_seed(epoch),
        # sample batches from data, run forward/backward, update weights.
        # Return the final loss rounded to 4 decimals.
        

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        max_start = len(data) - context_length

        for epoch in range(epochs):
            # 1. Seed per epoch
            torch.manual_seed(epoch)

            # 2. Sample batch
            ix = torch.randint(high=max_start, size=(batch_size,))
            x = torch.stack([data[i : i + context_length] for i in ix])
            y = torch.stack([data[i + 1 : i + 1 + context_length] for i in ix])

            # 3. Forward pass
            logits = model(x)  # Shape: (batch_size, context_length, vocab_size)
            
            # Flatten predictions and targets to compute cross-entropy
            # logits: (B * T, V), y: (B * T)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

            # 4. Backward pass & optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 5. Return final loss rounded to 4 decimals
        return round(loss.item(), 4)