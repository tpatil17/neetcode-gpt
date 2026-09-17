import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        # 2. Encode each sentence by replacing words with their IDs
        # 3. Combine positive + negative into one list of tensors
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)

        combine = []
        
        for sentence in positive:
            combine.append(sentence.split(" "))
        
        for sentence in negative:
            combine.append(sentence.split(" "))
        
        bag = set()

        for s in combine:
            for word in s:
                bag.add(word)
        
        bag = list(bag)

        bag = sorted(bag)

        store = {}

        for i in range(len(bag)):
            store[bag[i]]= i+1
        
        # word = index dict
        result = []

        for sent in combine:
            buf = []
            for word in sent:
                buf.append(store[word])
            result.append(torch.tensor(buf))
        
        ans = nn.utils.rnn.pad_sequence(result, batch_first=True)

        return ans
        

        
