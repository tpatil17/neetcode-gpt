from typing import Dict, List, Tuple

class Solution:
    def build_vocab(self, text: str) -> Tuple[Dict[str, int], Dict[int, str]]:
        # Return (stoi, itos) where:
        # - stoi maps each unique character to a unique integer (sorted alphabetically)
        # - itos is the reverse mapping (integer to character)
        stoi = {}
        itos = {}
        split_text = sorted(list(set([c for c in text])))
        for i in range(len(split_text)):
            char = split_text[i]
            if char not in stoi:
                stoi[char] = i
                itos[i] = char
        
        return (stoi, itos)

    def encode(self, text: str, stoi: Dict[str, int]) -> List[int]:
        # Convert a string to a list of integers using stoi mapping
        result = [stoi[c] for c in text]

        return result

    def decode(self, ids: List[int], itos: Dict[int, str]) -> str:
        # Convert a list of integers back to a string using itos mapping
        
        result = [itos[i] for i in ids]

        return "".join(result)
