from typing import List, Dict

class Solution:
    def _tokenize(self, text: str, vocab: Dict[str, int]) -> List[str]:
        tokens = []
        i = 0
        n = len(text)
        
        while i < n:
            match = None
            # Look for the longest substring starting at `i` present in `vocab`
            for j in range(n, i, -1):
                sub = text[i:j]
                if sub in vocab:
                    match = sub
                    break
            
            if match:
                tokens.append(match)
                i += len(match)
            else:
                # If no match is found, consume a single character as a fallback
                tokens.append(text[i])
                i += 1
                
        return tokens

    def tokenize_numbers(self, numbers: List[int], vocab: Dict[str, int]) -> List[List[str]]:
        return [self._tokenize(str(num), vocab) for num in numbers]

    def count_tokens(self, text: str, vocab: Dict[str, int]) -> int:
        # Tokenize the full raw string directly (do NOT use text.split())
        return len(self._tokenize(text, vocab))

    def fertility_score(self, text: str, vocab: Dict[str, int]) -> float:
        words = text.split()
        if not words:
            return 0.0
        
        num_tkns = self.count_tokens(text, vocab)
        return round(num_tkns / len(words), 4)