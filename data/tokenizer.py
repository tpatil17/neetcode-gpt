from typing import List


class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        # 1. Split corpus into a list of individual characters
        # 2. For each merge step:
        #    a. Count frequency of all adjacent token pairs
        #    b. Find the most frequent pair (break ties lexicographically)
        #    c. Merge all non-overlapping occurrences left to right
        #    d. Record the merge as [token_a, token_b]
        # 3. Return the list of merges performed
        
        tokens =  [c for c in corpus]
        print(tokens)

        result = []

        for i in range(num_merges):
            store = {} # track the freq of each pair

            ind = 0

            if ind+1 == len(tokens):
                break # not enough tokens to pair
            
            max_pair = (tokens[0], tokens[1])
            max_fq = 1

            while ind+1 < len(tokens):

                pair = (tokens[ind], tokens[ind+1])

                if not pair in store:
                    store[pair] = 1
                else:
                    store[pair]+=1
                
                fq = store[pair]

                if fq > max_fq:
                    max_fq = fq
                    max_pair = pair
                else:
                    if fq == max_fq:
                        if pair < max_pair:
                            max_pair = pair
                
                ind+=1
            
            # merge step
            nind= 0
            buffer = []
            result.append([max_pair[0], max_pair[1]])
            while nind+1 < len(tokens):

                if (tokens[nind], tokens[nind+1]) == max_pair:
                    # merge
                    
                    buffer.append(f"{tokens[nind]}{tokens[nind+1]}")
                    nind += 2 # skip then next token
                else:
                    buffer.append(tokens[nind])
                    nind+=1 # try next token
            
            tokens = buffer # update token with the merged list
        
        return result







        
