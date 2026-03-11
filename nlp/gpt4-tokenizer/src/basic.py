from overrides import overrides

from .base import BaseTokenizer

class BasicTokenizer(BaseTokenizer):


    def __count_pairs(self, encoded_text):
        counts = {}
        for pair in zip(encoded_text, encoded_text[1:]):
            counts[pair] = counts.get(pair, 0) + 1

        return counts

    def __merge(self, tokens, pair, merge_token):
        merged_tokens = []
        i = 0

        while i < len(tokens):
            # if we have a match we siwtch the pair tokens with our token
            if i+1 < len(tokens) and pair[0] == tokens[i] and pair[1] == tokens[i+1]:
                merged_tokens.append(merge_token)
                i+=2
            else: # if not just add the tokens as they are
                merged_tokens.append(tokens[i])
                i+=1

        return merged_tokens

    @overrides(check_signature=False)
    def train(self, text:str, vocab_size:int, verbose=False):
        tokens_utf8 = list(text.encode('utf-8'))
        tokens_utf8 = list(map(int, tokens_utf8))

        counter = self.__count_pairs(tokens_utf8)
        most_common_pair = max(counter.keys(), key=counter.get)




    @overrides
    def encode(self, text):
        print(text)

    @overrides
    def decode(self, ids):
        print(ids)