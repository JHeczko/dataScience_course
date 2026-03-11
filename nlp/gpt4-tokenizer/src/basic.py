from overrides import overrides

from .base import BaseTokenizer

class BasicTokenizer(BaseTokenizer):
    def __init__(self):
        super().__init__()

        self.encode_dict = {}
        self.decode_dict = {}

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
        assert vocab_size >= 256

        tokens_utf8 = list(text.encode('utf-8'))
        tokens_utf8 = list(map(int, tokens_utf8))

        merged_tokens = tokens_utf8

        num_merges = vocab_size - 256

        replace_token = 257

        self.decode_dict = {token_id: bytes([token_id]) for token_id in range(256)}

        for i in range(num_merges):
            counter = self.__count_pairs(merged_tokens)
            most_common_pair = max(counter.keys(), key=counter.get)
            merged_tokens = self.__merge(merged_tokens, most_common_pair, replace_token)

            self.encode_dict[most_common_pair] = replace_token
            self.decode_dict[replace_token] = self.decode_dict[most_common_pair[0]] + self.decode_dict[most_common_pair[1]]

            if verbose:
                print(f"{i}/{num_merges} | Merging {most_common_pair} to {replace_token}")

            replace_token+=1

        return merged_tokens


    @overrides
    def encode(self, text):
        print(text)

    @overrides
    def decode(self, tokens):
        text_bytes = b"".join(self.decode_dict[token_id] for token_id in tokens)
        text_out = text_bytes.decode('utf-8')
        return text_out