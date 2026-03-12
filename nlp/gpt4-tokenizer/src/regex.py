import regex as re
from overrides import overrides
from .base import BaseTokenizer

class RegexTokenizer(BaseTokenizer):
    def __init__(self):
        super().__init__()

        self.regex = re.compile(r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""")

        self.encode_dict = {}
        self.decode_dict = {}

    def __count_pairs(self, encoded_text, stats: dict[tuple[int,int]]):
        counts = {}
        for pair in zip(encoded_text, encoded_text[1:]):
            counts[pair] = counts.get(pair, 0) + 1

        if stats is not None:
            for pair, count in counts.items():
                stats[pair] = stats.get(pair, 0) + count
            return stats

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

        text_splitted = re.findall(self.regex, text)

        tokens_utf8 = [list(text_chunk.encode("utf-8")) for text_chunk in text_splitted]

        merged_tokens = tokens_utf8

        num_merges = vocab_size - 256

        replace_token = 257

        self.decode_dict = {token_id: bytes([token_id]) for token_id in range(256)}

        # TODO - trzeba ogarnac ta funkcyje zeby robila statystyke po wszystkich chunkow i potem laczyla w kazdym chunku odpowiednio pary, nastepnie aktualizowala slownik tak jak ma robic. Jeszcze weryfikacja tego calego workflow na gorze
        for i in range(num_merges):
            counter = self.__count_pairs(merged_tokens)
            most_common_pair = max(counter.keys(), key=counter.get)
            merged_tokens = self.__merge(merged_tokens, most_common_pair, replace_token)

            self.encode_dict[most_common_pair] = replace_token
            self.decode_dict[replace_token] = self.decode_dict[most_common_pair[0]] + self.decode_dict[most_common_pair[1]]

            if verbose:
                print(f"{i+1}/{num_merges} | Merging {most_common_pair} to {replace_token}")

            replace_token+=1


    # TODO - cale to
    @overrides
    def encode(self, text):
        text_tokens = list(text.encode('utf-8'))

        while len(text_tokens) >= 2:
            counter = self.__count_pairs(text_tokens)

            # choose the pair that was merged earliest during training
            # (i.e. the pair with the lowest merge index)
            # we only consider pairs that currently appear in the token sequence
            # this ensures we apply merges in the same order as they were learned during training
            pair = min(counter, key=lambda p: self.encode_dict.get(p, float("inf")))

            if pair not in self.encode_dict:
                break

            text_tokens = self.__merge(text_tokens, pair, self.encode_dict[pair])

        return text_tokens

    # TODO - cale to
    @overrides
    def decode(self, tokens):
        text_bytes = b"".join(self.decode_dict[token_id] for token_id in tokens)
        text_out = text_bytes.decode('utf-8', errors='replace')
        return text_out