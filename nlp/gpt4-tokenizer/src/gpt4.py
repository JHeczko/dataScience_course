from overrides import overrides

from .base import Tokenizer

import tiktoken

class GPT4Tokenizer(Tokenizer):

    def __init__(self):
        super().__init__()

    @overrides
    def train(self, text:str, vocab_size:int, verbose=False):
        # it is pretrained
        return

    @overrides
    def encode(self, text):
        pass

    @overrides
    def decode(self, tokens):
        pass