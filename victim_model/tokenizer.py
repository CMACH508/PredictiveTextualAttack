import json
import os

import transformers
from nltk.tokenize import word_tokenize


class GloveTokenizer(object):
    def __init__(self, tokenizer_path):
        config_path = os.path.join(tokenizer_path, 'tokenizer_config.json')
        vocab_path = os.path.join(tokenizer_path, 'vocab.txt')
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"File not found: {config_path}")
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"File not found: {vocab_path}")
        with open(config_path, 'r', encoding='utf8') as f:
            self.config = json.load(f)
        with open(vocab_path, 'r', encoding='utf8') as f:
            self.id2word = [line.strip() for line in f]
            self.word2id = {w: i for i, w in enumerate(self.id2word)}

    def encode(self, text):
        if isinstance(text, str):
            tokens = word_tokenize(text.lower())
            tokens = [self.word2id.get(t, self.config['unk_token_id']) for t in tokens]
            if 'max_length' in self.config:
                tokens = tokens[:self.config['max_length']]
        elif isinstance(text, (list, tuple)) and len(text) == 2:
            tokens = (self.encode(text[0]), self.encode(text[1]))
        else:
            raise TypeError('GloveTokenizer.encode() arg `text` must be a str or a str pair')
        return tokens

    def batch_encode(self, texts):
        if not isinstance(texts, (tuple, list)):
            raise TypeError('GloveTokenizer.batch_encode() arg `texts` must be a list or a tuple.')
        if all(isinstance(x, str) for x in texts):
            result = list(map(self.encode, texts))
            max_length = max(len(tokens) for tokens in result)
            result = [tokens + [self.config['pad_token_id']] * (max_length - len(tokens))
                      for tokens in result]
        elif all(isinstance(x, (list, tuple)) and len(x) == 2 for x in texts):
            result = list(zip(self.batch_encode([t[0] for t in texts]), self.batch_encode([t[1] for t in texts])))
        else:
            raise TypeError('GloveTokenizer.batch_encode() arg `texts` must keep all elements of the same length.')
        return result


class TransformerTokenizer(object):
    """A generic class that convert text to tokens and tokens to IDs. Supports
    any type of tokenization, be it word, wordpiece, or character-based. Based
    on the ``AutoTokenizer`` from the ``transformers`` library, but
    standardizes the functionality for TextAttack.

    Args:
        tokenizer_path: the identifying name of the tokenizer, for example, ``bert-base-uncased``
            (see AutoTokenizer,
            https://github.com/huggingface/transformers/blob/master/src/transformers/tokenization_auto.py)
        max_length: if set, will truncate & pad tokens to fit this length
    """

    def __init__(
            self,
            tokenizer_path=None,
            tokenizer=None,
            max_length=256,
            use_fast=True,
    ):
        if not (tokenizer_path or tokenizer):
            raise ValueError("Must pass tokenizer path or tokenizer")
        if tokenizer_path and tokenizer:
            raise ValueError("Cannot pass both tokenizer path and tokenizer")

        if tokenizer_path:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                tokenizer_path, use_fast=use_fast
            )
        else:
            self.tokenizer = tokenizer
        self.max_length = max_length
        self.save_pretrained = self.tokenizer.save_pretrained

    def encode(self, input_text):
        """Encodes ``input_text``.

        ``input_text`` may be a string or a tuple of strings, depending
        if the model takes 1 or multiple inputs. The
        ``transformers.AutoTokenizer`` will automatically handle either
        case.
        """
        if isinstance(input_text, str):
            input_text = (input_text,)
        encoded_text = self.tokenizer.encode_plus(
            *input_text,
            max_length=self.max_length,
            add_special_tokens=True,
            padding=True,
            truncation=True,
        )
        return dict(encoded_text)

    def batch_encode(self, input_text_list):
        """The batch equivalent of ``encode``."""
        if hasattr(self.tokenizer, "batch_encode_plus"):
            if isinstance(input_text_list[0], tuple) and len(input_text_list[0]) == 1:
                # Unroll tuples of length 1.
                input_text_list = [t[0] for t in input_text_list]
            encodings = self.tokenizer.batch_encode_plus(
                input_text_list,
                truncation=True,
                max_length=self.max_length,
                add_special_tokens=True,
                padding="max_length",
            )
            # Encodings is a `transformers.utils.BatchEncode` object, which
            # is basically a big dictionary that contains a key for all input
            # IDs, a key for all attention masks, etc.
            dict_of_lists = {k: list(v) for k, v in encodings.data.items()}
            list_of_dicts = [
                {key: value[index] for key, value in dict_of_lists.items()}
                for index in range(max(map(len, dict_of_lists.values())))
            ]
            # We need to turn this dict of lists into a dict of lists.
            return list_of_dicts
        else:
            return [self.encode(input_text) for input_text in input_text_list]

    def convert_ids_to_tokens(self, ids):
        return self.tokenizer.convert_ids_to_tokens(ids)

    @property
    def pad_token_id(self):
        if hasattr(self.tokenizer, "pad_token_id"):
            return self.tokenizer.pad_token_id
        else:
            raise AttributeError("Tokenizer does not have `pad_token_id` attribute.")

    @property
    def mask_token_id(self):
        if hasattr(self.tokenizer, "mask_token_id"):
            return self.tokenizer.mask_token_id
        else:
            raise AttributeError("Tokenizer does not have `mask_token_id` attribute.")
