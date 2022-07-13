import pickle

import torch
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer


def bert_tokenize_batch(batch_words, bert_tokenizer, return_tensors=None, max_length=512):
    """
    Args:
        batch_words (List[[List[str], List[str]]]): a batch of text pairs, represented as tokens
        bert_tokenizer: a transformer `Tokenizer`
        return_tensors: if set to 'pt', return Pytorch tensors
        max_length: max length for truncation
    Returns:
        batch_sub_words (List[List[str]]): the sub words corresponding to `batch_words`,
                                           single text input: [CLS] text [SEP],
                                           text pair input: [CLS] text1 [SEP] text2 [SEP]
        batch_alignment (List[[alignment, alignment_pair]]): alignment between `batch_words` and `batch_sub_words`,
                                                             `alignment` and `alignment_pair` are alignment for
                                                             the two texts
        batch_encoding (Dict): batch encoding which meets the input requirements of transformers model
    """
    bos, eos, pad = [bert_tokenizer.cls_token], [bert_tokenizer.sep_token], [bert_tokenizer.pad_token]
    batch_size = len(batch_words)
    batch_sub_words, batch_sub_words_pair, batch_alignment, batch_alignment_pair = [], [], [], []
    for i in range(batch_size):
        words, words_pair = batch_words[i]
        sep = [bert_tokenizer.sep_token] if words_pair else []
        if sep == ['</s>']:  # RoBERTa model
            sep = ['</s>', '</s>']

        sub_words, alignment, sub_words_pair, alignment_pair = [], [], [], []
        sub_words += bos

        for word in words:
            sub = bert_tokenizer.tokenize(word)
            if len(sub_words) + len(sub) + len(sep) + len(eos) > max_length:
                break
            alignment.append((len(sub_words), len(sub_words) + len(sub)))
            sub_words += sub
        sub_words += sep
        if not words_pair:
            sub_words += eos
        batch_sub_words.append(sub_words)
        batch_alignment.append(alignment)

        for word in words_pair:
            sub = bert_tokenizer.tokenize(word)
            if len(sub_words) + len(sub_words_pair) + len(sub) + len(eos) > max_length:
                break
            alignment_pair.append((len(sub_words) + len(sub_words_pair),
                                   len(sub_words) + len(sub_words_pair) + len(sub)))
            sub_words_pair += sub
        if words_pair:
            sub_words_pair += eos
        batch_sub_words_pair.append(sub_words_pair)
        batch_alignment_pair.append(alignment_pair)

    max_length = max(len(sw) + len(swp) for sw, swp in zip(batch_sub_words, batch_sub_words_pair))
    input_ids, token_type_ids, attention_mask = [], [], []
    for i in range(batch_size):
        n, m = len(batch_sub_words[i]), len(batch_sub_words_pair[i])
        current_input_ids = batch_sub_words[i] + batch_sub_words_pair[i] + pad * (max_length - n - m)
        current_input_ids = bert_tokenizer.convert_tokens_to_ids(current_input_ids)
        current_token_type_ids = [0] * n + [1] * m + [0] * (max_length - n - m)
        current_attention_mask = [1] * (n + m) + [0] * (max_length - n - m)
        input_ids.append(current_input_ids)
        token_type_ids.append(current_token_type_ids)
        attention_mask.append(current_attention_mask)
    if return_tensors == 'pt':
        input_ids = torch.tensor(input_ids, dtype=torch.int64)
        token_type_ids = torch.tensor(token_type_ids, dtype=torch.int64)
        attention_mask = torch.tensor(attention_mask, dtype=torch.int64)
    batch_encoding = {'input_ids': input_ids,
                      'token_type_ids': token_type_ids,
                      'attention_mask': attention_mask}
    batch_encoding = {k: batch_encoding[k] for k in bert_tokenizer.model_input_names}
    batch_sub_words = [sw + swp for sw, swp in zip(batch_sub_words, batch_sub_words_pair)]
    batch_alignment = [[ba, bap] for ba, bap in zip(batch_alignment, batch_alignment_pair)]
    return batch_sub_words, batch_alignment, batch_encoding


def nltk_tokenize(text):
    words = list(filter(lambda x: not x.isspace(), word_tokenize(text)))
    words_with_pos = pos_tag(words, tagset='universal')
    tokens = [item[0] for item in words_with_pos]
    tags = [item[1] for item in words_with_pos]
    return tokens, tags


def nltk_detokenize(tokens):
    return TreebankWordDetokenizer().detokenize(tokens)


def recover_word_case(word: str, reference_word: str):
    """Makes the case of `word` like the case of `reference_word`.
    Supports lowercase, UPPERCASE, and Capitalized.
    """
    if reference_word.islower():
        return word.lower()
    elif reference_word.isupper() and len(reference_word) > 1:
        return word.upper()
    elif reference_word[0].isupper() and reference_word[1:].islower():
        return word.capitalize()
    else:
        return word


class Vocab(object):
    def __init__(self, word2id, id2word, pos_index):
        self.__word2id = word2id
        self.__id2word = id2word
        self.__pos_index = pos_index

    def __len__(self):
        return len(self.__word2id)

    def __contains__(self, word):
        return word in self.__word2id

    def word2id(self, words):
        if isinstance(words, list) or isinstance(words, tuple):
            return [self.word2id(w) for w in words]
        else:
            for w in [words, words.lower(), words.upper(), words.capitalize()]:
                if w in self.__word2id:
                    return self.__word2id[w]
            raise KeyError(f'{words}')

    def id2word(self, idx):
        if isinstance(idx, list) or isinstance(idx, tuple):
            return [self.__id2word[w] for w in idx]
        else:
            return self.__id2word[idx]

    def get_index_by_pos(self, pos):
        return self.__pos_index[pos]

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            vocab_dict = pickle.load(f)
        return Vocab(vocab_dict['word2id'], vocab_dict['id2word'], vocab_dict['pos_index'])

    def save(self, path):
        vocab_dict = {
            'word2id': self.__word2id,
            'id2word': self.__id2word,
            'pos_index': self.__pos_index
        }
        with open(path, 'wb') as f:
            pickle.dump(vocab_dict, f)


class SynonymQuery(object):
    def __init__(self, synonyms):
        self.synonyms = synonyms
        self.part_of_speech = ['ADJ', 'ADV', 'NOUN', 'VERB']

    def __call__(self, word, tag, word_case_recover=False):
        tag = tag.upper()
        if tag not in self.part_of_speech:
            return []
        synonyms = []
        for w in [word, word.lower(), word.upper(), word.capitalize()]:
            if w in self.synonyms:
                synonyms = self.synonyms[w][tag]
                break
        if word_case_recover:
            synonyms = [recover_word_case(syn, word) for syn in synonyms]
        return synonyms

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            synonyms = pickle.load(f)
        return SynonymQuery(synonyms)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.synonyms, f)
