import argparse
import copy
import json
import os
import random

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, GPT2LMHeadModel, GPT2TokenizerFast

from attack.utils import SynonymQuery
from attack.utils import nltk_tokenize, nltk_detokenize, recover_word_case
from misc.utils import preprocess_data, cal_ppl
from victim_model import HuggingFaceModelWrapper, RNNModelForSequenceClassification, PyTorchModelWrapper
from victim_model import TransformerTokenizer, GloveTokenizer


def pretty_floats(obj):
    if isinstance(obj, float):
        return round(obj, 10)
    elif isinstance(obj, dict):
        return dict((k, pretty_floats(v)) for k, v in obj.items())
    elif isinstance(obj, (list, tuple)):
        return list(map(pretty_floats, obj))
    return obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--synonym_query_path', type=str, default='resources/hownet/synonyms.pkl')
    parser.add_argument('--sent_encoder_path', type=str, default='stsb-mpnet-base-v2')
    parser.add_argument('--dataset_path', type=str, default='resources/datasets/bert_original/mnli-mismatched.json')
    parser.add_argument('--output_path', type=str, default='resources/datasets/bert_train/snli.json')
    parser.add_argument('--model_path', type=str, default='resources/victim_models/bert-snli')
    parser.add_argument('--gpt_path', type=str, default='gpt2')
    parser.add_argument('--random_seed', type=int, default=0)
    parser.add_argument('--train_data_num', type=int, default=30000)
    args = parser.parse_args()
    print(args)

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    device = torch.device(args.device)
    synonym_query = SynonymQuery.load(args.synonym_query_path)
    sent_encoder = SentenceTransformer(args.sent_encoder_path, device=args.device)

    gpt_model = GPT2LMHeadModel.from_pretrained(args.gpt_path).to(device)
    gpt_tokenizer = GPT2TokenizerFast.from_pretrained(args.gpt_path)

    pretrained_path = args.model_path
    if pretrained_path.endswith('/'):
        pretrained_path = pretrained_path[:-1]
    if os.path.split(pretrained_path)[-1].startswith('bert'):
        bert_tokenizer = TransformerTokenizer(tokenizer_path=pretrained_path)
        victim_model = AutoModelForSequenceClassification.from_pretrained(pretrained_path).to(device)
        victim_model = HuggingFaceModelWrapper(victim_model, bert_tokenizer, batch_size=256)
    elif os.path.split(pretrained_path)[-1].startswith('lstm'):
        glove_tokenizer = GloveTokenizer(tokenizer_path=pretrained_path)
        victim_model = RNNModelForSequenceClassification.from_pretrained(pretrained_path).to(device)
        victim_model = PyTorchModelWrapper(victim_model, glove_tokenizer, batch_size=256)
    else:
        raise ValueError('Invalid victim model path')

    data = preprocess_data(args.dataset_path, nltk_tokenize, synonym_query, mode='train')[:args.train_data_num]

    results = []
    for d in tqdm(data):
        text = d['text']
        label = d['label']
        tokens = d['tokens']
        tags = d['tags']
        indices = d['indices']
        text = text if text[1] else text[0]
        indices = [(0, i) for i in indices[0]] + [(1, i) for i in indices[1]]
        if not indices:
            continue

        pred = victim_model([text])
        prob = torch.softmax(torch.from_numpy(pred), dim=-1)
        prob = prob[0, label].item()
        orig_embed = sent_encoder.encode([text])
        orig_ppl = cal_ppl(text, gpt_model, gpt_tokenizer)

        synonyms, texts, nums = [], [], []
        ppl = []
        masked_texts = []
        for i, j in indices:
            cur_synonyms = synonym_query(tokens[i][j], tags[i][j]) + [tokens[i][j]]
            synonyms.extend(cur_synonyms)
            for syn in cur_synonyms:
                tokens[i][j] = recover_word_case(syn, tokens[i][j])
                if isinstance(text, str):
                    cur_text = nltk_detokenize(tokens[i])
                else:
                    cur_text = copy.copy(text)
                    cur_text[i] = nltk_detokenize(tokens[i])
                ppl.append(cal_ppl(cur_text, gpt_model, gpt_tokenizer))
                texts.append(cur_text)
            nums.append(len(cur_synonyms))

            tmp = tokens[i][j]
            tokens[i][j] = '[UNK]'
            if isinstance(text, str):
                cur_text = nltk_detokenize(tokens[i])
            else:
                cur_text = copy.copy(text)
                cur_text[i] = nltk_detokenize(tokens[i])
            masked_texts.append(cur_text)
            tokens[i][j] = tmp
        embed = sent_encoder.encode(texts, batch_size=512)
        sem_sim = cosine_similarity(orig_embed, embed).reshape(-1).tolist()
        atk_pred = victim_model(texts)
        pred_prob = torch.softmax(torch.from_numpy(atk_pred), dim=-1)
        pred_prob = pred_prob[:, label].tolist()

        masked_pred = victim_model(masked_texts)
        masked_prob = torch.softmax(torch.from_numpy(masked_pred), dim=-1)
        word_importance = (prob - masked_prob[:, label]).tolist()

        presum = 0
        candidates = []
        for (i, j), num, wi in zip(indices, nums, word_importance):
            index = j if isinstance(text, str) else (i, j)
            candidates.append({
                'index': index,
                'saliency': wi,
                'synonyms': synonyms[presum:(presum + num)],
                'sem_sim': sem_sim[presum:(presum + num)],
                'pred_prob': pred_prob[presum:(presum + num)],
                'ppl': ppl[presum:(presum + num)]
            })
            presum += num
        assert (presum == len(texts) == len(synonyms) == len(ppl))
        tokens = tokens if tokens[1] else tokens[0]
        tags = tags if tags[1] else tags[0]
        results.append({
            'text': text,
            'tokens': tokens,
            'tags': tags,
            'label': label,
            'prob': prob,
            'ppl': orig_ppl,
            'candidates': candidates
        })
    with open(args.output_path, 'w', encoding='utf8') as f:
        json.dump(pretty_floats(results), f)


if __name__ == '__main__':
    main()
