import argparse
import copy
import os
import time

import language_tool_python
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, GPT2LMHeadModel, GPT2TokenizerFast

from attack.utils import SynonymQuery
from attack.utils import nltk_tokenize, nltk_detokenize
from misc.utils import preprocess_data, cal_ppl
from victim_model import HuggingFaceModelWrapper, PyTorchModelWrapper, RNNModelForSequenceClassification
from victim_model import TransformerTokenizer, GloveTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--synonym_query_path', type=str, default='resources/hownet/synonyms.pkl')
    parser.add_argument('--sent_encoder_path', type=str, default='stsb-mpnet-base-v2')
    parser.add_argument('--dataset_path', type=str, default='resources/datasets/bert_original/qqp.json')
    parser.add_argument('--model_path', type=str, default='resources/victim_models/bert-qqp')
    parser.add_argument('--output_path', type=str, default='results/greedy/qqp/bert/seq-thr-0.9.txt')
    parser.add_argument('--gpt_path', type=str, default='gpt2')
    parser.add_argument('--word_order', type=str, default='seq')
    parser.add_argument('--sim_threshold', type=float, default=0.9)
    args = parser.parse_args()
    print(args)

    device = torch.device(args.device)
    synonym_query = SynonymQuery.load(args.synonym_query_path)
    sent_encoder = SentenceTransformer(args.sent_encoder_path, device=args.device)

    pretrained_path = args.model_path
    if pretrained_path[-1] == '/':
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

    data = preprocess_data(args.dataset_path, nltk_tokenize, synonym_query, mode='test')
    print(f'dataset size: {len(data)}')

    gpt_model = GPT2LMHeadModel.from_pretrained(args.gpt_path).to(device)
    gpt_tokenizer = GPT2TokenizerFast.from_pretrained(args.gpt_path)

    tool = language_tool_python.LanguageTool('en-US')

    def cal_error(texts):
        if isinstance(texts, str):
            texts = [texts]
        texts = ' '.join(texts)
        matches = tool.check(texts)
        return len(matches)

    results = []  # record the results.
    for d in tqdm(data):
        text = d['text']
        label = d['label']
        tokens = d['tokens']
        tags = d['tags']
        indices = d['indices']
        text = text if text[1] else text[0]
        indices = [(0, i) for i in indices[0]] + [(1, i) for i in indices[1]]

        start = time.time()
        num_model_queries = 0

        results.append({
            'orig_text': text,
            'atk_text': text,
            'orig_tokens': tokens,
            'atk_tokens': tokens,
            'sem_sim': -1,
            'num_model_queries': 0,
            'consuming_time': 0.0,
            'mod_rate': 0.0,
            'ppl': 0.0,
            'err_inc': 0.0
        })

        if not indices:
            continue

        orig_pred = victim_model([text])
        num_model_queries += 1
        orig_prob = torch.softmax(torch.from_numpy(orig_pred), dim=-1)[0]
        min_prob = orig_prob[label].item()
        assert label == torch.argmax(orig_prob)

        orig_embed = sent_encoder.encode([text])

        if args.word_order == 'seq':
            pass
        elif args.word_order == 'text-fooler':
            batch_texts = []
            for i, j in indices:
                if isinstance(text, str):
                    cur_text = nltk_detokenize(tokens[i][:j] + tokens[i][j + 1:])
                else:
                    cur_text = copy.copy(text)
                    cur_text[i] = nltk_detokenize(tokens[i][:j] + tokens[i][j + 1:])
                batch_texts.append(cur_text)
            prob = torch.softmax(torch.from_numpy(victim_model(batch_texts)), dim=-1)
            num_model_queries += len(batch_texts)
            score = orig_prob[label] - prob[:, label]
            pred = torch.argmax(prob, dim=-1)
            score += (prob[torch.arange(len(batch_texts)), pred] - orig_prob[pred]) * torch.not_equal(pred, label)
            idx = torch.argsort(score, descending=True).tolist()
            indices = [indices[i] for i in idx]
        else:
            raise ValueError('Invalid word_order value.')

        atk_tokens = copy.deepcopy(tokens)
        for i, j in indices:
            synonyms = synonym_query(atk_tokens[i][j], tags[i][j], word_case_recover=True) + [atk_tokens[i][j]]

            batch_texts = []
            for syn in synonyms:
                atk_tokens[i][j] = syn
                cur_text = [nltk_detokenize(atk_tokens[0]), nltk_detokenize(atk_tokens[1])] if atk_tokens[1] \
                    else nltk_detokenize(atk_tokens[0])
                batch_texts.append(cur_text)

            # filter out candidate texts with sem_sim below the threshold
            embed = sent_encoder.encode(batch_texts)
            sem_sim = cosine_similarity(orig_embed, embed).reshape(-1)
            if np.max(sem_sim) < args.sim_threshold:
                atk_tokens[i] = tokens[i]
                continue
            sem_sim = sem_sim.tolist()
            tmp = [(atk_text, syn, sim) for atk_text, syn, sim in zip(batch_texts, synonyms, sem_sim)
                   if sim >= args.sim_threshold]
            batch_texts, synonyms, sem_sim = zip(*tmp)
            batch_texts = list(batch_texts)

            # query the texts' score from the victim model
            pred = victim_model(batch_texts)
            num_model_queries += len(batch_texts)
            prob = torch.softmax(torch.from_numpy(pred), dim=-1)[:, label]
            syn_idx = torch.argmin(prob).item()
            if prob[syn_idx] >= min_prob:
                atk_tokens[i][j] = tokens[i][j]
                continue
            min_prob = prob[syn_idx].item()
            atk_tokens[i][j] = synonyms[syn_idx]

            if np.argmax(pred[syn_idx]) != label:
                results[-1]['atk_text'] = batch_texts[syn_idx]
                results[-1]['atk_tokens'] = atk_tokens
                results[-1]['sem_sim'] = sem_sim[syn_idx]
                break
        results[-1]['num_model_queries'] = num_model_queries
        results[-1]['consuming_time'] = time.time() - start
        orig_tokens = tokens[0] + tokens[1]
        atk_tokens = atk_tokens[0] + atk_tokens[1]
        results[-1]['mod_rate'] = sum(t1 != t2 for t1, t2 in zip(orig_tokens, atk_tokens)) / len(orig_tokens)
        results[-1]['ppl'] = cal_ppl(results[-1]['atk_text'], gpt_model, gpt_tokenizer)
        n_orig_err, n_atk_err = cal_error(results[-1]['orig_text']), cal_error(results[-1]['atk_text'])
        results[-1]['err_inc'] = (n_atk_err - n_orig_err) / len(orig_tokens)

    assert (len(results) == len(data))
    num_success = sum(r['sem_sim'] != -1 for r in results)
    asr = num_success / len(data)
    sem_sim = sum(r['sem_sim'] for r in results if r['sem_sim'] != -1) / num_success  # only for successful samples
    num_model_queries = sum(r['num_model_queries'] for r in results) / len(results)
    mod_rate = sum(r['mod_rate'] for r in results if r['sem_sim'] != -1) / num_success
    ppl = sum(r['ppl'] for r in results if r['sem_sim'] != -1) / num_success
    err_inc = sum(r['err_inc'] for r in results if r['sem_sim'] != -1) / num_success
    consuming_time = sum(r['consuming_time'] for r in results) / len(results)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    with open(os.path.join(args.output_path, f'{args.word_order}.txt'), 'w', encoding='utf8') as f:
        print(f'asr {asr:.3f}, model qry {num_model_queries:.0f}, '
              f'sem sim {sem_sim:.3f}, mod {mod_rate:.3f}, ppl {ppl:.1f}, err_inc {err_inc:.3f}, '
              f'consuming_time {consuming_time:.3f} sec', file=f)
        for r in results:
            if r['sem_sim'] == -1:
                continue
            orig_text, atk_text, sem_sim = r['orig_text'], r['atk_text'], r['sem_sim']
            if isinstance(orig_text, list):
                orig_text = ' '.join(orig_text)
            if isinstance(atk_text, list):
                atk_text = ' '.join(atk_text)
            print(f'semantic similarity {sem_sim:.3f}', file=f)
            print(f'{orig_text}\n{atk_text}\n', file=f)


if __name__ == '__main__':
    main()
