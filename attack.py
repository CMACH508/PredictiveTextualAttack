import argparse
import copy
import datetime
import json
import os
import random
import time
from itertools import chain

import language_tool_python
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, GPT2LMHeadModel, GPT2TokenizerFast

from attack import CandidateSelectionModule, WordRankingModule
from attack.utils import Vocab, SynonymQuery
from attack.utils import nltk_tokenize, nltk_detokenize, recover_word_case
from misc.utils import preprocess_data, cal_ppl
from victim_model import HuggingFaceModelWrapper, PyTorchModelWrapper, RNNModelForSequenceClassification
from victim_model import TransformerTokenizer, GloveTokenizer


def preprocess_candidate_selection_module_train_data(path, sim_threshold, top_k, ppl_proportion=0.9):
    with open(path, 'r', encoding='utf8') as f:
        data = json.load(f)
    preprocessed_data = []
    for d in data:
        tokens = d['tokens']
        label = d['label']
        if isinstance(tokens[0], str):  # single text input
            tokens = [tokens, []]
        candidates = []
        for can in d['candidates']:
            index = can['index']
            synonyms = can['synonyms']
            sem_sim = can['sem_sim']
            pred_prob = can['pred_prob']
            text_ppl = can['ppl']
            if isinstance(index, int):
                index = [0, index]
            # filter synonyms
            selected_synonyms = [(syn, sim, ap, ppl) for syn, sim, ap, ppl in
                                 zip(synonyms, sem_sim, pred_prob, text_ppl) if sim >= sim_threshold]
            selected_synonyms.sort(key=lambda x: x[3])
            selected_synonyms = selected_synonyms[:int(len(selected_synonyms) * ppl_proportion)]
            selected_synonyms.sort(key=lambda x: x[2])
            selected_synonyms = selected_synonyms[:top_k]
            if selected_synonyms:
                candidates.append({
                    'index': index,
                    'synonyms': selected_synonyms
                })
        if candidates:
            preprocessed_data.append({
                'tokens': tokens,
                'label': label,
                'candidates': candidates
            })
    return preprocessed_data


def preprocess_word_ranking_module_train_data(path, sim_threshold, ppl_proportion=0.9):
    with open(path, 'r', encoding='utf8') as f:
        data = json.load(f)
    preprocessed_data = []
    for d in data:
        tokens = d['tokens']
        prob = d['prob']
        label = d['label']
        if isinstance(tokens[0], str):
            tokens = [tokens, []]
        candidates = []
        for can in d['candidates']:
            index = can['index']
            if isinstance(index, int):
                index = [0, index]
            saliency = can['saliency']
            sem_sim = can['sem_sim']
            pred_prob = can['pred_prob']
            text_ppl = can['ppl']
            # filter synonyms
            selected_synonyms = [(sim, ap, ppl) for sim, ap, ppl in zip(sem_sim, pred_prob, text_ppl)
                                 if sim >= sim_threshold]
            selected_synonyms.sort(key=lambda x: x[2])
            selected_synonyms = selected_synonyms[:int(len(selected_synonyms) * ppl_proportion)]
            attack_effect = max([(prob - ap) * sim for sim, ap, _ in selected_synonyms] + [1e-10])
            candidates.append({
                'index': index,
                'saliency': saliency,
                'attack_effect': attack_effect
            })
        preprocessed_data.append({
            'tokens': tokens,
            'prob': prob,
            'label': label,
            'candidates': candidates
        })
    return preprocessed_data


def train_candidate_selection_module(args):
    print(args)
    train_data = preprocess_candidate_selection_module_train_data(args.train_data_path, args.sim_threshold, args.top_k,
                                                                  args.ppl_proportion)
    if args.num_samples > 0:
        train_data = train_data[:args.num_samples]
    print(f'training data size {len(train_data)}')

    vocab = Vocab.load(args.vocab_path)
    model_config = {
        'encoder_path': args.encoder_path,
        'hidden_dim': args.hidden_dim,
        'num_classes': args.num_classes,
        'act_dim': len(vocab)
    }
    print(model_config)
    device = torch.device(args.device)
    candidate_selection_module = CandidateSelectionModule(model_config).to(device)
    candidate_selection_module.train()

    optimizer = torch.optim.Adam(candidate_selection_module.parameters(), lr=1e-5)
    loss = torch.nn.KLDivLoss(reduction='batchmean')

    batch_size = args.batch_size
    tot_cost, tot_samples = 0, 0

    for epoch in range(args.num_epochs):
        for idx in range(0, len(train_data), batch_size):
            batch_tokens = [d['tokens'] for d in train_data[idx:(idx + batch_size)]]
            batch_indices = [[can['index'] for can in d['candidates']]
                             for d in train_data[idx:(idx + batch_size)]]
            batch_labels = [[d['label']] * len(d['candidates'])
                            for d in train_data[idx:(idx + batch_size)]]
            batch_labels = list(chain(*batch_labels))
            obs = batch_tokens, batch_indices, batch_labels
            pred_prob = candidate_selection_module.forward_for_lm(obs)

            batch_prob = []
            for d in train_data[idx:(idx + batch_size)]:
                for can in d['candidates']:
                    p = torch.zeros(len(vocab))
                    synonyms = can['synonyms']
                    syn_indices = vocab.word2id([syn[0] for syn in synonyms])
                    syn_indices = torch.tensor(syn_indices)
                    p[syn_indices] = 1
                    p /= p.sum()
                    batch_prob.append(p)
            batch_prob = torch.stack(batch_prob).to(device)
            cost = loss(torch.log(pred_prob), batch_prob)
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            tot_cost += cost.item() * pred_prob.shape[0]
            tot_samples += pred_prob.shape[0]
        ctime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"{ctime} epoch {epoch}, cost {tot_cost / tot_samples:.4f}")
        tot_cost, tot_samples = 0, 0

    train_args = {
        'encoder_path': args.encoder_path,
        'sim_threshold': args.sim_threshold,
        'top_k': args.top_k,
        'hidden_dim': args.hidden_dim,
        'num_classes': args.num_classes,
        'num_epochs': args.num_epochs,
        'num_samples': args.num_samples,
        'batch_size': args.batch_size
    }
    if not os.path.exists(args.module_path):
        os.makedirs(args.module_path)
    with open(os.path.join(args.module_path, "train_args.json"), 'w', encoding='utf8') as f:
        json.dump(train_args, f, indent=4)
    torch.save(candidate_selection_module.state_dict(), os.path.join(args.module_path, "pytorch_model.bin"))


def train_word_ranking_module(args):
    print(args)
    train_data = preprocess_word_ranking_module_train_data(args.train_data_path, args.sim_threshold,
                                                           args.ppl_proportion)
    if args.num_samples > 0:
        train_data = train_data[:args.num_samples]
    print(f'training data size {len(train_data)}')

    model_config = {
        'encoder_path': args.encoder_path,
        'hidden_dim': args.hidden_dim,
        'dropout': args.dropout,
        'num_classes': args.num_classes
    }
    print(model_config)
    device = torch.device(args.device)
    word_ranking_module = WordRankingModule(model_config).to(device)
    word_ranking_module.train()

    optimizer = torch.optim.Adam(word_ranking_module.parameters(), lr=1e-5)
    loss = torch.nn.BCELoss()

    batch_size = args.batch_size
    tot_cost, tot_samples = 0, 0

    for epoch in range(args.num_epochs):
        for idx in range(0, len(train_data), batch_size):
            batch_tokens = [d['tokens'] for d in train_data[idx:(idx + batch_size)]]
            batch_indices = [[can['index'] for can in d['candidates']]
                             for d in train_data[idx:(idx + batch_size)]]
            batch_labels = [[d['label']] * len(d['candidates'])
                            for d in train_data[idx:(idx + batch_size)]]
            batch_labels = list(chain(*batch_labels))
            obs = batch_tokens, batch_indices, batch_labels
            pred_prob = word_ranking_module(obs)

            batch_rank_labels = []
            batch_masks = []
            for d in train_data[idx:(idx + batch_size)]:
                rank_label = [0] * len(d['candidates'])
                masks = [0] * len(d['candidates'])
                saliency = torch.tensor([can['saliency'] for can in d['candidates']])
                saliency = torch.softmax(saliency, dim=-1)
                attack_effect = torch.tensor([can['attack_effect'] for can in d['candidates']])
                h = saliency * attack_effect

                n_pos = int(round(args.pos_top_k * len(rank_label), 0))
                n_neg = int(round(args.neg_top_k * len(rank_label), 0))
                n_pos = max(1, n_pos)
                n_neg = min(max(1, n_neg), len(rank_label) - n_pos)
                idx = torch.argsort(h).tolist()
                for p in idx[:n_neg]:
                    masks[p] = 1
                for p in idx[-n_pos:]:
                    rank_label[p] = 1
                    masks[p] = 1
                batch_rank_labels.extend(rank_label)
                batch_masks.extend(masks)
            batch_rank_labels = torch.tensor(batch_rank_labels, dtype=torch.float, device=device)
            batch_masks = torch.tensor(batch_masks, dtype=torch.bool, device=device)
            pred_prob = pred_prob.masked_select(batch_masks)
            batch_rank_labels = batch_rank_labels.masked_select(batch_masks)
            cost = loss(pred_prob, batch_rank_labels)
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            tot_cost += cost.item() * pred_prob.shape[0]
            tot_samples += pred_prob.shape[0]
        ctime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"{ctime} epoch {epoch}, cost {tot_cost / tot_samples:.4f}")
        tot_cost, tot_samples = 0, 0

    train_args = {
        'encoder_path': args.encoder_path,
        'sim_threshold': args.sim_threshold,
        'pos_top_k': args.pos_top_k,
        'neg_top_k': args.neg_top_k,
        'hidden_dim': args.hidden_dim,
        'num_classes': args.num_classes,
        'dropout': args.dropout,
        'num_epochs': args.num_epochs,
        'num_samples': args.num_samples,
        'batch_size': args.batch_size
    }
    if not os.path.exists(args.module_path):
        os.makedirs(args.module_path)
    with open(os.path.join(args.module_path, "train_args.json"), 'w', encoding='utf8') as f:
        json.dump(train_args, f, indent=4)
    torch.save(word_ranking_module.state_dict(), os.path.join(args.module_path, "pytorch_model.bin"))


def word_ranking(text_info, victim_model, candidate_selection_module, synonym_query, vocab, top_k, result):
    text = text_info['text']
    label = text_info['label']
    tokens = text_info['tokens']
    tags = text_info['tags']
    indices = text_info['indices']
    orig_prob = text_info['orig_prob']

    # Calculate saliency
    batch_texts = []
    for i, j in indices:
        tmp = tokens[i][j]
        tokens[i][j] = '[UNK]'
        if isinstance(text, str):
            cur_text = nltk_detokenize(tokens[i])
        else:
            cur_text = copy.copy(text)
            cur_text[i] = nltk_detokenize(tokens[i])
        batch_texts.append(cur_text)
        tokens[i][j] = tmp
    prob = torch.softmax(torch.from_numpy(victim_model(batch_texts)), dim=-1)
    result['num_model_queries'] += len(batch_texts)
    saliency = prob[:, label]
    saliency = torch.softmax(saliency, dim=-1)

    # Calculate attack effect
    attack_effect = []
    obs = [tokens], [indices], [label] * len(indices)
    with torch.no_grad():
        candidate_prob = candidate_selection_module.forward_for_lm(obs)
    candidate_prob = candidate_prob.cpu().numpy()
    for (i, j), can_prob in zip(indices, candidate_prob):
        tmp = tokens[i][j]
        attack_texts = []
        synonyms = synonym_query(tokens[i][j], tags[i][j])
        synonym_index = np.array(vocab.word2id(synonyms))
        can_prob[synonym_index] += 1
        select_num = min(top_k, len(synonyms))
        synonym_index = np.argsort(can_prob)[-1:-(select_num + 1):-1].tolist()
        synonyms = vocab.id2word(synonym_index)
        for syn in synonyms:
            tokens[i][j] = syn
            if isinstance(text, str):
                cur_text = nltk_detokenize(tokens[i])
            else:
                cur_text = copy.copy(text)
                cur_text[i] = nltk_detokenize(tokens[i])
            attack_texts.append(cur_text)
        attack_pred = victim_model(attack_texts)
        result['num_model_queries'] += len(attack_texts)
        attack_prob = torch.softmax(torch.from_numpy(attack_pred), dim=-1)[:, label]
        attack_effect.append(orig_prob - torch.min(attack_prob))
        tokens[i][j] = tmp
    attack_effect = torch.tensor(attack_effect)
    h = saliency * attack_effect
    idx = torch.argsort(h).tolist()[::-1]
    indices = [indices[i] for i in idx]
    return indices


def evaluate(d, candidate_selection_module, word_ranking_module, victim_model, sent_encoder, vocab, synonym_query,
             params):
    text = d['text']
    label = d['label']
    tokens = d['tokens']
    tags = d['tags']
    indices = d['indices']
    text = text if text[1] else text[0]
    indices = [(0, i) for i in indices[0]] + [(1, i) for i in indices[1]]

    start = time.time()

    result = {
        'orig_text': text,
        'atk_text': text,
        'orig_tokens': tokens,
        'atk_tokens': tokens,
        'sem_sim': -1,
        'num_model_queries': 0,
        'consuming_time': 0.0
    }

    if not indices:
        return result

    orig_pred = victim_model([text])
    result['num_model_queries'] += 1
    orig_prob = torch.softmax(torch.from_numpy(orig_pred), dim=-1)
    min_prob = orig_prob[0, label].item()

    orig_embed = sent_encoder.encode([text])
    atk_tokens = copy.deepcopy(tokens)

    if params['attack_method'] == 'tap-cs':
        pass
    elif params['attack_method'] in ['tap-wr', 'tap-full']:
        obs = [tokens], [indices], [label] * len(indices)
        with torch.no_grad():
            importance = word_ranking_module(obs)
        idx = importance.argsort().tolist()[::-1]
        indices = [indices[i] for i in idx]
    else:
        raise ValueError('Invalid attack method.')

    for i, j in indices:
        if params['attack_method'] not in ['tap-cs', 'tap-full']:
            synonyms = synonym_query(atk_tokens[i][j], tags[i][j], word_case_recover=True) + [atk_tokens[i][j]]
        else:
            obs = [atk_tokens], [[i, j]], [label]
            with torch.no_grad():
                prob = candidate_selection_module(obs)
            prob = prob.squeeze(0)
            prob = prob.cpu().numpy()

            synonyms = synonym_query(atk_tokens[i][j], tags[i][j]) + [atk_tokens[i][j]]
            synonym_index = np.array(vocab.word2id(synonyms))
            prob[synonym_index] += 1
            select_num = min(params['top_k'], len(synonyms))
            synonym_index = np.argsort(prob)[-1:-(select_num + 1):-1].tolist()
            synonyms = vocab.id2word(synonym_index)

        batch_texts = []
        for syn in synonyms:
            atk_tokens[i][j] = recover_word_case(syn, tokens[i][j])
            if atk_tokens[1]:
                batch_texts.append([nltk_detokenize(atk_tokens[0]), nltk_detokenize(atk_tokens[1])])
            else:
                batch_texts.append(nltk_detokenize(atk_tokens[0]))
        atk_tokens[i][j] = tokens[i][j]

        # filter out candidate texts with sem_sim below the threshold
        embed = sent_encoder.encode(batch_texts)
        sem_sim = cosine_similarity(orig_embed, embed).reshape(-1)
        if np.max(sem_sim) < params['sim_threshold']:
            continue
        sem_sim = sem_sim.tolist()
        tmp = [(atk_text, syn, sim) for atk_text, syn, sim in zip(batch_texts, synonyms, sem_sim)
               if sim >= params['sim_threshold']]
        batch_texts, synonyms, sem_sim = zip(*tmp)
        batch_texts = list(batch_texts)

        # query the texts' score from the victim model
        pred = victim_model(batch_texts)
        result['num_model_queries'] += len(batch_texts)
        prob = torch.softmax(torch.from_numpy(pred), dim=-1)[:, label]

        syn_idx = torch.argmin(prob).item()
        if prob[syn_idx] >= min_prob:
            continue
        min_prob = prob[syn_idx].item()
        atk_tokens[i][j] = recover_word_case(synonyms[syn_idx], tokens[i][j])

        if np.argmax(pred[syn_idx]) != label:
            result['atk_text'] = batch_texts[syn_idx]
            result['atk_tokens'] = atk_tokens
            result['sem_sim'] = sem_sim[syn_idx]
            break
    result['consuming_time'] = time.time() - start

    return result


def test(args):
    print(args)
    vocab = Vocab.load(args.vocab_path)

    device = torch.device(args.device)
    synonym_query = SynonymQuery.load(args.synonym_query_path)
    sent_encoder = SentenceTransformer(args.sent_encoder_path, device=args.device)

    pretrained_path = args.victim_model_path
    if pretrained_path[-1] == '/':
        pretrained_path = pretrained_path[:-1]
    if os.path.split(pretrained_path)[-1].startswith('bert'):
        bert_tokenizer = TransformerTokenizer(tokenizer_path=pretrained_path)
        victim_model = AutoModelForSequenceClassification.from_pretrained(pretrained_path).to(device)
        victim_model = HuggingFaceModelWrapper(victim_model, bert_tokenizer, batch_size=64)
    elif os.path.split(pretrained_path)[-1].startswith('lstm'):
        glove_tokenizer = GloveTokenizer(tokenizer_path=pretrained_path)
        victim_model = RNNModelForSequenceClassification.from_pretrained(pretrained_path).to(device)
        victim_model = PyTorchModelWrapper(victim_model, glove_tokenizer, batch_size=512)
    else:
        raise ValueError('Invalid victim model path')

    candidate_selection_module_config = {
        'encoder_path': args.encoder_path,
        'hidden_dim': args.hidden_dim,
        'num_classes': args.num_classes,
        'act_dim': len(vocab)
    }
    candidate_selection_module = CandidateSelectionModule(candidate_selection_module_config).to(device)
    candidate_selection_module.load_state_dict(torch.load(args.candidate_selection_module_path, map_location='cpu'))

    word_ranking_module_config = {
        'encoder_path': args.encoder_path,
        'hidden_dim': args.hidden_dim,
        'dropout': args.dropout,
        'num_classes': args.num_classes
    }
    word_ranking_module = WordRankingModule(word_ranking_module_config).to(device)
    word_ranking_module.load_state_dict(torch.load(args.word_ranking_module_path, map_location='cpu'))

    candidate_selection_module.eval()
    word_ranking_module.eval()

    test_data = preprocess_data(args.dataset_path, nltk_tokenize, synonym_query, 'test')
    print(f'testing data size {len(test_data)}')

    gpt_model = GPT2LMHeadModel.from_pretrained(args.gpt_path).to(device)
    gpt_tokenizer = GPT2TokenizerFast.from_pretrained(args.gpt_path)

    tool = language_tool_python.LanguageTool('en-US')

    def cal_error(texts):
        if isinstance(texts, str):
            texts = [texts]
        texts = ' '.join(texts)
        matches = tool.check(texts)
        return len(matches)

    for attack_method in ['tap-full', 'tap-wr', 'tap-cs']:
        params = {
            'sim_threshold': args.sim_threshold,
            'attack_method': attack_method,
            'top_k': args.top_k,
        }
        results = [evaluate(d, candidate_selection_module, word_ranking_module, victim_model, sent_encoder,
                            vocab, synonym_query, params)
                   for d in tqdm(test_data)]
        assert len(results) == len(test_data)
        for r in results:
            r['mod_rate'] = r['ppl'] = r['err_inc'] = 0.0
            if r['sem_sim'] == -1:
                continue
            orig_tokens = r['orig_tokens'][0] + r['orig_tokens'][1]
            atk_tokens = r['atk_tokens'][0] + r['atk_tokens'][1]
            r['mod_rate'] = sum(t1 != t2 for t1, t2 in zip(orig_tokens, atk_tokens)) / len(orig_tokens)
            r['ppl'] = cal_ppl(r['atk_text'], gpt_model, gpt_tokenizer)
            n_orig_err, n_atk_err = cal_error(r['orig_text']), cal_error(r['atk_text'])
            r['err_inc'] = (n_atk_err - n_orig_err) / len(orig_tokens)

        num_success = sum(r['sem_sim'] != -1 for r in results)
        asr = num_success / len(test_data)
        sem_sim = sum(r['sem_sim'] for r in results if r['sem_sim'] != -1) / num_success  # only for successful samples
        num_model_queries = sum(r['num_model_queries'] for r in results) / len(results)
        mod_rate = sum(r['mod_rate'] for r in results if r['sem_sim'] != -1) / num_success
        ppl = sum(r['ppl'] for r in results if r['sem_sim'] != -1) / num_success
        err_inc = sum(r['err_inc'] for r in results if r['sem_sim'] != -1) / num_success
        consuming_time = sum(r['consuming_time'] for r in results) / len(results)

        with open(os.path.join(args.output_path, 'results.txt'), 'a', encoding='utf8') as f:
            print(f"{attack_method:8}, top_k {args.top_k}, sem_thr {args.sim_threshold:.1f}, "
                  f"asr {asr:.3f}, model qry {num_model_queries:.0f}, "
                  f"sem sim {sem_sim:.3f}, mod {mod_rate:.3f}, ppl {ppl:.1f}, err_inc {err_inc:.3f}, "
                  f"consuming_time {consuming_time:.3f} sec", file=f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', type=int, default=0)
    subparsers = parser.add_subparsers()

    parser_train_attack = subparsers.add_parser('train_candidate_selection_module', help='candidate selection module')
    parser_train_attack.add_argument('--device', type=str, default='cuda:0')
    parser_train_attack.add_argument('--encoder_path', type=str, default='resources/encoder_models/bert')
    parser_train_attack.add_argument('--vocab_path', type=str, default='resources/hownet/vocab.pkl')
    parser_train_attack.add_argument('--train_data_path', type=str, default='resources/datasets/bert_train/qqp.json')
    parser_train_attack.add_argument('--module_path', type=str,
                                     default='results/prediction/bert/qqp/bert/candidate_selection_module')
    parser_train_attack.add_argument('--sim_threshold', type=float, default=0.95)
    parser_train_attack.add_argument('--top_k', type=int, default=1)
    parser_train_attack.add_argument('--ppl_proportion', type=float, default=0.9)
    parser_train_attack.add_argument('--hidden_dim', type=int, default=128)
    parser_train_attack.add_argument('--num_classes', type=int, default=2)
    parser_train_attack.add_argument('--num_epochs', type=int, default=5)
    parser_train_attack.add_argument('--num_samples', type=int, default=-1)
    parser_train_attack.add_argument('--batch_size', type=int, default=5)
    parser_train_attack.set_defaults(func=train_candidate_selection_module)

    parser_train_rank = subparsers.add_parser('train_word_ranking_module', help='word ranking module')
    parser_train_rank.add_argument('--device', type=str, default='cuda:0')
    parser_train_rank.add_argument('--encoder_path', type=str, default='resources/encoder_models/bert')
    parser_train_rank.add_argument('--train_data_path', type=str, default='resources/datasets/bert_train/qqp.json')
    parser_train_rank.add_argument('--module_path', type=str,
                                   default='results/prediction/bert/qqp/bert/word_ranking_module')
    parser_train_rank.add_argument('--sim_threshold', type=float, default=0.9)
    parser_train_rank.add_argument('--pos_top_k', type=float, default=0.05)
    parser_train_rank.add_argument('--neg_top_k', type=float, default=0.50)
    parser_train_rank.add_argument('--ppl_proportion', type=float, default=0.9)
    parser_train_rank.add_argument('--hidden_dim', type=int, default=128)
    parser_train_rank.add_argument('--num_classes', type=int, default=2)
    parser_train_rank.add_argument('--dropout', type=float, default=0.5)
    parser_train_rank.add_argument('--num_epochs', type=int, default=5)
    parser_train_rank.add_argument('--num_samples', type=int, default=-1)
    parser_train_rank.add_argument('--batch_size', type=int, default=5)
    parser_train_rank.set_defaults(func=train_word_ranking_module)

    parser_test = subparsers.add_parser('test', help='testing')
    parser_test.add_argument('--device', type=str, default='cuda:0')
    parser_test.add_argument('--encoder_path', type=str, default='resources/encoder_models/bert')
    parser_test.add_argument('--sent_encoder_path', type=str, default='stsb-mpnet-base-v2')
    parser_test.add_argument('--vocab_path', type=str, default='resources/hownet/vocab.pkl')
    parser_test.add_argument('--synonym_query_path', type=str, default='resources/hownet/synonyms.pkl')
    parser_test.add_argument('--dataset_path', type=str, default='resources/datasets/bert_original/qqp.json')
    parser_test.add_argument('--gpt_path', type=str, default='gpt2')
    parser_test.add_argument('--hidden_dim', type=int, default=128)
    parser_test.add_argument('--num_classes', type=int, default=2)
    parser_test.add_argument('--dropout', type=float, default=0.5)
    parser_test.add_argument('--top_k', type=int, default=10)
    parser_test.add_argument('--sim_threshold', type=float, default=0.9)
    parser_test.add_argument('--victim_model_path', type=str, default='resources/victim_models/bert-qqp')
    parser_test.add_argument('--candidate_selection_module_path', type=str,
                             default='results/prediction/bert/qqp/bert/candidate_selection_module')
    parser_test.add_argument('--word_ranking_module_path', type=str,
                             default='results/prediction/bert/qqp/bert/word_ranking_module')
    parser_test.add_argument('--output_path', type=str, default='results/prediction/bert/qqp/bert')
    parser_test.set_defaults(func=test)

    args = parser.parse_args()
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    args.func(args)


if __name__ == '__main__':
    main()
