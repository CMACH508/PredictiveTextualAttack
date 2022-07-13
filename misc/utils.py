import json
import math

import torch


def cal_ppl(text, gpt_model, gpt_tokenizer):
    training = gpt_model.training
    gpt_model.eval()
    if isinstance(text, str):
        text = [text]
    text = ' '.join(text)
    encoding = gpt_tokenizer(text, return_tensors='pt')
    input_ids = encoding.input_ids.to(gpt_model.device)
    target_ids = input_ids.clone()
    with torch.no_grad():
        outputs = gpt_model(input_ids, labels=target_ids)
    neg_log_likelihood = outputs[0].item()
    gpt_model.train(training)
    return math.exp(neg_log_likelihood)


def preprocess_data(path, tokenize, synonym_query, mode='test'):
    with open(path, 'r', encoding='utf8') as f:
        data = json.load(f)
    data = data[mode]

    preprocessed_data = []
    for text, label in data:
        if isinstance(text, str):
            text = [text, '']
        tokens, tags, indices = [], [], []
        for sent in text:
            if sent == '':
                cur_tokens, cur_tags, cur_indices = [], [], []
            else:
                cur_tokens, cur_tags = tokenize(sent)
                cur_indices = []
                for i, (token, tag) in enumerate(zip(cur_tokens, cur_tags)):
                    if synonym_query(token, tag):
                        cur_indices.append(i)
            tokens.append(cur_tokens)
            tags.append(cur_tags)
            indices.append(cur_indices)
        preprocessed_data.append({
            'text': text,
            'label': label,
            'tokens': tokens,
            'tags': tags,
            'indices': indices
        })

    return preprocessed_data
