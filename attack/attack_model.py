import torch
from transformers import AutoTokenizer, AutoModel

from .utils import bert_tokenize_batch


class CandidateSelectionModule(torch.nn.Module):
    def __init__(self, config):
        super(CandidateSelectionModule, self).__init__()
        self.config = config
        self.bert_model = AutoModel.from_pretrained(config['encoder_path'])
        self.bert_tokenizer = AutoTokenizer.from_pretrained(config['encoder_path'])
        self.transform = torch.nn.Linear(in_features=self.bert_model.config.hidden_size,
                                         out_features=config['num_classes'] * config['hidden_dim'],
                                         bias=False)
        self.out = torch.nn.Linear(in_features=config['hidden_dim'],
                                   out_features=config['act_dim'])

    def forward(self, obs):
        """
        Args:
            obs: (tokens, indices)
                tokens (List[[List[str], List[str]]]): a batch of text pairs, represented as tokens
                indices (List[[int, int]]): the substitute positions of each text pair
                                            each text has one position for substitution
                                            position format: text_id (0 or 1), position
                labels (List[int]): ground truth label
        Returns:
            Tensor of shape (batch_size, act_dim)
        """
        tokens, indices, labels = obs
        batch_size = len(tokens)

        device = next(self.bert_model.parameters()).device

        _, batch_alignment, batch_encoding = bert_tokenize_batch(
            batch_words=tokens,
            bert_tokenizer=self.bert_tokenizer,
            return_tensors='pt'
        )
        batch_encoding = {k: v.to(device) for k, v in batch_encoding.items()}
        bert_output = self.bert_model(**batch_encoding)
        bert_last_hidden_state = bert_output['last_hidden_state']  # (batch_size, seq_len, 768)

        # # mean
        # indices = [align[text_id][i] for align, (text_id, i) in zip(batch_alignment, indices)]
        # word_encoding = [torch.mean(bert_last_hidden_state[i, st:ed], dim=0) for i, (st, ed) in enumerate(indices)]
        # word_encoding = torch.stack(word_encoding, dim=0) # (batch_size, 768)
        # first
        indices = torch.tensor([align[text_id][i][0] for align, (text_id, i) in zip(batch_alignment, indices)],
                               device=device)
        word_encoding = bert_last_hidden_state[torch.arange(batch_size, device=device), indices]  # (batch_size, 768)
        word_encoding = self.transform(word_encoding)  # (batch_size, num_classes * hidden_dim)
        word_encoding = word_encoding.view(-1, self.config['num_classes'], self.config['hidden_dim'])
        word_encoding = word_encoding[torch.arange(batch_size), labels]  # (batch_size, hidden_dim)

        output = self.out(word_encoding)  # (batch_size, act_dim)
        output = torch.softmax(output, dim=-1)  # (batch_size, act_dim)

        return output

    def forward_for_lm(self, obs):
        """
        Args:
            obs: (tokens, indices, labels)
                tokens (List[[List[str], List[str]]]): a batch of text pairs, represented as tokens
                indices (List[List[[int, int]]]): the substitute positions of each text pair
                                                  each text has multiple positions for substitution
                                                  position format: text_id (0 or 1), position
                labels (List[int]): ground truth label
        Returns:
            Tensor of shape (n_1 + n_2 + ... + n_{batch_size}, act_dim)
        """
        tokens, indices, labels = obs
        batch_size = len(tokens)

        device = next(self.bert_model.parameters()).device
        word_encoding = []
        _, batch_alignment, batch_encoding = bert_tokenize_batch(
            batch_words=tokens,
            bert_tokenizer=self.bert_tokenizer,
            return_tensors='pt'
        )
        batch_encoding = {k: v.to(device) for k, v in batch_encoding.items()}
        bert_output = self.bert_model(**batch_encoding)
        bert_last_hidden_state = bert_output['last_hidden_state']  # (batch_size, seq_len, 768)
        for i in range(batch_size):
            align = batch_alignment[i]
            # # mean
            # index_i = [align[text_id][k] for text_id, k in indices[i]]
            # word_encoding.append(
            #     torch.stack([torch.mean(bert_last_hidden_state[i, st:ed], dim=0) for st, ed in index_i], dim=0)
            # )
            # first
            index_i = torch.tensor([align[text_id][k][0] for text_id, k in indices[i]], device=device)
            word_encoding.append(bert_last_hidden_state[i, index_i])  # (len(index_i), 768)

        word_encoding = torch.cat(word_encoding, dim=0)  # (sum(index_i), 768)
        word_encoding = self.transform(word_encoding)  # (sum(index_i), num_classes * hidden_dim)
        word_encoding = word_encoding.view(-1, self.config['num_classes'], self.config['hidden_dim'])
        word_encoding = word_encoding[torch.arange(word_encoding.shape[0]), labels]  # (sum(index_i), hidden_dim)

        output = self.out(word_encoding)  # (sum(index_i), act_dim)
        output = torch.softmax(output, dim=-1)

        return output


class WordRankingModule(torch.nn.Module):
    def __init__(self, config):
        super(WordRankingModule, self).__init__()
        self.config = config
        self.bert_model = AutoModel.from_pretrained(config['encoder_path'])
        self.bert_tokenizer = AutoTokenizer.from_pretrained(config['encoder_path'])
        self.transform = torch.nn.Linear(in_features=self.bert_model.config.hidden_size,
                                         out_features=config['num_classes'] * config['hidden_dim'],
                                         bias=False)
        self.out = torch.nn.Sequential(
            torch.nn.Dropout(config['dropout']),
            torch.nn.Linear(in_features=config['hidden_dim'], out_features=config['hidden_dim']),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=config['hidden_dim'], out_features=1)
        )

    def forward(self, obs):
        """
        Args:
            obs: (tokens, indices, labels)
                tokens (List[[List[str], List[str]]]): a batch of text pairs, represented as tokens
                indices (List[List[[int, int]]]): the substitute positions of each text pair
                                                  each text has multiple positions for substitution
                                                  position format: text_id (0 or 1), position
                labels (List[int]): ground truth label
        Returns:
            Tensor of shape (n_1 + n_2 + ... + n_{batch_size}, act_dim)
        """
        tokens, indices, labels = obs
        batch_size = len(tokens)

        device = next(self.bert_model.parameters()).device
        word_encoding = []
        _, batch_alignment, batch_encoding = bert_tokenize_batch(
            batch_words=tokens,
            bert_tokenizer=self.bert_tokenizer,
            return_tensors='pt'
        )
        batch_encoding = {k: v.to(device) for k, v in batch_encoding.items()}
        bert_output = self.bert_model(**batch_encoding)
        bert_last_hidden_state = bert_output['last_hidden_state']  # (batch_size, seq_len, 768)
        for i in range(batch_size):
            align = batch_alignment[i]
            # # mean
            # index_i = [align[text_id][k] for text_id, k in indices[i]]
            # word_encoding.append(
            #     torch.stack([torch.mean(bert_last_hidden_state[i, st:ed], dim=0) for st, ed in index_i], dim=0)
            # )
            # first
            index_i = torch.tensor([align[text_id][k][0] for text_id, k in indices[i]], device=device)
            word_encoding.append(bert_last_hidden_state[i, index_i])  # (len(index_i), 768)

        word_encoding = torch.cat(word_encoding, dim=0)  # (sum(index_i), 768)
        word_encoding = self.transform(word_encoding)  # (sum(index_i), num_classes * hidden_dim)
        word_encoding = word_encoding.view(-1, self.config['num_classes'], self.config['hidden_dim'])
        word_encoding = word_encoding[torch.arange(word_encoding.shape[0]), labels]  # (sum(index_i), hidden_dim)
        output = self.out(word_encoding).view(-1)  # (sum(index_i),)
        output = torch.sigmoid(output)  # (sum(index_i),)

        return output
