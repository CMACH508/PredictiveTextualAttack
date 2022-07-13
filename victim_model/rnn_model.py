import json
import os

import torch


class RNNModelForSequenceClassification(torch.nn.Module):
    def __init__(self, config):
        super(RNNModelForSequenceClassification, self).__init__()
        self.config = config
        self.embed = torch.nn.Embedding(config['vocab_size'], config['embed_dim'])
        self.encoder = torch.nn.LSTM(input_size=config['embed_dim'],
                                     hidden_size=config['hidden_dim'],
                                     num_layers=config['num_layers'],
                                     bidirectional=config['bidirectional'])
        self.dropout = torch.nn.Dropout(p=config['dropout'])
        self.out = torch.nn.Linear(in_features=config['hidden_dim'] * (1 + config['bidirectional']),
                                   out_features=config['num_classes'])

    def forward(self, texts):
        """
        Args:
            texts: a batch of texts represented as token indices, 2D list of shape (batch, seq_len)
        Returns:
            out: score for each class, tensor of shape (batch, num_classes)
        """
        texts = torch.tensor(texts).to(self.device)
        batch_size = texts.shape[0]
        texts = texts.transpose(0, 1)  # (seq_len, batch)
        lengths = torch.sum(torch.ne(texts, self.config['pad_token_id']), dim=0).tolist()  # (batch,)
        texts = texts[:max(lengths), :]
        embed = self.embed(texts)  # (seq_len, batch, embed_dim)
        padded_sequences = torch.nn.utils.rnn.pack_padded_sequence(embed, lengths, enforce_sorted=False)
        _, (h, _) = self.encoder(padded_sequences)  # (num_layers * num_directions, batch, hidden_size)
        h = h.view(self.encoder.num_layers, -1, batch_size, self.encoder.hidden_size)
        h = h[-1]  # (num_directions, batch, hidden_size)
        h = h.transpose(0, 1).reshape(batch_size, -1)  # (batch, num_directions * hidden_size)
        h = self.dropout(h)

        out = self.out(h)  # (batch, num_classes)
        return out

    @staticmethod
    def from_pretrained(path):
        with open(os.path.join(path, 'config.json'), 'r', encoding='utf8') as f:
            config = json.load(f)
        model = RNNModelForSequenceClassification(config)
        state_dict = torch.load(os.path.join(path, 'pytorch_model.bin'), map_location='cpu')
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        with open(os.path.join(path, 'config.json'), 'w', encoding='utf8') as f:
            json.dump(self.config, f)
        torch.save(self.state_dict(), os.path.join(path, 'pytorch_model.bin'))

    @property
    def device(self):
        return next(self.encoder.parameters()).device


class RNNModelForNLI(torch.nn.Module):
    def __init__(self, config):
        super(RNNModelForNLI, self).__init__()
        self.config = config
        self.embed = torch.nn.Embedding(num_embeddings=config['vocab_size'],
                                        embedding_dim=config['embed_dim'])
        self.encoder = torch.nn.LSTM(input_size=config['embed_dim'],
                                     hidden_size=config['hidden_dim'],
                                     num_layers=1,
                                     bidirectional=True)
        self.composition = torch.nn.LSTM(input_size=config['hidden_dim'],
                                         hidden_size=config['hidden_dim'],
                                         num_layers=1,
                                         bidirectional=True)
        self.transform = torch.nn.Sequential(
            torch.nn.Linear(4 * 2 * config['hidden_dim'], config['hidden_dim']),
            torch.nn.ReLU()
        )
        self.output = torch.nn.Sequential(
            torch.nn.Dropout(p=config['dropout']),
            torch.nn.Linear(2 * 4 * config['hidden_dim'], config['hidden_dim']),
            torch.nn.Tanh(),
            torch.nn.Dropout(p=config['dropout']),
            torch.nn.Linear(config['hidden_dim'], config['num_classes'])
        )

    @staticmethod
    def sequence_encoding(encoder, sequences, lengths):
        """
        Encode `sequences` with `encoder`
        Args:
            encoder: an rnn based encoder which takes a sequence of embed-dim vectors as input,
                     and outputs hidden-dim vectors of the same of length.
            sequences: a batch of sequences, tensor of shape (batch, seq_len, embed)
            lengths: lengths of `sequences`, of shape (batch,)
        Returns:
            encoding: the output of encoding `sequences` with `encoder`, tensor of shape (batch, seq_len, hidden)
        """
        padded_sequences = torch.nn.utils.rnn.pack_padded_sequence(sequences,
                                                                   lengths.cpu(),
                                                                   batch_first=True,
                                                                   enforce_sorted=False)
        h, _ = encoder(padded_sequences)
        h, _ = torch.nn.utils.rnn.pad_packed_sequence(h, batch_first=True)  # (batch, seq_len, hidden)
        return h

    def forward(self, texts):
        """
        Args:
            texts: a batch of text pairs
                premises: a batch of texts represented as token indices, 2D list of shape (batch, premise_len)
                hypotheses: a batch of texts represented as token indices, 2D list of shape (batch, hypothesis_len)
        Returns:
            out: score for each class, tensor of shape (batch, num_classes)
        """
        premises, hypotheses = [t[0] for t in texts], [t[1] for t in texts]
        premises = torch.tensor(premises).to(self.device)  # (batch, p_len)
        hypotheses = torch.tensor(hypotheses).to(self.device)  # (batch, h_len)
        p_mask = torch.eq(premises, self.config['pad_token_id']).to(self.device)  # (batch, p_len)
        h_mask = torch.eq(hypotheses, self.config['pad_token_id']).to(self.device)  # (batch, h_len)
        p_lengths = torch.sum(torch.logical_not(p_mask), dim=1)  # (batch,)
        h_lengths = torch.sum(torch.logical_not(h_mask), dim=1)  # (batch,)

        p_max_length, h_max_length = torch.max(p_lengths).item(), torch.max(h_lengths).item()
        premises, hypotheses = premises[:, :p_max_length], hypotheses[:, :h_max_length]  # (batch, p_len/h_len)
        p_mask, h_mask = p_mask[:, :p_max_length], h_mask[:, :h_max_length]  # (batch, p_len/h_len)

        p_embed = self.embed(premises)  # (batch, p_len, embed_dim)
        h_embed = self.embed(hypotheses)  # (batch, h_len, embed_dim)

        p_encoding = self.sequence_encoding(self.encoder, p_embed, p_lengths)  # (batch, p_len, 2 * hidden_dim)
        h_encoding = self.sequence_encoding(self.encoder, h_embed, h_lengths)  # (batch, h_len, 2 * hidden_dim)

        p_attention = torch.matmul(p_encoding, h_encoding.transpose(1, 2))  # (batch, p_len, h_len)
        h_attention = p_attention.transpose(1, 2).contiguous()  # (batch, h_len, p_len)
        p_attention.masked_fill_(h_mask.unsqueeze(1), -1e6)  # (batch, p_len, h_len)
        h_attention.masked_fill_(p_mask.unsqueeze(1), -1e6)  # (batch, h_len, p_len)
        p_attention = torch.softmax(p_attention, dim=-1)  # (batch, p_len, h_len)
        h_attention = torch.softmax(h_attention, dim=-1)  # (batch, h_len, p_len)
        p_encoding_inference = torch.matmul(p_attention, h_encoding)  # (batch, p_len, 2 * hidden_dim)
        h_encoding_inference = torch.matmul(h_attention, p_encoding)  # (batch, h_len, 2 * hidden_dim)

        p_encoding_enhanced = torch.cat([p_encoding,
                                         p_encoding_inference,
                                         p_encoding - p_encoding_inference,
                                         p_encoding * p_encoding_inference],
                                        dim=-1)  # (batch, p_len, 4 * 2 * hidden_dim)
        h_encoding_enhanced = torch.cat([h_encoding,
                                         h_encoding_inference,
                                         h_encoding - h_encoding_inference,
                                         h_encoding * h_encoding_inference],
                                        dim=-1)  # (batch, h_len, 4 * 2 * hidden_dim)

        p_encoding = self.transform(p_encoding_enhanced)  # (batch, p_len, hidden_dim)
        h_encoding = self.transform(h_encoding_enhanced)  # (batch, h_len, hidden_dim)

        vp = self.sequence_encoding(self.composition, p_encoding, p_lengths)  # (batch, p_len, 2 * hidden_dim)
        vh = self.sequence_encoding(self.composition, h_encoding, h_lengths)  # (batch, h_len, 2 * hidden_dim)

        vp_avg = torch.sum(vp * p_mask.unsqueeze(2), dim=1) / p_lengths.unsqueeze(1)  # (batch, 2 * hidden_dim)
        vh_avg = torch.sum(vh * h_mask.unsqueeze(2), dim=1) / h_lengths.unsqueeze(1)  # (batch, 2 * hidden_dim)
        vp_max, _ = torch.max(vp.masked_fill(p_mask.unsqueeze(2), -1e6), dim=1)  # (batch, 2 * hidden_dim)
        vh_max, _ = torch.max(vh.masked_fill(h_mask.unsqueeze(2), -1e6), dim=1)  # (batch, 2 * hidden_dim)

        v = torch.cat([vp_avg, vp_max, vh_avg, vh_max], dim=1)  # (batch, 4 * 2 * hidden_dim)
        out = self.output(v)  # (batch, num_classes)
        return out

    @staticmethod
    def from_pretrained(path):
        with open(os.path.join(path, 'config.json'), 'r', encoding='utf8') as f:
            config = json.load(f)
        model = RNNModelForNLI(config)
        state_dict = torch.load(os.path.join(path, 'pytorch_model.bin'), map_location='cpu')
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        with open(os.path.join(path, 'config.json'), 'w', encoding='utf8') as f:
            json.dump(self.config, f)
        torch.save(self.state_dict(), os.path.join(path, 'pytorch_model.bin'))

    @property
    def device(self):
        return next(self.encoder.parameters()).device
