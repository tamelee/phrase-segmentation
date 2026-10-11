import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .crf import CRF


class BiRnnCrf(nn.Module):
    """BiRNN-CRF for sequences of notes.

    ADAPTED: each note is described by F categorical features (pitch, interval, duration, ...),
    so the input is [B, L, F] instead of [B, L]. `vocab_sizes` holds one vocabulary size per
    feature; every feature has its own embedding and the F embeddings are concatenated
    before the RNN.
    """

    def __init__(self, vocab_sizes, tagset_size, embedding_dim, hidden_dim, num_rnn_layers=1, rnn="lstm"):
        super(BiRnnCrf, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_sizes = vocab_sizes
        self.tagset_size = tagset_size

        # ADAPTED: one embedding per note feature (was a single nn.Embedding)
        self.embeddings = nn.ModuleList([nn.Embedding(size, embedding_dim) for size in vocab_sizes])
        RNN = nn.LSTM if rnn == "lstm" else nn.GRU
        self.rnn = RNN(embedding_dim * len(vocab_sizes), hidden_dim // 2, num_layers=num_rnn_layers,
                       bidirectional=True, batch_first=True)
        self.crf = CRF(hidden_dim, self.tagset_size)

    def __build_features(self, sentences):
        # ADAPTED: sentences is [B, L, F]. Feature 0 (pitch) is never 0 for a real note, so it
        # defines the padding mask.
        masks = sentences[:, :, 0].gt(0)
        sentences = sentences.long()
        embeds = torch.cat([emb(sentences[:, :, i]) for i, emb in enumerate(self.embeddings)], dim=-1)

        seq_length = masks.sum(1)
        sorted_seq_length, perm_idx = seq_length.sort(descending=True)
        embeds = embeds[perm_idx, :]

        # FIX: recent PyTorch versions require `lengths` on the CPU (fails on GPU otherwise)
        pack_sequence = pack_padded_sequence(embeds, lengths=sorted_seq_length.cpu(), batch_first=True)
        packed_output, _ = self.rnn(pack_sequence)
        lstm_out, _ = pad_packed_sequence(packed_output, batch_first=True)
        _, unperm_idx = perm_idx.sort()
        lstm_out = lstm_out[unperm_idx, :]

        return lstm_out, masks

    def loss(self, xs, tags):
        features, masks = self.__build_features(xs)
        loss = self.crf.loss(features, tags, masks=masks)
        return loss

    def forward(self, xs):
        # Get the emission scores from the BiLSTM
        features, masks = self.__build_features(xs)
        scores, tag_seq = self.crf(features, masks)
        return scores, tag_seq
