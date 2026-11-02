from bi_lstm_crf.app.preprocessing.preprocess import *
from bi_lstm_crf.model.model import BiRnnCrf
from bi_lstm_crf.app.train import *
from bi_lstm_crf.app.predict import *
import torch


def testing_model():
    torch.manual_seed(0)

    B, L, F = 4, 10, 3
    vocab_sizes = [20, 20, 20]      # one per feature
    tagset_size = 5
    embedding_dim = 8
    hidden_dim = 16                 # must be even (split across 2 directions)

    model = BiRnnCrf(vocab_sizes, tagset_size, embedding_dim, hidden_dim)

    # Random notes with values 1..19 (0 is reserved for padding)
    xs = torch.randint(1, 20, (B, L, F))
    # Different lengths per sequence, so sorting and packing are actually exercised
    lengths = [9, 8, 6, 3]
    for b, n in enumerate(lengths):
        xs[b, n:, :] = 0

    tags = torch.randint(0, tagset_size, (B, L))

    # 1. Features
    features, masks = model._BiRnnCrf__build_features(xs)
    print("Input:   ", xs.shape)        # [4, 10, 3]
    print("Features:", features.shape)  # expect [4, 10, 16]
    print("Mask:    ", masks.shape)     # [4, 10]
    assert features.shape == (B, L, hidden_dim)
    assert masks.sum(1).tolist() == lengths

    # 2. Loss + backward
    loss = model.loss(xs, tags)
    loss.backward()
    print("Loss:", loss.item())
    assert torch.isfinite(loss)

    # 3. Decoding
    model.eval()
    with torch.no_grad():
        scores, tag_seq = model(xs)
    print("Scores:", scores)
    print("Tag lengths:", [len(t) for t in tag_seq])


if __name__ == "__main__":
    testing_model()