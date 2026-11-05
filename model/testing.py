from bi_lstm_crf.app.preprocessing.preprocess import Preprocessor
from bi_lstm_crf.app.preprocessing.utils import PAD, OOV
from bi_lstm_crf.model.model import BiRnnCrf
from bi_lstm_crf.app.train import *
from bi_lstm_crf.app.predict import *
import torch
import json
import random
import tempfile
from os.path import join
from pprint import pprint


def testing_model():
    ''' Testing file: bi_lstm_crf.model.model.py '''
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


def testing_preprocessor():
    ''' Testing file: bi_lstm_crf.app.preprocessing.preprocess.py '''
    random.seed(0)
    tmp = tempfile.mkdtemp()

    # 1. Example config files 
    # Values are strings: sent_to_vector looks them up with str(v)
    vocab = {
        "pitch":    ["60", "62", "64", "65", "67"],   # feature 0 -> used for the mask
        "interval": ["-2", "0", "2"],
        "duration": ["0.5", "1.0", "2.0"],
    }
    tags = ["O", "E"]

    with open(join(tmp, "vocab.json"), "w") as f:
        json.dump(vocab, f)
    with open(join(tmp, "tags.json"), "w") as f:
        json.dump(tags, f)

    # 2. Example dataset: 10 songs of different lengths 
    # One song contains pitch 72 (not in vocab) to test OOV
    with open(join(tmp, "dataset.txt"), "w") as f:
        for i in range(10):
            n = random.randint(3, 7)
            notes = [[random.choice([60, 62, 64, 65, 67]),
                      random.choice([-2, 0, 2]),
                      random.choice([0.5, 1.0, 2.0])] for _ in range(n)]
            if i == 0:
                notes[0][0] = 72
            song_tags = [random.choice(tags) for _ in range(n)]
            f.write(json.dumps(notes) + "\t" + json.dumps(song_tags) + "\n")

    # 3. Build the preprocessor 
    p = Preprocessor(config_dir=tmp, verbose=True)

    print("\nPAD / OOV symbols:", repr(PAD), repr(OOV))
    print("\n=== vocab (list per feature) ===")
    pprint(p.vocab)
    print("\n=== vocab_dict (value -> index, per feature) ===")
    pprint(p.vocab_dict)
    print("\n=== tags_dict ===")
    pprint(p.tags_dict)
    print("\nPAD_IDX:", p.PAD_IDX)
    print("OOV_IDX:", p.OOV_IDX)

    # validation checks
    for feat in vocab:
        assert p.vocab_dict[feat][PAD] == 0
        assert p.vocab_dict[feat][OOV] == len(p.vocab[feat]) - 1
        # list and dict agree: vocab[f][idx] is the value with that idx
        for value, idx in p.vocab_dict[feat].items():
            assert p.vocab[feat][idx] == value

    # 4. One song -> vectors 
    song = [[72, 0, 1.0], [60, 2, 0.5], [64, -2, 2.0]]
    song_tags = ["O", "E"]
    print("\n=== sent_to_vector (max_seq_len=5) ===")
    print("song:   ", song)
    print("vector: ", p.sent_to_vector(song, max_seq_len=5))   # 72 -> OOV, 2 padding rows
    print("tags:   ", song_tags, "->", p.tags_to_vector(song_tags, max_seq_len=5))

    # 5. Full dataset
    print("\n=== load_dataset ===")
    (x_tr, y_tr), (x_va, y_va), (x_te, y_te) = p.load_dataset(
        tmp, val_split=0.2, test_split=0.2, max_seq_len=0)
    print("first train song:\n", x_tr[0])
    print("its tags:", y_tr[0])
    print("real lengths:", x_tr[:, :, 0].gt(0).sum(1).tolist())

    # 6. Augmentation 
    print("\n=== augment_transpose (max_shift=2) ===")
    x_aug, y_aug = p.augment_transpose(x_tr, y_tr, max_shift=2)
    print("before:", x_tr.shape, " after:", x_aug.shape)   # 1 original + 4 shifts = 5x
    print("pitch col, original :", x_tr[0, :, 0].tolist())
    print("pitch col, shift -2 :", x_aug[len(x_tr), :, 0].tolist())
    # Interval/duration columns and padding must be unchanged
    assert torch.equal(x_aug[len(x_tr), :, 1:], x_tr[0, :, 1:])
    assert torch.equal(x_aug[len(x_tr), :, 0] == 0, x_tr[0, :, 0] == 0)

    print("\nAll checks passed.")


if __name__ == "__main__":
    #testing_model()
    testing_preprocessor()