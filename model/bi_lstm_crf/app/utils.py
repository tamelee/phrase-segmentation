from os.path import exists, join
import torch
from bi_lstm_crf.model import BiRnnCrf

FILE_ARGUMENTS = "arguments.json"
FILE_MODEL = "model.pth"


def arguments_filepath(model_dir):
    return join(model_dir, FILE_ARGUMENTS)


def model_filepath(model_dir):
    return join(model_dir, FILE_MODEL)


def build_model(args, processor, load=True, verbose=False):
    # ADAPTED: one vocabulary size per note feature
    model = BiRnnCrf([len(v) for v in processor.vocab.values()], len(processor.tags),
                     embedding_dim=args.embedding_dim, hidden_dim=args.hidden_dim, num_rnn_layers=args.num_rnn_layers)

    # weights
    model_path = model_filepath(args.model_dir)
    if exists(model_path) and load:
        # FIX: map_location allows loading a model trained on GPU (e.g. Colab) on a CPU-only machine
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        if verbose:
            print("load model weights from {}".format(model_path))
    return model


def running_device(device):
    return device if device else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def boundary_prf(golds, preds, positive_tag):
    """ADDED: precision / recall / F1 of the phrase-end class, micro-averaged over all notes
    (TP/FP/FN counted per note, as in the TransPhrase paper).

    :param golds: list of gold tag sequences (tag indexes or tag strings)
    :param preds: list of predicted tag sequences, same lengths as golds
    :param positive_tag: the tag that marks the last note of a phrase
    """
    tp = fp = fn = correct = total = 0
    for gold, pred in zip(golds, preds):
        for g, p in zip(gold, pred):
            tp += g == positive_tag and p == positive_tag
            fp += g != positive_tag and p == positive_tag
            fn += g == positive_tag and p != positive_tag
            correct += g == p
            total += 1
    precision = tp / (tp + fp) if tp + fp else 0.
    recall = tp / (tp + fn) if tp + fn else 0.
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.
    return dict(precision=precision, recall=recall, f1=f1, accuracy=correct / max(total, 1),
                tp=int(tp), fp=int(fp), fn=int(fn))
