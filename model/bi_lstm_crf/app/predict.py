import argparse
import numpy as np
from bi_lstm_crf.app.preprocessing import *
from bi_lstm_crf.app.utils import *
# ADDED: MEI input
import csv
import os
import xml.etree.ElementTree as ET
from bi_lstm_crf.app.preprocessing.features import TAG_END, note_features, song_tags
from bi_lstm_crf.app.preprocessing.mei_parser import MEI_NS, find_mei_files, parse_mei


class WordsTagger:
    def __init__(self, model_dir, device=None):
        args_ = load_json_file(arguments_filepath(model_dir))
        args = argparse.Namespace(**args_)
        args.model_dir = model_dir
        self.args = args

        self.preprocessor = Preprocessor(config_dir=model_dir, verbose=False)
        self.model = build_model(self.args, self.preprocessor, load=True, verbose=False)
        self.device = running_device(device)
        self.model.to(self.device)

        self.model.eval()

    def __call__(self, sentences, end_tag=TAG_END):
        """predict songs

        :param sentences: ADAPTED: a list of songs, each a list of notes (note = list of feature values,
            see preprocessing.features.note_features)
        :param end_tag: ADAPTED: the tag marking the last note of a phrase
        :return: (tags per note, phrases as (start, end) note indexes)
        """
        if not isinstance(sentences, (list, tuple)):
            raise ValueError("sentences must be a list of sentence")

        try:
            # FIX: pad to the longest sequence (np.asarray fails for sequences of different lengths)
            max_len = max(len(s) for s in sentences)
            sent_tensor = np.asarray([self.preprocessor.sent_to_vector(s, max_seq_len=max_len) for s in sentences])
            sent_tensor = torch.from_numpy(sent_tensor).to(self.device)
            with torch.no_grad():
                _, tags = self.model(sent_tensor)
            tags = self.preprocessor.decode_tags(tags)
        except RuntimeError as e:
            print("*** runtime error: {}".format(e))
            raise e
        return tags, self.phrases_from_tags(tags, end_tag=end_tag)  # ADAPTED: phrases instead of tokens

    @staticmethod
    def phrases_from_tags(tags_list, end_tag=TAG_END):
        """ADDED: tags -> phrases [(start, end), ...] (note indexes, inclusive).
        A phrase ends at every `end_tag`; notes after the last one form a final phrase."""
        phrases_list = []
        for tags in tags_list:
            phrases, start = [], 0
            for idx, tag in enumerate(tags):
                if tag == end_tag:
                    phrases.append((start, idx))
                    start = idx + 1
            if start < len(tags):
                phrases.append((start, len(tags) - 1))
            phrases_list.append(phrases)
        return phrases_list

    @staticmethod
    def tokens_from_tags(sentences, tags_list, begin_tags):
        """extract entities from tags

        :param sentences: a list of sentence
        :param tags_list: a list of tags
        :param begin_tags:
        :return:
        """
        if not tags_list:
            return []

        def _tokens(sentence, ts):
            # begins: [(idx, label), ...]
            all_begin_tags = begin_tags + "O"
            begins = [(idx, t[2:]) for idx, t in enumerate(ts) if t[0] in all_begin_tags]
            begins = [
                         (idx, label)
                         for idx, label in begins
                         if ts[idx] != "O" or (idx > 0 and ts[idx - 1] != "O")
                     ] + [(len(ts), "")]

            tokens_ = [(sentence[s:e], label) for (s, label), (e, _) in zip(begins[:-1], begins[1:]) if label]
            return [((t, tag) if tag else t) for t, tag in tokens_]

        tokens_list = [_tokens(sentence, ts) for sentence, ts in zip(sentences, tags_list)]
        return tokens_list


def write_mei(src, dst, notes, phrases):
    """ADDED: copy of the MEI file with the predicted phrases in <supplied type="predicted phrases">,
    in the same <phrase startid endid> format as the annotations (to compare them in Verovio)."""
    ET.register_namespace("", MEI_NS)
    tree = ET.parse(src)
    # the <meiHead> also contains a <score> (musical incipit): use the one under <music>
    section = tree.getroot().find("{%s}music//{%s}score/{%s}section" % (MEI_NS, MEI_NS, MEI_NS))
    supplied = ET.Element("{%s}supplied" % MEI_NS, {"type": "predicted phrases"})
    for n, (s, e) in enumerate(phrases, 1):
        ET.SubElement(supplied, "{%s}phrase" % MEI_NS,
                      {"n": str(n), "startid": "#" + notes[s]["id"], "endid": "#" + notes[e]["id"]})
    children = list(section)
    pos = next((i + 1 for i, c in enumerate(children) if c.get("type") == "phrases"), 0)
    section.insert(pos, supplied)
    tree.write(dst, encoding="UTF-8", xml_declaration=True)


def main():
    parser = argparse.ArgumentParser()
    # ADAPTED: MEI files instead of a sentence
    parser.add_argument("mei", type=str, nargs="+", help="the MEI file(s) and/or directories to be predicted")
    parser.add_argument('--model_dir', type=str, required=True, help="the model directory for model files")
    parser.add_argument('--device', type=str, default=None,
                        help='the training device: "cuda:0", "cpu:0". It will be auto-detected by default')
    parser.add_argument('--out', type=str, default=None, help="ADDED: CSV file with one row per note")
    parser.add_argument('--write_mei', type=str, default=None,
                        help="ADDED: directory for MEI copies with the predicted phrases")

    args = parser.parse_args()

    tagger = WordsTagger(args.model_dir, args.device)
    if args.write_mei:
        os.makedirs(args.write_mei, exist_ok=True)
    rows, all_golds, all_preds = [], [], []
    for f in find_mei_files(args.mei):
        song = parse_mei(f, require_phrases=False)
        tags, phrases = tagger([note_features(song.notes)])
        tags, phrases = tags[0], phrases[0]
        gold = song_tags(song) if song.phrases else None

        print("{} ({} notes)".format(song.id, len(song.notes)))
        for n, (s, e) in enumerate(phrases, 1):
            print("\tphrase {}: notes {}-{} (measures {}-{})".format(
                n, s, e, song.notes[s]["measure"], song.notes[e]["measure"]))
        if gold:
            all_golds.append(gold)
            all_preds.append(tags)
            m = boundary_prf([gold], [tags], TAG_END)
            print("\tgold phrases: {} | precision {precision:.2f}, recall {recall:.2f}, f1 {f1:.2f}".format(
                len(song.phrases), **m))
        for i, note in enumerate(song.notes):
            rows.append([song.id, i, note["id"], note["measure"], note["name"], note["dur"], note["offset"],
                         gold[i] if gold else "", tags[i]])
        if args.write_mei:
            write_mei(f, join(args.write_mei, os.path.basename(f)), song.notes, phrases)

    if len(all_golds) > 1:
        print("TOTAL ({} annotated songs) | precision {precision:.4f}, recall {recall:.4f}, f1 {f1:.4f}".format(
            len(all_golds), **boundary_prf(all_golds, all_preds, TAG_END)))
    if args.out:
        with open(args.out, "w", newline="", encoding="utf8") as fp:
            writer = csv.writer(fp)
            writer.writerow(["song_id", "idx", "note_id", "measure", "pitch", "dur", "offset", "gold", "pred"])
            writer.writerows(rows)
        print("predictions => {}".format(args.out))


if __name__ == "__main__":
    main()
