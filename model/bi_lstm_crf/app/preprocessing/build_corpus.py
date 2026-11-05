"""
ADDED: build a corpus directory from the compiled EA-Digifolk dataset (EADIGIFOLK.gzip), in the
same structure the original code expects
(https://github.com/jidasheng/bi-lstm-crf/wiki/corpus-structure-and-format):

    corpus_dir/
        vocab.json     {feature: [values]}   (one vocabulary per note feature)
        tags.json      ["O", "E"]
        dataset.txt    one song per line: <json notes>\\t<json tags>
        songs.json     song ids, in the same order as dataset.txt (for tracing results)
        report.csv     status of every song in the dataset (ok / warning / error)

Usage:
    python -m bi_lstm_crf.app.preprocessing.build_corpus EADIGIFOLK.gzip corpus_dir [--countries PT ES MX]

The songs (notes, sung order and phrase labels) come from the EA-Digifolk parser itself, so the
song count matches the one used by the dataset owners. The old MEI-based parser (mei_parser.py)
is no longer used here.

Songs are shuffled (with --seed) so that the contiguous train/val/test split done by the
original Preprocessor.load_dataset is not biased by the order of the dataset.
"""
import argparse
import collections
import csv
import glob
import os
import random
from os.path import join

from bi_lstm_crf.app.preprocessing.eadigifolk import load_songs
from bi_lstm_crf.app.preprocessing.features import TAGS, build_vocab, note_features, song_tags
from bi_lstm_crf.app.preprocessing.utils import json, save_json_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, help="compiled EA-Digifolk dataset (EADIGIFOLK.gzip)")
    parser.add_argument("corpus_dir", type=str, help="output corpus directory")
    parser.add_argument("--countries", nargs="+", default=None,
                        help="keep only these countries (e.g. PT ES MX); default: all")
    parser.add_argument("--seed", type=int, default=42, help="seed used to shuffle the songs")
    args = parser.parse_args()

    os.makedirs(args.corpus_dir, exist_ok=True)

    samples, n_rows, n_errors = [], 0, 0
    per_country = collections.Counter()
    with open(join(args.corpus_dir, "report.csv"), "w", newline="", encoding="utf8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["song", "country", "title", "status", "notes", "phrases", "message"])
        for song_id, row, song in load_songs(args.dataset, args.countries):
            n_rows += 1
            if isinstance(song, Exception):
                writer.writerow([song_id, row.get("country"), row.get("title"), "error", "", "", str(song)])
                n_errors += 1
                continue
            samples.append((song.id, note_features(song.notes), song_tags(song)))
            per_country[song.country] += 1
            writer.writerow([song.id, song.country, song.title, "warning" if song.warnings else "ok",
                             len(song.notes), song.n_phrases, " | ".join(song.warnings)])
    if not n_rows:
        raise SystemExit("no songs found in {}".format(args.dataset))

    random.Random(args.seed).shuffle(samples)
    ids, sequences, tags = zip(*samples) if samples else ([], [], [])

    save_json_file(build_vocab(sequences), join(args.corpus_dir, "vocab.json"))
    save_json_file(TAGS, join(args.corpus_dir, "tags.json"))
    save_json_file(list(ids), join(args.corpus_dir, "songs.json"))
    with open(join(args.corpus_dir, "dataset.txt"), "w", encoding="utf8") as fp:
        for seq, ts in zip(sequences, tags):
            fp.write(json.dumps(seq, ensure_ascii=False) + "\t" + json.dumps(ts) + "\n")

    # the original Preprocessor caches the vectorised dataset: drop stale caches
    for cache in glob.glob(join(args.corpus_dir, "dataset_cache_*.npz")):
        os.remove(cache)

    n_notes = sum(len(s) for s in sequences)
    n_ends = sum(t.count(TAGS[1]) for t in tags)
    print("dataset rows: {} | songs: {} | errors: {} (see report.csv)".format(n_rows, len(samples), n_errors))
    print("songs per country: {}".format(", ".join("{}={}".format(c, n) for c, n in sorted(per_country.items()))))
    if samples:
        print("notes: {} | phrase ends: {} ({:.1f}%) | longest song: {} notes".format(
            n_notes, n_ends, 100 * n_ends / n_notes, max(len(s) for s in sequences)))
    print("corpus => {}".format(args.corpus_dir))


if __name__ == "__main__":
    main()