"""
ADDED: build a corpus directory from MEI files, in the same structure the original code expects
(https://github.com/jidasheng/bi-lstm-crf/wiki/corpus-structure-and-format):

    corpus_dir/
        vocab.json     {feature: [values]}   (one vocabulary per note feature)
        tags.json      ["O", "E"]
        dataset.txt    one song per line: <json notes>\\t<json tags>
        songs.json     song ids, in the same order as dataset.txt (for tracing results)
        report.csv     status of every MEI file (ok / warning / error)

Usage:
    python -m bi_lstm_crf.app.preprocessing.build_corpus mei_dir corpus_dir

Songs are shuffled (with --seed) so that the contiguous train/val/test split done by the
original Preprocessor.load_dataset is not biased by the file names.
"""
import argparse
import csv
import glob
import os
import random
from os.path import join

from bi_lstm_crf.app.preprocessing.features import TAGS, build_vocab, note_features, song_tags
from bi_lstm_crf.app.preprocessing.mei_parser import find_mei_files, parse_mei
from bi_lstm_crf.app.preprocessing.utils import json, save_json_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", help="MEI files and/or directories (searched recursively)")
    parser.add_argument("corpus_dir", type=str, help="output corpus directory")
    parser.add_argument("--seed", type=int, default=42, help="seed used to shuffle the songs")
    args = parser.parse_args()

    files = find_mei_files(args.inputs)
    if not files:
        raise SystemExit("no .mei files found")
    os.makedirs(args.corpus_dir, exist_ok=True)

    samples, n_errors = [], 0
    with open(join(args.corpus_dir, "report.csv"), "w", newline="", encoding="utf8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["file", "status", "strategy", "notes", "phrases", "message"])
        for f in files:
            try:
                song = parse_mei(f)
                if not song.notes:
                    raise ValueError("no notes")
            except Exception as e:  # noqa: BLE001 — log any failure and keep going
                writer.writerow([f, "error", "", "", "", str(e)])
                n_errors += 1
                continue
            samples.append((song.id, note_features(song.notes), song_tags(song)))
            writer.writerow([f, "warning" if song.warnings else "ok", song.strategy, len(song.notes),
                             len(song.phrases), " | ".join(song.warnings)])

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
    print("files: {} | songs: {} | errors: {} (see report.csv)".format(len(files), len(samples), n_errors))
    if samples:
        print("notes: {} | phrase ends: {} ({:.1f}%) | longest song: {} notes".format(
            n_notes, n_ends, 100 * n_ends / n_notes, max(len(s) for s in sequences)))
    print("corpus => {}".format(args.corpus_dir))


if __name__ == "__main__":
    main()
