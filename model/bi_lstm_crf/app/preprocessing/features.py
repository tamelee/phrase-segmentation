"""
ADDED: note features and tags for phrase segmentation.

Each note becomes a list of categorical feature values (stored as strings, like the tokens
of the original text corpus). Inspired by the TransPhrase 5-dim note vector
{bar, position, pitch, duration, time signature}, plus a few features that help with
phrase boundaries.

IMPORTANT: "pitch" must stay the FIRST feature — the model uses feature 0 to build the
padding mask (a real note never has index 0).
"""
import math

FEATURES = ["pitch", "interval", "dur", "dur_ratio", "pos", "meter", "rest_after", "fermata"]

# tags: "E" = last note of a phrase (y = 1 in the paper), "O" = any other note (y = 0)
TAG_OTHER = "O"
TAG_END = "E"
TAGS = [TAG_OTHER, TAG_END]

# fixed vocabularies (independent of the corpus, so transposed / unseen songs never go OOV)
FIXED_VOCABS = {
    "pitch": [str(m) for m in range(128)],                          # MIDI pitch
    "interval": ["START"] + [str(i) for i in range(-24, 25)],       # semitones, clipped to ±2 octaves
    "fermata": ["0", "1"],
}


def _q(x):
    """Quantise a float (duration / offset) into a stable string."""
    return "{:g}".format(round(float(x), 3))


def note_features(notes):
    """notes (dicts from mei_parser) -> [[feature values in FEATURES order], ...]"""
    out = []
    prev_midi, prev_dur = None, None
    for n in notes:
        midi, dur = n["midi"], n["dur"]
        if prev_midi is None:
            interval, ratio = "START", "START"
        else:
            interval = str(max(-24, min(24, midi - prev_midi)))
            ratio = "0" if prev_dur <= 0 or dur <= 0 else str(max(-4, min(4, round(math.log2(dur / prev_dur)))))
        values = {
            "pitch": str(midi),
            "interval": interval,                      # melodic interval to the previous note
            "dur": _q(dur),                            # duration (quarter lengths)
            "dur_ratio": ratio,                        # log2(dur / previous dur), rounded
            "pos": _q(n["offset"]),                    # position within the bar
            "meter": n["meter"],                       # time signature
            "rest_after": _q(n.get("rest_after", 0)),  # rest following the note (rests are not tokens)
            "fermata": str(int(bool(n.get("fermata")))),
        }
        out.append([values[f] for f in FEATURES])
        prev_midi, prev_dur = midi, dur
    return out


def song_tags(song):
    """0/1 labels of a parsed song -> tag strings"""
    return [TAG_END if y else TAG_OTHER for y in song.labels]


def build_vocab(sequences):
    """{feature: [values]} — fixed vocabularies + values seen in the corpus for the others."""
    vocab = {}
    for i, f in enumerate(FEATURES):
        if f in FIXED_VOCABS:
            vocab[f] = list(FIXED_VOCABS[f])
        else:
            vocab[f] = sorted({note[i] for seq in sequences for note in seq})
    return vocab
