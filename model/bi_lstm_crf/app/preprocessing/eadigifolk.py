"""
ADDED: loader for the compiled EA-Digifolk dataset (EADIGIFOLK.gzip).

The file is a gzip-pickled pandas DataFrame produced by the EA-Digifolk MEI parser, with one row
per song (index = song id, e.g. "PT-1998-PT-DM-011") and a `country` column (PT / ES / MX).
Per-note data is in the `features` column: a dict of equal-length lists, one entry per note,
already in sung order (repeats unfolded by that parser). The fields used here are:

    midipitch          MIDI pitch
    duration           duration in quarter lengths (string, e.g. "0.25")
    posinbar           position within the bar, as a fraction of the bar (0 <= x < 1)
    timesignature      e.g. "3/8"
    restduration_frac  rest after the note (string fraction, e.g. "1/2") or None
    phrase_end         True if the note is the last note of a phrase  -> label y = 1

Each song is converted into the same note dicts the old MEI parser produced
({midi, dur, offset, meter, rest_after, fermata}) so features.note_features is unchanged.
The dataset has no fermata information: `fermata` is always False.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

import pandas as pd

REQUIRED = ("midipitch", "duration", "posinbar", "timesignature", "phrase_end")


@dataclass
class Song:
    id: str
    title: str
    country: str
    notes: list                  # [{midi, dur, offset, meter, rest_after, fermata}, ...]
    labels: list                 # 0/1 per note (1 = last note of a phrase)
    warnings: list = field(default_factory=list)

    @property
    def n_phrases(self):
        return sum(self.labels)


def _num(x) -> float:
    """'3/2', '1.5', 1.5, None -> float (None -> 0)."""
    if x is None or x == "":
        return 0.0
    return float(Fraction(str(x)))


def _bar_len(ts: str) -> float:
    num, den = ts.split("/")
    return 4 * int(num) / int(den)


def load_dataframe(path):
    """Read the compiled dataset (a gzip-pickled pandas DataFrame)."""
    return pd.read_pickle(path, compression="gzip")


def row_to_song(song_id, row) -> Song:
    """One DataFrame row -> Song. Raises ValueError if the row cannot be used."""
    f = row["features"]
    if not f:
        raise ValueError("no features")
    missing = [k for k in REQUIRED if k not in f]
    if missing:
        raise ValueError("missing features: {}".format(", ".join(missing)))
    n = len(f["midipitch"])
    if n == 0:
        raise ValueError("no notes")
    if any(len(f[k]) != n for k in REQUIRED):
        raise ValueError("feature lists have different lengths")

    rests = f.get("restduration_frac") or [None] * n
    notes = []
    for i in range(n):
        ts = str(f["timesignature"][i])
        notes.append(dict(
            midi=int(f["midipitch"][i]),
            dur=_num(f["duration"][i]),
            offset=float(f["posinbar"][i]) * _bar_len(ts),   # quarter lengths from the barline
            meter=ts,
            rest_after=_num(rests[i]),
            fermata=False,
        ))
    labels = [int(bool(x)) for x in f["phrase_end"]]

    warnings = []
    if sum(labels) == 0:
        raise ValueError("no phrase ends (phrase_end is always False)")
    if not labels[-1]:
        warnings.append("last note is not a phrase end")
    n_grace = sum(1 for nt in notes if nt["dur"] == 0)
    if n_grace:
        warnings.append("{} zero-duration (grace) notes".format(n_grace))

    return Song(str(song_id), str(row.get("title") or ""), str(row.get("country") or ""),
                notes, labels, warnings)


def load_songs(path, countries=None):
    """Yield (song_id, row, Song or Exception) for every row of the dataset.

    Repeated song ids (different songs sharing an id) get a suffix: "<id>__2", "<id>__3", ...
    """
    df = load_dataframe(path)
    if countries:
        df = df[df["country"].isin(countries)]
    seen = {}
    for song_id, row in df.iterrows():
        seen[song_id] = seen.get(song_id, 0) + 1
        uid = song_id if seen[song_id] == 1 else "{}__{}".format(song_id, seen[song_id])
        try:
            yield uid, row, row_to_song(uid, row)
        except Exception as e:  # noqa: BLE001 — reported by the caller
            yield uid, row, e