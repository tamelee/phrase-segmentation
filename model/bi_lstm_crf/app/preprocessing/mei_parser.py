"""
MEI parser (EA-Digifolk / co_POEM files) for phrase segmentation.

For each song it returns the sequence of notes in the order they are *sung*
(i.e. with repeats and endings unfolded) and, for each note, the binary label
used in the TransPhrase / BiLSTM-CRF papers:

    y_i = 1  if note i is the last note of a phrase
    y_i = 0  otherwise

Phrases come from <supplied type="phrases"><phrase startid=".." endid=".."/>.
startid/endid point to *written* notes. Since the same written note can be sung
several times (repeats), phrases are mapped greedily, in phrase order, onto the
unfolded sequence.

Unfolding strategies (all are tried by `parse_mei`; the winner is the one that maps all the
annotated phrases and, among those, covers the most notes; ties go to the first one):
  - "minimal": each repeated block is played once per <ending> it has
               (1st ending, 2nd ending, ...); without endings it is played twice.
               This matches the annotations (e.g. PT-1998-PT-DM-011).
  - "written": written order, no repeats. Matches songs whose whole-song repeat is
               annotated only once (e.g. PT-1998-XX-DM-009).

Phrases are mapped in document order (or by @n when all @n are integers). Sub-numbered phrases
("1.1", "4.1") are variants of a phrase; if the mapping fails with them, it is retried without them.
"""
from __future__ import annotations

import glob
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, asdict
from fractions import Fraction
from typing import Optional

MEI_NS = "http://www.music-encoding.org/ns/mei"
XML_ID = "{http://www.w3.org/XML/1998/namespace}id"
NS = {"mei": MEI_NS}

STEP = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}
ACCID = {"s": 1, "f": -1, "ss": 2, "x": 2, "ff": -2, "n": 0, "xs": 3, "ts": 3,
         "tf": -3, "nf": -1, "ns": 1, "su": 1, "sd": 1, "fu": -1, "fd": -1}
SHARP_ORDER = "fcgdaeb"
FLAT_ORDER = "beadgcf"
DUR = {"maxima": 32, "long": 16, "breve": 8}


def _tag(el) -> str:
    return el.tag.split("}", 1)[-1]


def _id(el) -> Optional[str]:
    return el.get(XML_ID)


def _strip(ref: Optional[str]) -> Optional[str]:
    if ref is None:
        return None
    ref = ref.strip().split()[0]
    return ref[1:] if ref.startswith("#") else ref.split("#")[-1]


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #
@dataclass
class Event:
    """A note or rest, in unfolded order."""
    id: str
    is_rest: bool
    midi: Optional[int]          # None for rests
    name: str                    # e.g. "G4", "F#4", "rest"
    dur: float                   # duration in quarter lengths
    offset: float                # position within the bar (quarter lengths)
    meter: str                   # e.g. "4/4"
    bar_len: float
    measure: str                 # measure @n
    fermata: bool = False
    aliases: list = field(default_factory=list)   # other xml:ids that point to this event


@dataclass
class Song:
    id: str
    file: str
    title: str
    strategy: str
    notes: list                  # list of dicts (notes only, rests removed) — see _to_notes
    labels: list                 # 0/1 per note
    phrases: list                # [{"n","type","start","end"}] indexes into `notes`
    warnings: list


# --------------------------------------------------------------------------- #
# Meter / key signature context
# --------------------------------------------------------------------------- #
def _keysig_map(sig: Optional[str]) -> dict:
    if not sig or sig in ("0", "mixed"):
        return {}
    num, acc = int(sig[:-1]), sig[-1]
    order, val = (SHARP_ORDER, 1) if acc == "s" else (FLAT_ORDER, -1)
    return {p: val for p in order[:num]}


def _read_defs(el, state: dict):
    """Update meter/key from scoreDef/staffDef (and keySig/meterSig children)."""
    for d in [el] + list(el.iter()):
        t = _tag(d)
        if t in ("scoreDef", "staffDef"):
            if d.get("meter.count") and d.get("meter.unit"):
                state["meter"] = (int(d.get("meter.count").split("+")[0]), int(d.get("meter.unit")))
            if d.get("key.sig"):
                state["key"] = _keysig_map(d.get("key.sig"))
        elif t == "meterSig" and d.get("count") and d.get("unit"):
            try:
                cnt = sum(int(x) for x in d.get("count").split("+"))
                state["meter"] = (cnt, int(d.get("unit")))
            except ValueError:
                pass
        elif t == "keySig" and d.get("sig") is not None:
            state["key"] = _keysig_map(d.get("sig"))
        if t == "staffDef" and d.get("ppq"):
            state["ppq"] = int(d.get("ppq"))


# --------------------------------------------------------------------------- #
# Reading one measure
# --------------------------------------------------------------------------- #
def _alter(note, key_map, measure_acc) -> int:
    """Pitch alteration (semitones) from accidentals, key signature and earlier accidentals in the bar."""
    pname, oct_ = note.get("pname"), note.get("oct")
    ges = note.get("accid.ges")
    wr = note.get("accid")
    for a in note.findall("mei:accid", NS):
        ges = ges or a.get("accid.ges")
        wr = wr or a.get("accid")
    if ges is not None:
        return ACCID.get(ges, 0)
    if wr is not None:
        v = ACCID.get(wr, 0)
        measure_acc[(pname, oct_)] = v
        return v
    if (pname, oct_) in measure_acc:
        return measure_acc[(pname, oct_)]
    return key_map.get(pname, 0)


def _ql(el, ppq, tuplet: Fraction, fallback_note=None) -> Fraction:
    """Duration in quarter lengths."""
    if el.get("grace"):
        return Fraction(0)
    if ppq and el.get("dur.ppq") is not None:
        return Fraction(int(el.get("dur.ppq")), ppq)
    dur = el.get("dur") or (fallback_note.get("dur") if fallback_note is not None else None)
    if dur is None:
        return Fraction(0)
    base = Fraction(DUR[dur]) if dur in DUR else Fraction(4, int(dur))
    dots = el.get("dots") or (fallback_note.get("dots") if fallback_note is not None else None)
    dots = int(dots) if dots else len(el.findall("mei:dot", NS))
    total = base * (2 - Fraction(1, 2 ** dots))
    return total * tuplet


def _pitch_name(note, alter) -> str:
    acc = {1: "#", -1: "b", 2: "##", -2: "bb"}.get(alter, "")
    return f"{note.get('pname', '?').upper()}{acc}{note.get('oct', '?')}"


def _read_measure(measure, state, staff_n=None):
    """Return (measure events, map id -> local index, total duration)."""
    meter = state["meter"]
    bar_len = Fraction(4 * meter[0], meter[1])
    ppq = state.get("ppq")
    key_map = state["key"]

    staves = measure.findall("mei:staff", NS)
    if not staves:
        return [], {}, Fraction(0)
    staff = staves[0] if staff_n is None else next(
        (s for s in staves if s.get("n") == staff_n), staves[0])
    layers = staff.findall("mei:layer", NS)
    if not layers:
        return [], {}, Fraction(0)
    layer = layers[0]

    fermata_ids = {_strip(f.get("startid")) for f in measure.iter(f"{{{MEI_NS}}}fermata")}

    events, idmap, measure_acc = [], {}, {}
    cur = Fraction(0)

    def add_event(eid, is_rest, midi, name, dur, el, extra_ids=()):
        nonlocal cur
        ev = dict(id=eid, is_rest=is_rest, midi=midi, name=name, dur=dur, offset=cur,
                  fermata=bool(el.get("fermata")) or eid in fermata_ids or any(i in fermata_ids for i in extra_ids))
        idx = len(events)
        events.append(ev)
        idmap[eid] = idx
        for i in extra_ids:
            idmap[i] = idx
        cur += dur

    def walk(el, tuplet: Fraction):
        t = _tag(el)
        if t == "tuplet":
            num, numbase = el.get("num"), el.get("numbase")
            ratio = Fraction(int(numbase), int(num)) if num and numbase else Fraction(1)
            for c in el:
                walk(c, tuplet * ratio)
        elif t in ("beam", "ftrem", "bTrem", "fTrem", "graceGrp"):
            for c in el:
                walk(c, tuplet)
        elif t == "note":
            if el.get("grace"):            # grace notes: alias of the previous note
                if events and _id(el):
                    idmap[_id(el)] = len(events) - 1
                return
            alt = _alter(el, key_map, measure_acc)
            midi = (int(el.get("oct")) + 1) * 12 + STEP[el.get("pname").lower()] + alt
            add_event(_id(el), False, midi, _pitch_name(el, alt), _ql(el, ppq, tuplet), el)
        elif t == "chord":
            notes = el.findall("mei:note", NS)
            if not notes:
                return
            if el.get("grace"):
                for n in notes:
                    if events and _id(n):
                        idmap[_id(n)] = len(events) - 1
                return
            best, best_midi, best_alt = None, -1, 0
            for n in notes:            # melody = highest note of the chord
                alt = _alter(n, key_map, measure_acc)
                m = (int(n.get("oct")) + 1) * 12 + STEP[n.get("pname").lower()] + alt
                if m > best_midi:
                    best, best_midi, best_alt = n, m, alt
            dur = _ql(el, ppq, tuplet, fallback_note=notes[0])
            add_event(_id(el), False, best_midi, _pitch_name(best, best_alt), dur, el,
                      extra_ids=[_id(n) for n in notes if _id(n)])
        elif t in ("rest", "space"):
            add_event(_id(el) or f"_rest_{id(el)}", True, None, "rest", _ql(el, ppq, tuplet), el)
        elif t in ("mRest", "mSpace", "multiRest"):
            add_event(_id(el) or f"_mrest_{id(el)}", True, None, "rest", bar_len, el)

    for c in layer:
        walk(c, Fraction(1))

    total = cur
    return events, idmap, total


# --------------------------------------------------------------------------- #
# Unfolding repeats / endings
# --------------------------------------------------------------------------- #
def _flatten(section, items, state, group_counter):
    """Flat list of measures with <ending> information."""
    run = None  # current group of consecutive endings
    for child in section:
        t = _tag(child)
        if t == "section":
            run = None
            _flatten(child, items, state, group_counter)
        elif t in ("scoreDef", "staffDef"):
            _read_defs(child, state)
        elif t == "measure":
            run = None
            items.append(dict(measure=child, group=None, k=None, state=dict(state)))
        elif t == "ending":
            if run is None:
                group_counter[0] += 1
                run = dict(gid=group_counter[0], count=0, members=[])
            k = run["count"]
            run["count"] += 1
            for m in child.iter(f"{{{MEI_NS}}}measure"):
                it = dict(measure=m, group=run["gid"], k=k, state=dict(state), run=run)
                run["members"].append(it)
                items.append(it)
        # sb, pb, supplied, annot, ... are ignored (they do not break a group of endings)
    return items


def _unfold_minimal(items):
    out, i, start, pass_ = [], 0, 0, 0
    n, guard = len(items), 0
    while i < n:
        guard += 1
        if guard > 50 * n + 100:
            raise RuntimeError("unfolding did not terminate (unexpected repeat structure)")
        it = items[i]
        m = it["measure"]
        left, right = m.get("left"), m.get("right")
        if left in ("rptstart", "rptboth") and not it.get("_landed"):
            start, pass_ = i, 0
        it.pop("_landed", None)

        if it["group"] is not None:
            nk = it["run"]["count"]
            want = min(pass_, nk - 1)
            if it["k"] != want:
                i += 1
                continue
        out.append(it)

        if right in ("rptend", "rptboth"):
            if it["group"] is not None:
                do_jump = it["k"] < it["run"]["count"] - 1
            else:
                do_jump = pass_ == 0
            if do_jump:
                pass_ += 1
                i = start
                items[i]["_landed"] = True
                continue
            # repeat finished
            pass_ = 0
            start = i + 1
        elif it["group"] is not None and it["k"] == it["run"]["count"] - 1:
            # last ending played (no repeat barline): block finished
            nxt = i + 1
            if nxt >= n or items[nxt]["group"] != it["group"]:
                pass_ = 0
                start = nxt
        i += 1
    return out


def _unfold_written(items):
    return list(items)


# --------------------------------------------------------------------------- #
# Building the event sequence and mapping the phrases
# --------------------------------------------------------------------------- #
def _build_events(measure_items, staff_n=None):
    events = []
    first = True
    for it in measure_items:
        st = it["state"]
        evs, idmap, total = _read_measure(it["measure"], st, staff_n)
        meter = st["meter"]
        bar_len = Fraction(4 * meter[0], meter[1])
        shift = Fraction(0)
        if first and 0 < total < bar_len:          # anacrusis (pickup): right-align in the bar
            shift = bar_len - total
        if evs:
            first = False
        inv = {}
        for eid, idx in idmap.items():
            inv.setdefault(idx, []).append(eid)
        for idx, ev in enumerate(evs):
            events.append(Event(
                id=ev["id"], is_rest=ev["is_rest"], midi=ev["midi"], name=ev["name"],
                dur=float(ev["dur"]), offset=float(ev["offset"] + shift),
                meter=f"{meter[0]}/{meter[1]}", bar_len=float(bar_len),
                measure=it["measure"].get("n", "?"), fermata=ev["fermata"],
                aliases=[a for a in inv.get(idx, []) if a != ev["id"]]))
    return events


def _read_phrases(score):
    phrases = []
    for ph in score.iter(f"{{{MEI_NS}}}phrase"):
        s, e = _strip(ph.get("startid")), _strip(ph.get("endid"))
        if s and e:
            phrases.append(dict(n=ph.get("n"), type=ph.get("type"), startid=s, endid=e))
    if phrases and all(p["n"] and p["n"].isdigit() for p in phrases):
        phrases.sort(key=lambda p: int(p["n"]))
    # otherwise keep document order: sub-numbered phrases ("1.1", "4.1") are variants of a phrase
    # (another stanza / version), listed after the main phrases, not sung right after them
    return phrases


def _map_phrases(events, phrases):
    """Greedy mapping: each phrase starts after the end of the previous one."""
    pos = {}
    for i, ev in enumerate(events):
        for eid in [ev.id] + ev.aliases:
            pos.setdefault(eid, []).append(i)

    spans, cursor = [], 0
    for ph in phrases:
        s_list = [i for i in pos.get(ph["startid"], []) if i >= cursor]
        if not s_list:
            return None, f"startid #{ph['startid']} (phrase {ph['n']}) not found after position {cursor}"
        s = s_list[0]
        e_list = [i for i in pos.get(ph["endid"], []) if i >= s]
        if not e_list:
            return None, f"endid #{ph['endid']} (phrase {ph['n']}) not found after its start"
        e = e_list[0]
        spans.append(dict(n=ph["n"], type=ph["type"], start=s, end=e))
        cursor = e + 1
    return spans, None


def _to_notes(events, spans):
    """Drop rests (they become the `rest_after` feature) and convert event indexes -> note indexes."""
    ev2note, notes = {}, []
    for i, ev in enumerate(events):
        if ev.is_rest:
            if notes:
                notes[-1]["rest_after"] += ev.dur
            continue
        ev2note[i] = len(notes)
        d = asdict(ev)
        d.pop("is_rest")
        d["rest_after"] = 0.0
        notes.append(d)

    def first_note_at_or_after(i):
        while i < len(events) and i not in ev2note:
            i += 1
        return ev2note.get(i)

    def last_note_at_or_before(i):
        while i >= 0 and i not in ev2note:
            i -= 1
        return ev2note.get(i)

    labels = [0] * len(notes)
    note_spans = []
    for sp in spans:
        s, e = first_note_at_or_after(sp["start"]), last_note_at_or_before(sp["end"])
        if s is None or e is None or e < s:
            continue
        labels[e] = 1
        note_spans.append(dict(n=sp["n"], type=sp["type"], start=s, end=e))
    return notes, labels, note_spans


def _initial_state(score):
    state = {"meter": (4, 4), "key": {}, "ppq": None}
    sd = score.find("mei:scoreDef", NS)
    if sd is not None:
        _read_defs(sd, state)
    return state


def parse_mei(path, strategies=("minimal", "written"), require_phrases=True) -> Song:
    tree = ET.parse(path)
    root = tree.getroot()
    title_el = root.find(".//mei:titleStmt/mei:title", NS)
    title = (title_el.text or "").strip() if title_el is not None else ""
    song_id = (_id(title_el) if title_el is not None else None) or path.rsplit("/", 1)[-1].rsplit(".", 1)[0]

    score = root.find(".//mei:music//mei:score", NS)
    if score is None:
        raise ValueError("no <score> element")
    phrases = _read_phrases(score)
    if require_phrases and not phrases:
        raise ValueError("no phrase annotations (<phrase startid endid>)")

    warnings, best = [], None
    for strategy in strategies:
        state = _initial_state(score)
        items = []
        for sec in score.findall("mei:section", NS):
            _flatten(sec, items, state, [0])
        measure_items = _unfold_minimal(items) if strategy == "minimal" else _unfold_written(items)
        events = _build_events(measure_items)
        if not phrases:
            notes, labels, spans = _to_notes(events, [])
            return Song(song_id, path, title, strategy, notes, labels, spans, warnings)
        spans, err = _map_phrases(events, phrases)
        dropped = False
        if spans is None:
            warnings.append(f"[{strategy}] {err}")
            main = [p for p in phrases if not (p["n"] and "." in p["n"])]
            if not main or len(main) == len(phrases):
                continue
            # retry without the variant phrases ("x.y")
            spans, err = _map_phrases(events, main)
            if spans is None:
                warnings.append(f"[{strategy}] without variants: {err}")
                continue
            dropped = True
        notes, labels, note_spans = _to_notes(events, spans)
        covered = sum(sp["end"] - sp["start"] + 1 for sp in note_spans)
        # rank: mapping all annotated phrases first, then the fraction of notes covered by phrases
        coverage = (not dropped, covered / max(len(notes), 1))
        if best is None or coverage > best[0]:
            msg = [f"[{strategy}] variant phrases (x.y) ignored"] if dropped else []
            best = (coverage, Song(song_id, path, title, strategy, notes, labels, note_spans, msg))
        if coverage == (True, 1):
            break

    if best is None:
        raise ValueError("could not map the phrases: " + " | ".join(warnings))
    song = best[1]
    covered = sum(sp["end"] - sp["start"] + 1 for sp in song.phrases)
    if covered < 0.8 * len(song.notes):
        song.warnings.append(f"[{song.strategy}] phrases only cover {covered}/{len(song.notes)} notes")
    return song


def find_mei_files(inputs):
    """MEI files from a list of files and/or directories (searched recursively)."""
    files = []
    for path in inputs:
        if os.path.isdir(path):
            files += sorted(glob.glob(os.path.join(path, "**", "*.mei"), recursive=True))
        else:
            files.append(path)
    return files
