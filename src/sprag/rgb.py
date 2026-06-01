"""RGB (Retrieval-Augmented Generation Benchmark) loader + scorer.

RGB ships one JSON object per line (chen700564/RGB, data/en.json etc.):
  {"id", "query", "answer", "positive": [...passages], "negative": [...passages]}

`answer` comes in two shapes:
  - flat list of strings: each string is a *required* answer slot, single alias
    e.g. ["medical"]  or  ["Lawrence Williams", "Ralph Long Jr.", ...] (4 slots)
  - nested list of lists: each inner list is one slot's acceptable aliases
    e.g. [["January 2 2022", "Jan 2, 2022", ...]]   (1 slot, many aliases)

Scoring mirrors RGB's official checkanswer: an instance is correct iff *every*
slot has at least one alias appearing (case-insensitive substring) in the model
output. This is the noise-robustness / information-integration accuracy.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RGBRecord:
    rid: str
    query: str
    slots: list[list[str]]      # normalized: list of slots, each a list of aliases
    positive: list[str]         # gold passages
    negative: list[str]         # distractor passages

    def passages_shuffled(self, seed: int) -> tuple[list[str], list[bool]]:
        """Return (passages, is_gold) shuffled deterministically by seed.

        is_gold[i] is True if passages[i] came from the positive set — used for
        oracle retrieval and for tagging the assembled doc."""
        tagged = [(p, True) for p in self.positive] + [(p, False) for p in self.negative]
        random.Random(seed).shuffle(tagged)
        return [p for p, _ in tagged], [g for _, g in tagged]


def normalize_answer(answer) -> list[list[str]]:
    """answer -> list of slots; each slot is a list of acceptable alias strings."""
    slots: list[list[str]] = []
    for a in answer:
        if isinstance(a, list):
            slots.append([str(x) for x in a])
        else:
            slots.append([str(a)])
    return slots


def load_rgb(path: str | Path, limit: int | None = None) -> list[RGBRecord]:
    recs: list[RGBRecord] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            recs.append(RGBRecord(
                rid=str(r.get("id", len(recs))),
                query=r["query"],
                slots=normalize_answer(r["answer"]),
                positive=list(r.get("positive", [])),
                negative=list(r.get("negative", [])),
            ))
            if limit and len(recs) >= limit:
                break
    return recs


def matches(output: str, slots: list[list[str]]) -> bool:
    """Correct iff every slot has >=1 alias as a case-insensitive substring."""
    lo = output.lower()
    return all(any(alias.lower() in lo for alias in slot) for slot in slots)


def any_slot_alias_in(text: str, slots: list[list[str]]) -> bool:
    """True if ANY answer alias appears in `text` — used to tag oracle chunks."""
    lo = text.lower()
    return any(any(alias.lower() in lo for alias in slot) for slot in slots)
