"""Minimal Needle-in-a-Haystack generator (single-needle, RULER-style).

Generates jsonl cases at user-specified token lengths. Tokenization uses the
target model's tokenizer to ensure exact length.
"""
import argparse
import json
import random
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from transformers import AutoTokenizer
from sprag.loader import DEFAULT_MODEL_PATH


# Filler sentences — generic factual text to bulk up the haystack.
FILLERS = [
    "The history of computing has roots in attempts to mechanize counting in the 1800s.",
    "Charles Babbage designed the Difference Engine but never completed a full version.",
    "Ada Lovelace wrote algorithmic notes for Babbage's Analytical Engine in 1843.",
    "Alan Turing introduced the abstract concept of a universal computing machine.",
    "John von Neumann formalised the stored-program architecture in his 1945 draft.",
    "The ENIAC was unveiled to the public at the University of Pennsylvania in 1946.",
    "Grace Hopper led the team that wrote the first compiler called A-0 in 1952.",
    "The transistor was invented at Bell Labs by Bardeen, Brattain and Shockley.",
    "Integrated circuits emerged in the late 1950s through work by Kilby and Noyce.",
    "The IBM System/360 family unified mainframe architecture beginning in 1964.",
    "Ken Thompson and Dennis Ritchie developed Unix at Bell Labs starting in 1969.",
    "C was created by Dennis Ritchie between 1969 and 1973 as a systems language.",
    "The ARPANET sent its first message between two nodes in October of 1969.",
    "Xerox PARC researchers prototyped the modern desktop interface in the 1970s.",
    "Vint Cerf and Bob Kahn designed the TCP/IP protocol suite during the late 1970s.",
    "The IBM PC was released in August of 1981 and rapidly defined a standard.",
    "Tim Berners-Lee invented the World Wide Web at CERN in 1989 and 1990.",
    "Linus Torvalds released the first version of the Linux kernel in 1991.",
    "The first commercial web browser, Netscape Navigator, debuted in late 1994.",
    "Google was founded by Larry Page and Sergey Brin at Stanford in September 1998.",
]

# Needle templates: (template, fact_keys, question)
NEEDLES = [
    (
        "Buried in this document is a special note: the magic number stored in vault "
        "{vault} is {number}. Remember this carefully.",
        {"vault": ["Alpha", "Beta", "Gamma", "Delta", "Sigma"],
         "number": ["forty-two", "seventeen", "ninety-three", "one hundred and one"]},
        "What magic number is stored in vault {vault}?",
        "{number}",
    ),
    (
        "Important fact embedded in the haystack: the secret keeper is {name} and "
        "her favourite colour is {colour}. Please remember this for later.",
        {"name": ["Octavia", "Bridget", "Lenore", "Helena", "Marigold"],
         "colour": ["scarlet", "indigo", "amber", "cerulean", "viridian"]},
        "Who is the secret keeper and what is her favourite colour?",
        "{name} ... {colour}",
    ),
    (
        "Take note: the best place to find a quiet bookshop in {city} is on "
        "{street} Street. This information is unique to this document.",
        {"city": ["Lisbon", "Kyoto", "Tallinn", "Reykjavik", "Montevideo"],
         "street": ["Linden", "Mulberry", "Saffron", "Hawthorn", "Marigold"]},
        "Where is the best quiet bookshop in {city}?",
        "{street}",
    ),
]


def fill_needle(rng):
    tmpl, keys, q_tmpl, ans_tmpl = rng.choice(NEEDLES)
    picks = {k: rng.choice(v) for k, v in keys.items()}
    return (tmpl.format(**picks), q_tmpl.format(**picks), ans_tmpl.format(**picks),
            picks)


def make_case(tok, target_tokens: int, rng):
    needle_text, question, answer, picks = fill_needle(rng)

    # Build filler haystack to ~target_tokens (leaving room for needle + Q + slack)
    needle_tokens = len(tok(needle_text).input_ids)
    q_tokens = len(tok(question).input_ids)
    budget = target_tokens - needle_tokens - q_tokens - 32  # safety

    pieces = []
    used = 0
    while used < budget:
        s = rng.choice(FILLERS)
        n = len(tok(" " + s).input_ids)
        if used + n > budget + 16:
            break
        pieces.append(s)
        used += n

    # Insert needle at a random depth between 25% and 75%
    depth = rng.uniform(0.25, 0.75)
    insert_at = int(len(pieces) * depth)
    pieces.insert(insert_at, needle_text)
    haystack = " ".join(pieces)

    total = len(tok(haystack).input_ids)
    return {
        "haystack": haystack,
        "question": question,
        "answer": answer,
        "answer_picks": picks,
        "needle_depth": depth,
        "haystack_tokens": total,
        "needle_position_pieces": insert_at,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--target_tokens", type=int, default=4096)
    ap.add_argument("--n_cases", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(DEFAULT_MODEL_PATH)
    rng = random.Random(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for i in range(args.n_cases):
            case = make_case(tok, args.target_tokens, rng)
            case["id"] = i
            f.write(json.dumps(case) + "\n")
    print(f"Wrote {args.n_cases} cases ({args.target_tokens} tok target) -> {args.out}")


if __name__ == "__main__":
    main()
