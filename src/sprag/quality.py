"""QuALITY loader (NYU v1.0.1 htmlstripped) for the ACC-coverage probe.

Each article is a single coherent ~3-9k-token document with 4-way MC questions
and a `difficult` flag (the speed-vs-untimed-derived subset that genuinely needs
careful/global reading — our long-range-dependency signal, since QuALITY has no
gold evidence spans). MC format → likelihood scoring, no string-match noise.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class QualityQ:
    qid: str
    query: str
    options: list[str]      # 4 options
    gold_idx: int           # 0-based correct option
    difficult: bool


@dataclass
class QualityArticle:
    aid: str
    title: str
    article: str            # the full coherent document
    questions: list[QualityQ]


def load_quality(path: str | Path, difficult_only: bool = False,
                 limit: int | None = None) -> list[QualityArticle]:
    arts: list[QualityArticle] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            qs: list[QualityQ] = []
            for q in r["questions"]:
                gl = q.get("gold_label")
                if not gl:                       # test split hides labels
                    continue
                qs.append(QualityQ(
                    qid=q["question_unique_id"], query=q["question"],
                    options=q["options"], gold_idx=int(gl) - 1,
                    difficult=bool(q.get("difficult", 0))))
            if difficult_only:
                qs = [q for q in qs if q.difficult]
            if not qs:
                continue
            arts.append(QualityArticle(
                aid=str(r["article_id"]), title=r.get("title", ""),
                article=r["article"], questions=qs))
            if limit and len(arts) >= limit:
                break
    return arts
