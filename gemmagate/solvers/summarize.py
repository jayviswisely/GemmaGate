"""Extractive summarizer — frequency-scored sentence selection.

Used as the FIRST rung only when the prompt gives an explicit, checkable
length constraint (N sentences / N words) and the passage is short enough
that selection reads coherently. Otherwise summarization goes remote, since
abstractive quality is judged. Also serves as the deadline-pressure fallback.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Optional

from ..schemas import TaskSpec

_STOP = set("""a an the and or but if then than so of to in on at by for from
with as is are was were be been being it its this that these those he she
they we you i his her their our your my not no do does did have has had will
would can could should may might about into over after before while during
which who whom whose what when where why how all any both each few more most
other some such only own same too very just also there here up down out off
again further once because until against between""".split())


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 15]


def summarize(text: str, max_sentences: Optional[int] = None,
              max_words: Optional[int] = None) -> Optional[str]:
    sents = _sentences(text)
    if len(sents) == 1 and max_words:
        s0 = sents[0]
        if len(s0.split()) > max_words:
            # drop a leading adverbial ("After months of negotiation, ...")
            # so the words we keep carry the clause with the actual claim
            intro = re.match(r"(?:After|Before|Following|Despite|Amid|During|"
                             r"Over|With|Since|Although|While|In|On)\b"
                             r"[^,]{0,60},\s*(.+)$", s0, re.I)
            if intro and len(intro.group(1).split()) >= 6:
                s0 = intro.group(1)[0].upper() + intro.group(1)[1:]
        w = s0.split()
        if len(w) <= max_words:
            return s0 if s0.endswith(".") else s0 + "."
        out = " ".join(w[:max_words]).rstrip(",;:")
        # never end on a dangling connector
        out = re.sub(r"\s+(?:and|or|but|the|a|an|of|to|on|in|with|that)$",
                     "", out, flags=re.I)
        return out + ("" if out.endswith(".") else ".")
    if len(sents) < 2:
        return None
    words = [w for w in re.findall(r"[a-z']+", text.lower()) if w not in _STOP]
    freq = Counter(words)
    if not freq:
        return None

    def score(s: str, idx: int) -> float:
        toks = [w for w in re.findall(r"[a-z']+", s.lower()) if w not in _STOP]
        if not toks:
            return 0.0
        base = sum(freq[t] for t in toks) / len(toks)
        if idx == 0:                       # lead sentence usually carries the topic
            base *= 1.25
        return base

    n = max_sentences or max(1, min(3, len(sents) // 3))
    ranked = sorted(range(len(sents)), key=lambda i: score(sents[i], i), reverse=True)

    def toks(s: str) -> set:
        return {w for w in re.findall(r"[a-z']+", s.lower()) if w not in _STOP}

    chosen: list[int] = []
    for i in ranked:                       # greedy pick with redundancy penalty
        cand = toks(sents[i])
        if any(cand and len(cand & toks(sents[j])) / len(cand | toks(sents[j])) > 0.6
               for j in chosen):
            continue                       # near-duplicate of a chosen sentence
        chosen.append(i)
        if len(chosen) == n:
            break
    if len(chosen) < n:                    # backfill if the penalty over-pruned
        for i in ranked:
            if i not in chosen:
                chosen.append(i)
            if len(chosen) == n:
                break
    out = " ".join(sents[i] for i in sorted(chosen))

    if max_words and len(out.split()) > max_words:
        # trim to the last full sentence inside the limit if one exists
        clipped = " ".join(out.split()[:max_words])
        m = re.match(r"^(.*[.!?])\s", clipped + " ")
        out = m.group(1) if m and len(m.group(1).split()) >= max_words // 2 \
            else clipped.rstrip(",;:") + "."
    return out


def solve(spec: TaskSpec) -> Optional[str]:
    text = spec.payload or spec.prompt
    if spec.max_bullets:
        if len(text.split()) > 350:
            return None
        body = summarize(text, max_sentences=spec.max_bullets)
        if not body:
            return None
        sents = _sentences(body)
        if len(sents) < spec.max_bullets:
            return None
        return "\n".join("- " + s for s in sents[:spec.max_bullets])
    # only trust extraction when there's a checkable constraint & short passage
    if not (spec.max_sentences or spec.max_words):
        return None
    if len(text.split()) > 350:
        return None
    return summarize(text, spec.max_sentences, spec.max_words)


def failsafe(spec: TaskSpec) -> str:
    """Best-effort summary for deadline/failure paths (never returns None)."""
    text = spec.payload or spec.prompt
    if spec.max_bullets:
        sents = _sentences(text)[:spec.max_bullets] or [text[:120]]
        while len(sents) < spec.max_bullets:
            sents.append(sents[-1])
        return "\n".join("- " + s for s in sents)
    out = summarize(text, spec.max_sentences or 2, spec.max_words)
    if out:
        return out
    return " ".join(text.split()[: (spec.max_words or 40)])
