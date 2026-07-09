"""Logic puzzle solver — permutation brute force over parsed constraints.

Scope (deliberately narrow, high precision): ordering puzzles over a small
set of named entities with comparative or positional constraints:

    "Alice is taller than Bob."      -> alice > bob
    "Bob finished before Carol."     -> bob < carol   (earlier = smaller)
    "Dan is to the left of Erin."    -> dan < erin
    "Bob is not last."               -> position constraint

If the parser extracts >= 2 entities and >= 1 constraint, we enumerate all
permutations (n <= 8 => at most 40320 checks, microseconds) and keep the ones
satisfying every constraint. We answer ONLY if the surviving solutions agree
on what the question asks (a superlative, a specific position, or the full
order). Any ambiguity -> None -> escalate. This makes the solver's answers
essentially provably correct, at zero tokens.
"""
from __future__ import annotations

import itertools
import re
from typing import Optional

from ..schemas import TaskSpec

# "greater" side is index-larger in the ordering
_GT_REL = re.compile(
    r"\b([A-Z][a-z]+)\s+(?:is|was|runs?|works?)?\s*(?:much\s+)?"
    r"(taller|older|faster|bigger|heavier|stronger|richer|higher|larger)\s+than\s+([A-Z][a-z]+)")
_LT_REL = re.compile(
    r"\b([A-Z][a-z]+)\s+(?:is|was)?\s*"
    r"(shorter|younger|slower|smaller|lighter|weaker|poorer|lower)\s+than\s+([A-Z][a-z]+)")
_BEFORE_REL = re.compile(
    r"\b([A-Z][a-z]+)\s+(?:finished|arrived|came|ranked|is|was|sat)?\s*"
    r"(?:the\s+\w+\s+)?"    # optional object: "finished THE RACE before"
    r"(?:just\s+)?(before|after|to the left of|to the right of|ahead of|behind)\s+([A-Z][a-z]+)")
_NOT_POS = re.compile(r"\b([A-Z][a-z]+)\s+(?:is|was|did)\s+not\s+(first|last|second|third)\b", re.I)
_IS_POS = re.compile(r"\b([A-Z][a-z]+)\s+(?:is|was|finished|came)\s+(first|last|second|third)\b", re.I)

_POS_IDX = {"first": 0, "second": 1, "third": 2}

_SUPERLATIVE_Q = re.compile(
    r"who\s+(?:is|was|finished|came|ranks?|arrived)?\s*(?:the\s+)?"
    r"(tallest|oldest|fastest|biggest|heaviest|strongest|richest|highest|largest|"
    r"shortest|youngest|slowest|smallest|lightest|weakest|poorest|lowest|first|last)", re.I)

_LOW_SUPER = {"shortest", "youngest", "slowest", "smallest", "lightest",
              "weakest", "poorest", "lowest", "first"}
_STOPNAMES = {"If", "The", "Who", "What", "Which", "All", "Then", "And",
              "But", "So", "Given", "Assume", "Suppose", "Everyone", "There"}


def _parse(text: str):
    names: list[str] = []

    def note(n: str):
        if n not in names and n not in _STOPNAMES:
            names.append(n)

    gt_pairs = []          # (a, b) meaning a ranks HIGHER than b
    pos_must = {}          # name -> index (0-based from the "low" end)
    pos_not = []           # (name, index)

    for a, _, b in _GT_REL.findall(text):
        note(a); note(b); gt_pairs.append((a, b))
    for a, _, b in _LT_REL.findall(text):
        note(a); note(b); gt_pairs.append((b, a))
    # chained: "Tessa finished after Mo but before Yuri"
    for a, b, c in re.findall(
            r"\b([A-Z][a-z]+)\s+(?:finished|arrived|came|is|was)?\s*after\s+"
            r"([A-Z][a-z]+)\s+but\s+before\s+([A-Z][a-z]+)", text):
        note(a); note(b); note(c)
        gt_pairs.append((a, b))       # a later than b
        gt_pairs.append((c, a))       # c later than a
    for a, rel, b in _BEFORE_REL.findall(text):
        note(a); note(b)
        if rel.lower() in ("before", "to the left of", "ahead of"):
            gt_pairs.append((b, a))   # a earlier => a lower index => b "greater"
        else:
            gt_pairs.append((a, b))
    for name, word in _IS_POS.findall(text):
        if name in _STOPNAMES:
            continue                          # "Who finished last?" is a QUESTION
        note(name)
        pos_must[name] = _POS_IDX.get(word.lower(), -1)  # -1 => last
    for name, word in _NOT_POS.findall(text):
        if name in _STOPNAMES:
            continue
        note(name)
        pos_not.append((name, _POS_IDX.get(word.lower(), -1)))
    return names, gt_pairs, pos_must, pos_not


def solve(spec: TaskSpec) -> Optional[str]:
    text = spec.prompt
    names, gt_pairs, pos_must, pos_not = _parse(text)
    if len(names) < 2 or (not gt_pairs and not pos_must) or len(names) > 8:
        return solve_assignment(spec)

    def ok(order: tuple) -> bool:
        idx = {n: i for i, n in enumerate(order)}   # index 0 = lowest/earliest
        for a, b in gt_pairs:
            if a not in idx or b not in idx or idx[a] <= idx[b]:
                return False
        n = len(order)
        for name, p in pos_must.items():
            want = (n - 1) if p == -1 else p
            if idx.get(name) != want:
                return False
        for name, p in pos_not:
            avoid = (n - 1) if p == -1 else p
            if idx.get(name) == avoid:
                return False
        return True

    solutions = [p for p in itertools.permutations(names) if ok(p)]
    if not solutions:
        return None

    m = _SUPERLATIVE_Q.search(text)
    if m:
        word = m.group(1).lower()
        pick_low = word in _LOW_SUPER
        answers = {s[0] if pick_low else s[-1] for s in solutions}
        if len(answers) == 1:
            return answers.pop()
        return None                       # ambiguous under constraints

    # full-order question ("in what order", "rank them")
    if re.search(r"\b(what order|rank|arrange|order (of|from))\b", text, re.I) \
            and len(solutions) == 1:
        lo_to_hi = list(solutions[0])
        if re.search(r"\b(tallest|oldest|fastest|biggest|largest|highest) (first|to)\b", text, re.I):
            lo_to_hi = lo_to_hi[::-1]
        return ", ".join(lo_to_hi)
    return None


# ---------------------------------------------------------------- assignment
_PEOPLE_LIST = re.compile(
    r"\b([A-Z][a-z]+(?:,\s*[A-Z][a-z]+){1,5},?\s*and\s+[A-Z][a-z]+)\b")
_ATTR_LIST = re.compile(
    r"different\s+\w+\s*[:\-]?\s*((?:an?\s+)?\w+(?:\s*,\s*(?:an?\s+)?\w+){1,5}"
    r"\s*,?\s*(?:(?:and|or)\s+)?(?:an?\s+)?\w+)")
_OWN_VERB = r"(?:owns?|has|have|likes?|plays?|drinks?|drives?|keeps?)"
_POS_FACT = re.compile(r"\b([A-Z][a-z]+)\s+" + _OWN_VERB + r"\s+(?:the\s+|an?\s+)?(\w+)")
_NEG_FACT = re.compile(
    r"\b([A-Z][a-z]+)\s+(?:does\s*n[o']t|doesn't|did not|didn't|never|cannot|can't)\s+"
    + _OWN_VERB + r"\s+(?:the\s+|an?\s+)?(\w+)")
_WHO_Q = re.compile(r"\bwho\s+" + _OWN_VERB + r"\s+(?:the\s+|an?\s+)?(\w+)", re.I)


def solve_assignment(spec: TaskSpec) -> Optional[str]:
    """Brute-force one-to-one assignment puzzles (pets, colors, drinks...)."""
    text = spec.prompt
    pm = _PEOPLE_LIST.search(text)
    am = _ATTR_LIST.search(text)
    qm = _WHO_Q.search(text)
    if not (pm and am and qm):
        return None
    people = [p.strip() for p in re.split(r",|\band\b", pm.group(1))
              if p.strip() and p.strip() not in _STOPNAMES]
    attrs = [re.sub(r"^an?\s+", "", a.strip().lower())
             for a in re.split(r",|\band\b|\bor\b", am.group(1)) if a.strip()]
    if len(people) != len(set(people)) or len(people) != len(attrs):
        return None
    target = qm.group(1).lower().rstrip("s")
    attr_set = {a.rstrip("s") for a in attrs}
    if target not in attr_set:
        return None

    neg = [(p, a.lower().rstrip("s")) for p, a in _NEG_FACT.findall(text)
           if p in people and a.lower().rstrip("s") in attr_set]
    neg_spans = {(p, a) for p, a in neg}
    pos = [(p, a.lower().rstrip("s")) for p, a in _POS_FACT.findall(text)
           if p in people and a.lower().rstrip("s") in attr_set
           and (p, a.lower().rstrip("s")) not in neg_spans]

    canon = [a.rstrip("s") for a in attrs]
    solutions = []
    for perm in itertools.permutations(canon):
        assign = dict(zip(people, perm))
        if all(assign[p] == a for p, a in pos) and \
           all(assign[p] != a for p, a in neg):
            solutions.append(assign)
    winners = {next(p for p, a in sol.items() if a == target) for sol in solutions}
    if len(winners) == 1:
        return winners.pop()
    return None
