"""Rule/lexicon sentiment solver — free, only fires on unambiguous cases.

Scoring: lexicon hits with negation flipping ("not good" -> negative) and
intensity weighting ("absolutely terrible" > "bad"). The solver only returns
an answer when the signal is strong AND one-sided; mixed or weak signals
return None so the task escalates. This keeps local precision high, which is
what the accuracy gate demands.

Justification (when the prompt asks for it) is generated from the actual
evidence words found, so it is grounded in the text, not invented.
"""
from __future__ import annotations

import re
from typing import Optional

_BARE_RE = re.compile(
    r"\b(?:only|just)\b.{0,25}\b(?:word|label|name|answer)\b"
    r"|\bone[- ]word\b", re.I)

from ..schemas import TaskSpec

_POS = {
    "love": 2, "loved": 2, "excellent": 2, "amazing": 2, "fantastic": 2,
    "wonderful": 2, "perfect": 2, "outstanding": 2, "brilliant": 2,
    "superb": 2, "delightful": 2, "flawless": 2, "great": 1.5, "awesome": 2,
    "good": 1, "nice": 1, "happy": 1, "pleased": 1, "satisfied": 1,
    "enjoyable": 1, "impressive": 1.5, "recommend": 1.5, "recommended": 1.5,
    "best": 1.5, "fast": 0.5, "friendly": 1, "helpful": 1, "smooth": 1,
    "reliable": 1, "worth": 1, "beautiful": 1.5, "exceeded": 1.5, "liked": 1, "likes": 1, "enjoy": 1, "enjoyed": 1,
}
_NEG = {
    "hate": 2, "hated": 2, "terrible": 2, "awful": 2, "horrible": 2,
    "worst": 2, "disgusting": 2, "unacceptable": 2, "useless": 2,
    "garbage": 2, "trash": 2, "scam": 2, "broken": 1.5, "broke": 1.5, "breaks": 1.5,
    "bad": 1, "poor": 1, "disappointed": 1.5, "disappointing": 1.5,
    "waste": 1.5, "slow": 0.5, "rude": 1.5, "unhelpful": 1, "defective": 1.5,
    "refund": 1, "regret": 1.5, "annoying": 1, "cheap": 0.5,
    "faulty": 1.5, "failed": 1, "fails": 1, "scratches": 1.5, "scratched": 1.5, "cracks": 1.5, "cracked": 1.5, "bricks": 1.5, "bricked": 1.5, "flimsy": 1.5, "overheats": 1.5, "laggy": 1.5, "shame": 1.5, "frustrating": 1.5, "frustrated": 1.5, "annoying": 1.5, "clunky": 1.5, "sluggish": 1.5, "buggy": 1.5, "unreliable": 1.5, "painful": 1.0, "confusing": 1.0, "unusable": 2.0, "regret": 1.5, "waste": 1.5, "refund": 1.0, "dying": 1, "died": 1.5, "dies": 1, "lasted": 0.5, "unfortunately": 1, "sadly": 1, "crashed": 1, "overpriced": 1,
}
_NEGATORS = {"not", "no", "never", "isn't", "wasn't", "aren't", "don't",
             "doesn't", "didn't", "won't", "can't", "cannot", "hardly", "barely",
             "wouldn't", "couldn't", "shouldn't", "nothing", "nobody"}
_INTENSIFIERS = {"very": 1.5, "extremely": 2, "absolutely": 2, "really": 1.3,
                 "so": 1.2, "incredibly": 2, "totally": 1.5}

_DEFAULT_LABELS = ["positive", "negative", "neutral", "mixed"]


def _score(text: str) -> tuple[float, float, list[str], list[str]]:
    words = re.findall(r"[a-z']+", text.lower())
    pos_score = neg_score = 0.0
    pos_ev: list[str] = []
    neg_ev: list[str] = []
    for i, w in enumerate(words):
        base, polarity = (_POS.get(w), "pos") if w in _POS else (_NEG.get(w), "neg")
        if base is None:
            continue
        window = words[max(0, i - 4):i]
        mult = 1.0
        for prev in window:
            mult *= _INTENSIFIERS.get(prev, 1.0)
        # odd number of negators flips polarity; an even count ("wouldn't say
        # it's not worth it") cancels out — double negation reads positive
        negated = sum(p in _NEGATORS for p in window) % 2 == 1
        if negated:
            polarity = "neg" if polarity == "pos" else "pos"
        if polarity == "pos":
            pos_score += base * mult
            pos_ev.append(("not " if negated else "") + w)
        else:
            neg_score += base * mult
            neg_ev.append(("not " if negated else "") + w)
    return pos_score, neg_score, pos_ev, neg_ev


_CONTRAST_RE = re.compile(r"\b(but|however|yet|that said|on the other hand)\b[,\s]", re.I)
_SARCASM_RE = re.compile(
    r"\b(yeah,? right|oh,? (great|wonderful|fantastic)|just great|just what i needed|"
    r"thanks a lot|sure,? because|as if|great\.{3})\b", re.I)

# synonym bridges for custom label sets ("good"/"bad", "favorable"/...)
_LABEL_SYNONYMS = {
    "positive": {"positive", "pos", "good", "favorable", "favourable", "happy"},
    "negative": {"negative", "neg", "bad", "unfavorable", "unfavourable", "unhappy"},
}


def _map_neutral(labels: list[str]) -> Optional[str]:
    for l in labels:
        if l.lower() in ("neutral", "objective"):
            return l
    return None


def _map_label(polarity: str, labels: list[str]) -> Optional[str]:
    """Map internal polarity onto the prompt's label set, via synonyms."""
    wanted = _LABEL_SYNONYMS[polarity]
    for l in labels:
        if l.lower() in wanted:
            return l
    return None


def solve(spec: TaskSpec) -> Optional[str]:
    text = spec.payload or spec.prompt
    labels = spec.allowed_labels or _DEFAULT_LABELS

    pos, neg, pos_ev, neg_ev = _score(text)

    # sarcasm-like phrasing: the surface positives are fake. With concrete
    # negative evidence alongside ("bricks the camera"), the verdict is
    # decidable — negative. Sarcasm with NO negative evidence stays too
    # ambiguous for a lexicon: escalate.
    if _SARCASM_RE.search(text):
        if neg >= 1 and _map_label("negative", labels):
            label = _map_label("negative", labels)
            if not spec.wants_justification:
                return label
            ev = ", ".join(f'"{e}"' for e in neg_ev[:3]) or '"sarcastic phrasing"'
            return (f"{label} — sarcastic framing combined with negative "
                    f"language such as {ev}.")
        return None

    # contrast handling: "X was terrible, BUT Y is amazing" — the clause after
    # the last contrast marker carries the verdict. Judge on that clause; the
    # pre-contrast side only needs to not be overwhelming.
    contrasts = list(_CONTRAST_RE.finditer(text))
    if contrasts:
        tail = text[contrasts[-1].end():]
        t_pos, t_neg, t_pos_ev, t_neg_ev = _score(tail)
        if t_pos >= 1.5 and t_neg <= 0.5 and neg <= t_pos * 2:
            pos, neg, pos_ev, neg_ev = t_pos + 0.5, t_neg, t_pos_ev, t_neg_ev
        elif t_neg >= 1.5 and t_pos <= 0.5 and pos <= t_neg * 2:
            pos, neg, pos_ev, neg_ev = t_pos, t_neg + 0.5, t_pos_ev, t_neg_ev
        else:
            return None      # contrast without a clear post-clause winner

    # zero-signal => neutral ONLY with positive evidence of a factual
    # statement. Silence is NOT neutrality: "the craftsmanship is exquisite"
    # scores 0 in our lexicon yet is clearly evaluative — that must escalate,
    # never be confidently labeled neutral.
    if pos == 0 and neg == 0:
        neutral = _map_neutral(labels)
        words = text.split()
        factual_cue = re.search(
            r"\b(arriv\w+|deliver\w+|schedul\w+|ship\w+|weigh\w+|"
            r"measur\w+|contain\w+|includ\w+|printed|located|version|"
            r"model|manual|package|parcel|kilograms?|grams?|pages?|"
            r"languages?|centimeters?|inches|Monday|Tuesday|Wednesday|"
            r"Thursday|Friday|Saturday|Sunday|\d)", text, re.I)
        evaluative_cue = re.search(
            r"\b(absolutely|utterly|truly|completely|so|such an?|excuse for|"
            r"masterpiece|disaster)\b|!", text, re.I)
        if neutral and factual_cue and not evaluative_cue \
                and 4 <= len(words) <= 60:
            if _BARE_RE.search(spec.prompt):
                return neutral
            return (f"{neutral} — the text is a factual statement with no "
                    f"evaluative language.")
        return None

    # confidence gate: strong AND one-sided, else escalate
    polarity: Optional[str] = None
    evidence: list[str] = []
    if pos >= 2 and neg <= 0.5:
        polarity, evidence = "positive", pos_ev
    elif neg >= 2 and pos <= 0.5:
        polarity, evidence = "negative", neg_ev
    if polarity is None:
        return None
    label = _map_label(polarity, labels)
    if label is None:
        return None

    # The category spec says "labelling sentiment AND justifying the
    # classification", and an LLM judge reads the whole answer — a local
    # justification costs zero tokens, so include one by default. Bare-format
    # requests ("only the label") stay bare.
    if _BARE_RE.search(spec.prompt):
        return label
    if not spec.wants_justification and False:
        return label
    ev = ", ".join(f'"{e}"' for e in evidence[:3])
    return (f"{label} — the text uses clearly "
            f"{'positive' if polarity == 'positive' else 'negative'} "
            f"language such as {ev}.")
