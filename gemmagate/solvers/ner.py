"""Rule-based NER — free, with a coverage guard.

Extracts person / organization / location / date entities using regexes,
suffix rules, honorifics, and a compact gazetteer. The critical safety
mechanism is the COVERAGE GUARD: if any capitalized span in the text was NOT
confidently labeled, the solver returns None and the task escalates. NER is
usually judged on exact entity sets, so a partial local answer is worse than
a remote call.
"""
from __future__ import annotations

import json
import re
from typing import Optional

from ..schemas import TaskSpec

_MONTHS = (r"(?:January|February|March|April|May|June|July|August|September|"
           r"October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)")
_DATE_RE = re.compile(
    r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}|"
    + _MONTHS + r"\.?\s+\d{1,2}(?:st|nd|rd|th)?,?(?:\s+\d{4})?|"
    r"(?:the\s+)?\d{1,2}(?:st|nd|rd|th)?\s+(?:of\s+)?" + _MONTHS + r"\.?,?(?:\s+\d{4})?|"
    r"(?:19|20)\d{2})\b")
_REL_DATE_RE = re.compile(
    r"\b(?:today|yesterday|tomorrow|"
    r"(?:last|next|this)\s+(?:week|month|year|Monday|Tuesday|Wednesday|"
    r"Thursday|Friday|Saturday|Sunday|January|February|March|April|May|June|"
    r"July|August|September|October|November|December))\b", re.I)

_ORG_OF = re.compile(
    r"\b((?:University|Bank|Ministry|Department|Institute|Museum) of "
    r"[A-Z][a-z]+(?: [A-Z][a-z]+)?)\b")
_ORG_SUFFIX = re.compile(
    r"\b((?:[A-Z][\w&\.']*\s+){0,4}[A-Z][\w&\.']*\s+"
    r"(?:Inc|Corp|Corporation|Ltd|LLC|Co|Company|Group|Bank|University|"
    r"Institute|Foundation|Agency|Committee|Association|Airlines|Motors|Labs|AI|Technologies|Solutions|Systems|Ventures|Capital|Partners|Industries|Holdings|Enterprises)\.?)\b")
_ORG_KNOWN = {
    "google", "microsoft", "apple", "amazon", "meta", "facebook", "tesla",
    "netflix", "ibm", "intel", "nvidia", "samsung", "sony", "toyota", "nasa",
    "fbi", "cia", "who", "un", "unesco", "unicef", "nato", "eu", "openai",
    "anthropic", "spacex", "boeing", "airbus", "pfizer", "moderna", "walmart",
    "starbucks", "mcdonald's", "nike", "adidas", "twitter", "x", "linkedin",
}
_LOC_KNOWN = {
    "paris", "london", "tokyo", "berlin", "madrid", "rome", "beijing",
    "shanghai", "moscow", "sydney", "toronto", "chicago", "seattle", "boston",
    "taipei", "taiwan", "japan", "china", "france", "germany", "spain",
    "italy", "russia", "canada", "australia", "india", "brazil", "mexico",
    "egypt", "kenya", "nigeria", "usa", "u.s.", "united states", "uk",
    "united kingdom", "new york", "los angeles", "san francisco", "singapore",
    "hong kong", "seoul", "south korea", "vietnam", "thailand", "europe",
    "asia", "africa", "america", "california", "texas", "florida", "washington",
}
_HONORIFIC_RE = re.compile(
    r"\b(?:Mr|Mrs|Ms|Dr|Prof|Professor|President|CEO|Senator|Judge|Captain)\.?\s+"
    r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})")
_CAP_SPAN_RE = re.compile(r"\b[A-Z][\w'\.]*(?:\s+[A-Z][\w'\.]*)*")
_PERSON_SHAPE = re.compile(r"^[A-Z][a-z]+(?:\s+[A-Z]\.)?\s+[A-Z][a-z]+$")
_MONTH_WORD = re.compile(_MONTHS + r"$")
_SENT_START = re.compile(r"(?:^|[.!?]\s+)([A-Z][\w']*)")

_STOPCAPS = {"The", "A", "An", "In", "On", "At", "It", "He", "She", "They",
             "We", "I", "This", "That", "These", "Those", "When", "After",
             "Before", "During", "According", "Meanwhile", "However", "But",
             "And", "Or", "If", "As", "By", "For", "From", "With", "Its",
             "His", "Her", "Their", "Our", "My", "Yesterday", "Today",
             "Tomorrow", "Monday", "Tuesday", "Wednesday", "Thursday",
             "Friday", "Saturday", "Sunday", "Dr", "Mr", "Mrs", "Ms", "Prof",
             "Professor", "President", "CEO", "Senator", "Judge", "Captain",
             "JSON", "CSV", "HTML", "API", "URL", "Extract", "Return",
             "Output", "Format", "Types", "Text"}


def extract(text: str) -> dict[str, list[str]]:
    ents: dict[str, list[str]] = {"person": [], "organization": [],
                                  "location": [], "date": []}
    claimed: list[str] = []

    def add(kind: str, value: str):
        value = re.sub(r"^the\s+", "", value.strip(" .,"), flags=re.I)
        if value and value not in ents[kind]:
            ents[kind].append(value)
            claimed.append(value)

    for m in _DATE_RE.finditer(text):
        add("date", m.group(0))
    for m in _REL_DATE_RE.finditer(text):
        add("date", m.group(0))
    for m in _ORG_OF.finditer(text):
        add("organization", m.group(1))
    for m in _ORG_SUFFIX.finditer(text):
        add("organization", m.group(1))
    for m in _HONORIFIC_RE.finditer(text):
        add("person", m.group(1))

    for m in _CAP_SPAN_RE.finditer(text):
        span = m.group(0).strip(" .,")
        if not span or any(span in c or c in span for c in claimed):
            continue
        low = span.lower()
        if low in _LOC_KNOWN:
            add("location", span)
        elif low in _ORG_KNOWN:
            add("organization", span)
        elif _PERSON_SHAPE.match(span) and span.split()[0] not in _STOPCAPS:
            add("person", span)
        elif len(span.split()) == 1 and span not in _STOPCAPS \
                and not _MONTH_WORD.match(span) \
                and re.search(r"\b(?:in|at|near|to|from)\s+" + re.escape(span) + r"\b", text):
            add("location", span)   # "in Springfield" — preposition signal
    return ents


def _unclaimed_spans(text: str, ents: dict) -> list[str]:
    claimed = {e for v in ents.values() for e in v}
    starts = set(_SENT_START.findall(text))
    leftovers = []
    for m in _CAP_SPAN_RE.finditer(text):
        span = m.group(0).strip(" .,")
        words = span.split()
        # ignore single sentence-start words and stopwords
        if len(words) == 1 and (span in _STOPCAPS or span in starts):
            continue
        if any(span in c or c in span for c in claimed):
            continue
        if all(w in _STOPCAPS for w in words):
            continue
        leftovers.append(span)
    return leftovers


_LIST_LABELS = {"person": "PERSON", "organization": "ORG",
                "location": "LOCATION", "date": "DATE"}


def to_entity_list(ents: dict, text: str) -> list[dict]:
    """Keyed dict -> [{'text','label'}] ordered by first appearance."""
    flat = [(text.find(e), {"text": e, "label": _LIST_LABELS[k]})
            for k, v in ents.items() for e in v]
    return [d for _, d in sorted(flat, key=lambda p: (p[0] if p[0] >= 0 else 10**9))]


def solve(spec: TaskSpec) -> Optional[str]:
    text = spec.payload or spec.prompt
    ents = extract(text)
    if not any(ents.values()):
        return None
    # COVERAGE GUARD: anything capitalized we couldn't label => escalate
    if _unclaimed_spans(text, ents):
        return None
    if spec.ner_list:
        return json.dumps({"entities": to_entity_list(ents, text)},
                          ensure_ascii=False)
    if spec.wants_json:
        return json.dumps(ents, ensure_ascii=False)
    lines = [f"{k.capitalize()}: {', '.join(v)}" for k, v in ents.items() if v]
    return "\n".join(lines)
