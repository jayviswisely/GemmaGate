"""Task Classifier + prompt-structure extraction for the eight categories.

Stage 1: high-precision regex heuristics (free, instant) — resolves ~95% of
benchmark-style prompts.
Stage 2: optional local model vote for still-unknown prompts (free tokens).
Remote models are NEVER used for classification.

Also extracts: instruction vs payload split, JSON requirement, justification
requirement, allowed label sets, word/sentence limits, code language.
"""
from __future__ import annotations

import re
from typing import Optional

from .schemas import Category, Risk, TaskSpec

_CODE_FENCE_RE = re.compile(r"```(\w+)?\n(.*?)```", re.S)
_CODE_HINT_RE = re.compile(
    r"\b(def |class |return |import |function\s*\(|=>|;\s*$|console\.log|print\s*\()", re.M)

_RULES: list[tuple[Category, re.Pattern]] = [
    (Category.CODE_DEBUG, re.compile(
        r"\b(fix|debug|find (the )?bug|what('s| is) wrong|correct (the|this) (code|function|implementation)|bug in)\b", re.I)),
    (Category.CODE_GEN, re.compile(
        r"\b(write|implement|create|generate)\b.{0,60}\b(function|method|class|script|program|code)\b", re.I | re.S)),
    (Category.NER, re.compile(
        r"\b(named entit(?:y|ies)|extract (?:and label )?(?:all )?(?:the )?entit(?:y|ies)|"
        r"identify (?:all )?(?:the )?(?:person|people|organization|location)s?\b.{0,40}\b(?:and|,)|"
        r"label.{0,30}entit(?:y|ies)|NER)\b", re.I | re.S)),
    (Category.SENTIMENT, re.compile(
        r"\b(sentiment|is (this|the) (review|text|comment|tweet) (positive|negative)|classify (the )?(tone|emotion|review))\b", re.I)),
    (Category.SUMMARIZATION, re.compile(
        r"\b(summar(y|ize|ise)|tl;?dr|condense|shorten (this|the) (passage|text|article)|key points)\b", re.I)),
    (Category.MATH, re.compile(
        r"(\d+(\.\d+)?\s*[\+\-\*/×÷^]\s*\d+)|\d+(\.\d+)?\s*%|\b(calculate|compute|how (much|many|long)|what (is|will).{0,40}\d|percent|interest|total cost|new price|divide|split|ratio|per (hour|year|month)|average|profit|discount)\b", re.I)),
    (Category.LOGIC, re.compile(
        r"\b(if .{0,80}then|all of the following|constraints?|who (is|sits|lives|owns|plays|drinks|drives|has|drew|wears)|logic puzzle|deduce|must be true|taller than|older than|to the (left|right) of|finished (before|after))\b", re.I | re.S)),
    (Category.FACTUAL, re.compile(
        r"\b(what (is|are|was|were)|who (is|was)|explain|describe|define|how (does|do|did).{0,60}work|why (is|does|do))\b", re.I)),
]

_LOCAL_CLS_PROMPT = (
    "Pick one label for the task: factual_knowledge, mathematical_reasoning, "
    "sentiment_classification, text_summarization, named_entity_recognition, "
    "code_debugging, logical_reasoning, code_generation.\nTask: {t}\nLabel:")

_PAYLOAD_MARKERS = ["```", "Text:", "TEXT:", "Passage:", "Review:", "Article:",
                    "Code:", "Input:", "Sentence:", "Document:", '"""', "---"]

_MAX_WORDS_RE = re.compile(
    r"\b(?:in|within|at most|no more than|maximum(?: of)?|under|using|exactly)\s+(?:exactly\s+)?(\d+)\s+words?\b", re.I)
_MAX_SENTS_RE = re.compile(
    r"\b(?:in|within|at most|no more than|maximum(?: of)?|exactly)\s+(?:exactly\s+)?(\d+|one|two|three|four|five)\s+sentences?\b", re.I)
_WORD_NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
_MAX_BULLETS_RE = re.compile(
    r"\b(?:in|as|using|with)?\s*(?:exactly\s+)?(\d+|one|two|three|four|five)\s+"
    r"bullet(?:\s+point)?s?\b", re.I)
_LANG_RE = re.compile(r"\b(python|javascript|typescript|java|c\+\+|c#|golang|go|rust|sql)\b", re.I)
_LABELSET_RE = re.compile(
    r"(?:one of|either|choose from|label(?:\s+\w+){0,3}\s+as|classify(?:\s+\w+){0,3}\s+as)\s*[:\-]?\s*((?:[\"'`]?\w[\w\s]{0,20}[\"'`]?\s*(?:,|/|\bor\b)\s*)+[\"'`]?\w[\w\s]{0,20}[\"'`]?)", re.I)


def _split(prompt: str) -> tuple[str, str]:
    fence = _CODE_FENCE_RE.search(prompt)
    if fence:
        idx = fence.start()
        instr = (prompt[:idx] + " " + prompt[fence.end():]).strip()
        return instr, fence.group(2)
    for marker in _PAYLOAD_MARKERS:
        idx = prompt.find(marker)
        if idx > 15:
            return prompt[:idx].strip(), prompt[idx + len(marker):].strip().strip('"')
    # long quoted block
    m = re.search(r"[\"“]([^\"”]{80,})[\"”]", prompt)
    if m:
        return (prompt[:m.start()] + prompt[m.end():]).strip(), m.group(1)
    return prompt.strip(), ""


# Rule weights: category-defining verbs beat incidental keyword overlap, so a
# prompt matching SENTIMENT strongly and FACTUAL weakly ("what is the
# sentiment...") lands correctly. Votes also yield a confidence signal.
_RULE_WEIGHTS = {
    Category.CODE_DEBUG: 3.0, Category.CODE_GEN: 2.5, Category.NER: 3.0,
    Category.SENTIMENT: 3.0, Category.SUMMARIZATION: 3.0, Category.MATH: 2.0,
    Category.LOGIC: 1.5, Category.FACTUAL: 1.0,
}


def _vote(prompt: str, head: str) -> tuple[Category, float]:
    scores: dict[Category, float] = {}
    for cat, pat in _RULES:
        # vote on the instruction head only: numbers/percents inside a payload
        # (e.g. "voted 7-2" in a passage to summarize) must not out-vote an
        # explicit instruction verb. Math/logic prompts rarely have payload
        # markers, so their head IS the full prompt anyway.
        hits = len(pat.findall(head))
        if hits:
            scores[cat] = _RULE_WEIGHTS[cat] * min(hits, 3)
    if not scores:
        return Category.UNKNOWN, 0.0
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top_cat, top = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else 0.0
    # confidence: margin over runner-up, saturating
    conf = min(1.0, 0.5 + (top - second) / (top + 1e-9) * 0.5 + (0.2 if top >= 3 else 0))
    return top_cat, conf


def classify(task_id: str, prompt: str, local_model=None) -> TaskSpec:
    spec = TaskSpec(task_id=task_id, prompt=prompt)
    spec.instruction, spec.payload = _split(prompt)
    head = spec.instruction or prompt

    # ---- category: weighted rule voting
    spec.category, spec.cls_confidence = _vote(prompt, head)

    # code-shaped payload + explicit bug/fix language forces CODE_DEBUG,
    # overriding ANY keyword vote ("count how many times ... has a bug"
    # must not be routed to math just because 'how many' scored)
    if spec.payload and _CODE_HINT_RE.search(spec.payload) and re.search(
            r"\b(bug|buggy|fix|broken|wrong|error|incorrect|corrected)\b", head, re.I):
        spec.category = Category.CODE_DEBUG
        spec.cls_confidence = max(spec.cls_confidence, 0.9)
    # code-shaped payload overrides weaker text categories
    elif spec.category in (Category.UNKNOWN, Category.FACTUAL) and spec.payload:
        if _CODE_HINT_RE.search(spec.payload):
            spec.category = (Category.CODE_DEBUG
                             if re.search(r"\b(bug|fix|wrong|error)\b", head, re.I)
                             else Category.CODE_GEN)

    # ---- local model fallback (free) for the residue
    if spec.category == Category.UNKNOWN and local_model is not None:
        try:
            out = local_model.generate(
                _LOCAL_CLS_PROMPT.format(t=prompt[:500]), max_tokens=8,
                temperature=0.0).text.strip().lower()
            label = re.sub(r"[^a-z_]", "", out.split()[0]) if out else ""
            spec.category = Category(label)
        except Exception:
            spec.category = Category.UNKNOWN
    if spec.category == Category.UNKNOWN:
        spec.category = Category.FACTUAL  # safest general-purpose default
        spec.cls_confidence = min(spec.cls_confidence, 0.3)

    # ---- structure extraction
    spec.wants_json = bool(re.search(r"\bjson\b", prompt, re.I))
    spec.wants_justification = bool(re.search(
        r"\b(justify|explain\s+(?:your|the)\s+\w+|give (a )?reason|why)\b",
        head, re.I))
    m = _MAX_WORDS_RE.search(prompt)
    if m:
        spec.max_words = int(m.group(1))
    m = _MAX_SENTS_RE.search(prompt)
    if m:
        v = m.group(1).lower()
        spec.max_sentences = _WORD_NUM.get(v) or int(v)
    m = _MAX_BULLETS_RE.search(prompt)
    if m:
        v = m.group(1).lower()
        spec.max_bullets = _WORD_NUM.get(v) or int(v)
        spec.max_sentences = None            # bullet limit supersedes
    # entities-list schema: {"entities": [{"text","label"}]}
    spec.ner_list = bool(re.search(r'"entities"|entities\s*:\s*\[|list of entit',
                                   prompt, re.I))
    m = _LANG_RE.search(prompt)
    if m:
        spec.language = m.group(1).lower()
    m = _LABELSET_RE.search(prompt)
    if m:
        labels = [w.strip(" \"'`") for w in re.split(r",|/|\bor\b", m.group(1))]
        spec.allowed_labels = [l.lower() for l in labels if 0 < len(l) <= 20][:8]
    return spec


# --------------------------------------------------------------- risk

_EASY_MATH_RE = re.compile(r"^\D{0,40}\d+(\.\d+)?\s*[\+\-\*/×÷^%]")

_BASE_RISK = {
    Category.MATH: Risk.LOW,          # tools try first; word problems bump up
    Category.SENTIMENT: Risk.LOW,
    Category.NER: Risk.MEDIUM,
    Category.SUMMARIZATION: Risk.MEDIUM,
    Category.FACTUAL: Risk.MEDIUM,
    Category.LOGIC: Risk.MEDIUM,      # brute-force tries first
    Category.CODE_DEBUG: Risk.HIGH,
    Category.CODE_GEN: Risk.HIGH,
    Category.UNKNOWN: Risk.MEDIUM,
}


def estimate_risk(spec: TaskSpec) -> TaskSpec:
    risk = _BASE_RISK.get(spec.category, Risk.MEDIUM)
    n = len(spec.prompt)
    if spec.category == Category.MATH and not _EASY_MATH_RE.search(spec.prompt) \
            and len(re.findall(r"\d+(?:\.\d+)?", spec.prompt)) >= 3 and n > 150:
        risk = Risk.MEDIUM                      # multi-step word problem
    if spec.category == Category.LOGIC and n > 600:
        risk = Risk.HIGH
    if spec.category == Category.SUMMARIZATION and n > 2500:
        risk = Risk.HIGH
    if spec.category == Category.FACTUAL and n < 120 and not spec.wants_json:
        risk = Risk.MEDIUM                      # still remote-cheap; never guess facts
    # classifier uncertainty compensator: a shaky category assignment means the
    # chosen ladder may be wrong, so make the ladder longer/stronger.
    if spec.cls_confidence < 0.45 and risk == Risk.LOW:
        risk = Risk.MEDIUM
    elif spec.cls_confidence < 0.3 and risk == Risk.MEDIUM:
        risk = Risk.HIGH
    spec.risk = risk
    return spec
