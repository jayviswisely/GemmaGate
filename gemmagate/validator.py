"""Validation layer — the free gate between every escalation rung.

Layers: sanity -> category format -> explicit constraints -> semantics.
Validators REPAIR trivially recoverable issues (strip fences, extract the
ANSWER: line, pull embedded JSON) rather than triggering a paid retry.
"""
from __future__ import annotations

import json
import re
from typing import Optional

from .schemas import Category, TaskSpec, Validation
from .solvers import math_solver, code_tools

_REFUSAL = re.compile(r"^(i('m| am) sorry|i cannot|i can't|as an ai|unfortunately\b)", re.I)
_PREAMBLE = re.compile(
    r"^\s*(sure[,!]?|here('s| is)( the| your)? \w+\s*[:\-]?|the answer is\s*[:\-]?)\s*", re.I)
_ANSWER_LINE = re.compile(r"ANSWER\s*[:=]\s*(.+?)\s*$", re.I | re.M)
_NUM_TOKEN = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?%?")


def _strip(text: str) -> str:
    text = (text or "").strip()
    m = re.match(r"^```(?:\w+)?\n(.*?)\n?```\s*$", text, re.S)
    if m:
        text = m.group(1).strip()
    return _PREAMBLE.sub("", text).strip()


def _find_json(text: str) -> Optional[str]:
    for op, cl in (("{", "}"), ("[", "]")):
        start = text.find(op)
        if start == -1:
            continue
        depth = 0
        for i in range(start, len(text)):
            if text[i] == op:
                depth += 1
            elif text[i] == cl:
                depth -= 1
                if depth == 0:
                    cand = text[start:i + 1]
                    try:
                        json.loads(cand)
                        return cand
                    except json.JSONDecodeError:
                        break
    return None


def validate(spec: TaskSpec, answer: str) -> Validation:
    a = _strip(answer)
    if not a:
        return Validation(False, 0.0, ["empty answer"])
    if _REFUSAL.match(a):
        return Validation(False, 0.0, ["refusal instead of answer"])
    if len(a) > 20000:
        return Validation(False, 0.1, ["answer absurdly long"])

    cat = spec.category
    reasons: list[str] = []
    score = 0.6

    # ------------------------------------------------------------- MATH
    if cat == Category.MATH:
        m = _ANSWER_LINE.search(answer)
        if m:
            a = m.group(1).strip().rstrip(".")
        a = re.sub(r"\b(approximately|about|around|roughly)\b\s*", "", a, flags=re.I)
        nums = _NUM_TOKEN.findall(a)
        if not nums:
            return Validation(False, 0.2, ["no numeric answer found"], a)
        if len(a.split()) > 12 or re.search(r"[a-zA-Z]{3,}", a):
            a = nums[-1]                 # keep only the value if verbose/unit-y
            reasons.append("value extracted")
        derived = math_solver.solve(spec.prompt)
        if derived is not None:
            try:
                ans_f = float(nums[-1].replace("$", "").replace(",", "").rstrip("%"))
                der_f = float(derived.rstrip("%"))
                if abs(ans_f - der_f) > 1e-6 * max(1.0, abs(der_f)):
                    return Validation(False, 0.05,
                                      [f"numeric re-check failed (expected {derived})"],
                                      derived)
                score = 1.0
            except ValueError:
                pass
        else:
            score += 0.2

    # -------------------------------------------------------- SENTIMENT
    elif cat == Category.SENTIMENT:
        labels = [l.lower() for l in
                  (spec.allowed_labels or ["positive", "negative", "neutral", "mixed"])]
        head = a.split("—")[0].split("-")[0].split(":")[0].strip().lower().rstrip(".!")
        if head in labels:
            if not spec.wants_justification:
                # normalize to the exact requested label form
                req = (spec.allowed_labels or
                       ["positive", "negative", "neutral", "mixed"])
                a = next((l for l in req if l.lower() == head), head)
                reasons.append("label normalized")
            score += 0.3
        else:
            hits = [l for l in labels if re.search(rf"\b{re.escape(l)}\b", a.lower())]
            if len(hits) == 1:
                prefix = hits[0]
                a = prefix if not spec.wants_justification else a
                reasons.append("label extracted")
                score += 0.15
            else:
                return Validation(False, 0.1, [f"label not in {labels}"], a)
        if spec.wants_justification and len(a.split()) < 3:
            return Validation(False, 0.35, ["justification requested but missing"], a)

    # -------------------------------------------------------------- NER
    elif cat == Category.NER:
        obj = None
        try:
            obj = json.loads(a)
        except json.JSONDecodeError:
            found = _find_json(a)
            if found:
                obj = json.loads(found)
                a = found
                reasons.append("JSON extracted")
        if spec.wants_json or obj is not None:
            if not isinstance(obj, dict) or not obj:
                return Validation(False, 0.1, ["NER output is not a JSON object"], a)
            _KEYMAP = {"people": "person", "persons": "person",
                       "organizations": "organization", "orgs": "organization",
                       "companies": "organization", "locations": "location",
                       "places": "location", "dates": "date"}
            _L2K = {"PERSON": "person", "PER": "person", "ORG": "organization",
                    "ORGANIZATION": "organization", "LOC": "location",
                    "LOCATION": "location", "GPE": "location", "DATE": "date"}
            _K2L = {"person": "PERSON", "organization": "ORG",
                    "location": "LOCATION", "date": "DATE"}
            if any(k.lower() in _KEYMAP for k in obj):
                obj = {_KEYMAP.get(k.lower(), k.lower()): v for k, v in obj.items()}
                a = json.dumps(obj, ensure_ascii=False)
                reasons.append("keys normalized")
            # schema conversion: whichever shape came back, emit the requested one
            is_list_shape = isinstance(obj.get("entities"), list)
            if spec.ner_list and not is_list_shape:
                ents = [{"text": e, "label": _K2L.get(k.lower(), str(k).upper())}
                        for k, vv in obj.items() if isinstance(vv, list)
                        for e in vv if isinstance(e, str)]
                obj = {"entities": ents}
                a = json.dumps(obj, ensure_ascii=False)
                reasons.append("converted to entities-list schema")
            elif not spec.ner_list and is_list_shape:
                keyed = {"person": [], "organization": [], "location": [], "date": []}
                for item in obj["entities"]:
                    if isinstance(item, dict) and "text" in item:
                        k = _L2K.get(str(item.get("label", "")).upper())
                        if k:
                            keyed[k].append(str(item["text"]))
                obj = keyed
                a = json.dumps(obj, ensure_ascii=False)
                reasons.append("converted to keyed schema")
            bad_vals = [k for k, v in obj.items()
                        if not isinstance(v, (list, str))]
            if bad_vals:
                return Validation(False, 0.2, [f"bad value types for {bad_vals}"], a)
            # hallucination guard: extracted strings must appear in source
            src = (spec.payload or spec.prompt).lower()
            if isinstance(obj.get("entities"), list):
                ents = [str(d.get("text", "")) for d in obj["entities"]
                        if isinstance(d, dict)]
            else:
                ents = [e for v in obj.values() if isinstance(v, list) for e in v
                        if isinstance(e, str)]
            missing = [e for e in ents if e.lower() not in src]
            if ents and len(missing) > len(ents) / 2:
                return Validation(False, 0.2,
                                  ["entities not present in source text"], a)
            score += 0.3
        else:
            if not re.search(r"(person|organization|location|date|entity)", a, re.I):
                return Validation(False, 0.3, ["no entity labels in output"], a)
            score += 0.2

    # ---------------------------------------------------- SUMMARIZATION
    elif cat == Category.SUMMARIZATION:
        if spec.max_bullets:
            bullets = [l for l in a.splitlines()
                       if re.match(r"\s*(?:[-•*]|\d+[.)])\s+\S", l)]
            if len(bullets) > spec.max_bullets:
                a = "\n".join(bullets[:spec.max_bullets])
                reasons.append("truncated to bullet limit")
                score -= 0.05
            elif len(bullets) < spec.max_bullets:
                return Validation(False, 0.3,
                                  [f"{len(bullets)} bullets, need {spec.max_bullets}"], a)
        # length overruns are SAFELY repairable by truncation at sentence
        # boundaries — a free repair always beats a paid retry
        if spec.max_sentences:
            sents = [s for s in re.split(r"(?<=[.!?])\s+", a) if s.strip()]
            if len(sents) > spec.max_sentences:
                a = " ".join(sents[:spec.max_sentences])
                reasons.append("truncated to sentence limit")
                score -= 0.05
        if spec.max_words and len(a.split()) > spec.max_words * 1.15:
            words = a.split()[:spec.max_words]
            a = " ".join(words).rstrip(",;:")
            if not a.endswith((".", "!", "?")):
                a += "."
            reasons.append("truncated to word limit")
            score -= 0.05
        src = spec.payload or ""
        if src and len(a) > 0.8 * len(src):
            return Validation(False, 0.3, ["summary not shorter than source"], a)
        score += 0.25

    # ------------------------------------------------------------- CODE
    elif cat in (Category.CODE_DEBUG, Category.CODE_GEN):
        keep_prose = (cat == Category.CODE_DEBUG and spec.wants_justification)
        code = code_tools.extract_code(a)
        if keep_prose and code == a.strip():
            # no fences: split leading prose from the code that starts at def/class
            m = re.search(r"(?m)^(?:def |class |import |from )", a)
            if m and m.start() > 0:
                code = a[m.start():].strip()
        if not code or not re.search(r"[=(){}:;]|\breturn\b|\bdef\b", code):
            return Validation(False, 0.2, ["no code in output"], a)
        if not keep_prose:
            a = code
        is_py = (spec.language == "python") or (
            spec.language is None and code_tools.looks_like_python(code))
        if is_py:
            ok, err = code_tools.python_syntax_ok(code)
            if not ok:
                return Validation(False, 0.15, [f"python syntax error: {err}"], a)
            score += 0.3
        else:
            if code.count("{") != code.count("}"):
                return Validation(False, 0.2, ["unbalanced braces"], a)
            score += 0.15
        # if the spec names a function, it must exist in the answer
        fn = re.search(r"function\s+(?:named\s+|called\s+)?`?([a-zA-Z_]\w*)\s*\(",
                       spec.instruction or spec.prompt)
        if fn and not re.search(r"\b(?:def|function)\s+" + re.escape(fn.group(1)) + r"\b",
                                code):
            return Validation(False, 0.25,
                              [f"required function {fn.group(1)} not defined"], a)
        # debug tasks: output must differ from the buggy input
        if cat == Category.CODE_DEBUG and spec.payload:
            if re.sub(r"\s+", "", code) == re.sub(r"\s+", "", spec.payload):
                return Validation(False, 0.2, ["code unchanged from buggy input"], a)

    # ------------------------------------------------------------ LOGIC
    elif cat == Category.LOGIC:
        if len(a.split()) > 80:
            return Validation(False, 0.35, ["over-long logic answer"], a)
        derived = None
        try:
            from .solvers import logic as logic_solver
            derived = logic_solver.solve(spec)
        except Exception:
            pass
        if derived is not None:
            if derived.lower() not in a.lower():
                return Validation(False, 0.1,
                                  [f"contradicts brute-force solution ({derived})"],
                                  derived)
            score = 0.95
        else:
            score += 0.15

    # ---------------------------------------------------------- FACTUAL
    else:
        if spec.wants_json:
            try:
                json.loads(a)
            except json.JSONDecodeError:
                found = _find_json(a)
                if not found:
                    return Validation(False, 0.2, ["invalid JSON"], a)
                a = found
                reasons.append("JSON extracted")
        if len(a.split()) < 2 and not spec.wants_json:
            return Validation(False, 0.4, ["suspiciously short factual answer"], a)
        score += 0.2

    # generic constraints
    if spec.max_words and cat != Category.SUMMARIZATION \
            and len(a.split()) > spec.max_words * 1.2:
        return Validation(False, 0.35, [f"exceeds {spec.max_words} words"], a)

    return Validation(True, min(1.0, score), reasons, a)
