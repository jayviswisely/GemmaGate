"""Prompt templates + compression — every remote token justified.

Per-category templates encode two things:
  1. the minimal instruction a capable model needs, and
  2. the output contract our validator will check.

Compression policy (accuracy-preserving):
  * filler / politeness / meta-instructions stripped from the INSTRUCTION
  * payloads preserved verbatim for code, math, logic, and NER (the spec data
    IS the task — compressing it loses accuracy)
  * summarization payloads are passed through (the model needs the text),
    but the instruction around them is canonicalized to a few tokens

Math is the one category where a hard "no reasoning" rule hurts the accuracy
gate: small models are far more accurate when allowed brief working. We allow
minimal working but demand a final `ANSWER: <value>` line the validator can
extract, and cap max_tokens to keep the working short.
"""
from __future__ import annotations

import re

from .schemas import Category, TaskSpec

_FILLER = [
    r"\b(please|kindly)\b",
    r"\b(i want you to|i need you to|i would like you to|can you|could you)\b",
    r"\byour (task|job|goal) is to\b",
    r"\b(make sure to|be sure to|remember to|note that)\b",
    r"\bas an ai( language model)?\b",
    r"\bthank you[.!]?",
]


def compress_instruction(text: str) -> str:
    for pat in _FILLER:
        text = re.sub(pat, "", text, flags=re.I)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\s+([,.;:?])", r"\1", text)
    # dedupe repeated sentences
    seen, out = set(), []
    for s in re.split(r"(?<=[.!?])\s+", text):
        k = re.sub(r"\W+", "", s.lower())
        if k and k not in seen:
            seen.add(k)
            out.append(s.strip())
    return " ".join(out).strip()


def _len_clause(spec: TaskSpec) -> str:
    if spec.max_bullets:
        return (f" in exactly {spec.max_bullets} bullet points, one line each, "
                "each starting with '- '")
    if spec.max_sentences:
        return f" in exactly {spec.max_sentences} sentence(s)"
    if spec.max_words:
        return f" in at most {spec.max_words} words"
    return ""


def build_prompt(spec: TaskSpec) -> tuple[str, int, bool]:
    """Returns (prompt, max_tokens, json_mode) for the remote call."""
    instr = compress_instruction(spec.instruction or spec.prompt)
    payload = spec.payload
    cat = spec.category

    if cat == Category.MATH:
        p = (f"{compress_instruction(spec.prompt)}\n"
             "Solve. Brief working allowed. Final line must be exactly: "
             "ANSWER: <value>")
        return p, 320, False

    if cat == Category.SENTIMENT:
        labels = "|".join(spec.allowed_labels or
                          ["positive", "negative", "neutral", "mixed"])
        if spec.wants_justification:
            p = (f"Sentiment of the text. Reply '<label> — <one short reason>'. "
                 f"Labels: {labels}.\nText: {payload or spec.prompt}")
            return p, 60, False
        p = f"Sentiment ({labels}). Reply with the label only.\nText: {payload or spec.prompt}"
        return p, 8, False

    if cat == Category.NER:
        if spec.ner_list:
            p = ('Extract entities present in the text. Return valid JSON only: '
                 '{"entities": [{"text": "...", "label": '
                 '"PERSON|ORG|LOCATION|DATE"}]}.\nText: '
                 + (payload or spec.prompt))
        else:
            p = ("Extract named entities as JSON with keys person, organization, "
                 "location, date (arrays of strings; empty array if none). "
                 "JSON only.\nText: " + (payload or spec.prompt))
        return p, 240, True

    if cat == Category.SUMMARIZATION:
        p = f"Summarize{_len_clause(spec)}. Output the summary only.\nText: {payload or spec.prompt}"
        cap = 60 + 12 * (spec.max_words or 0) // 8 if spec.max_words else 180
        return p, min(max(cap, 60), 300), False

    if cat == Category.CODE_DEBUG:
        lang = spec.language or "the same language"
        if spec.wants_justification:
            p = (f"Fix the bug(s). First line: 'Bug: <one short sentence>'. "
                 f"Then the corrected code in {lang} in a fenced block."
                 f"\n```\n{payload or spec.prompt}\n```")
        else:
            p = (f"Fix the bug(s). Output only the corrected code in {lang}, "
                 f"no explanation.\n```\n{payload or spec.prompt}\n```")
        return p, 600, False

    if cat == Category.CODE_GEN:
        p = f"{instr}\nOutput code only, no explanation."
        if payload:
            p += f"\nSpec/context:\n{payload}"
        return p, 600, False

    if cat == Category.LOGIC:
        p = (f"{compress_instruction(spec.prompt)}\n"
             "Reason internally. Reply with the final answer"
             + (" and one short justification sentence." if spec.wants_justification
                else " only."))
        return p, 100, False

    # FACTUAL / UNKNOWN
    p = f"{instr} Answer concisely (2-4 sentences)."
    if payload:
        p += f"\nContext: {payload}"
    if spec.wants_json:
        p += "\nOutput valid JSON only."
    return p, 180, spec.wants_json


def build_repair_prompt(spec: TaskSpec, bad_answer: str, reasons: list[str]) -> tuple[str, int, bool]:
    """Retry prompt with enough task context to fix semantic mistakes.

    A tiny repair prompt saves tokens, but if it omits the original task the
    model can only reformat the bad answer. Accuracy-first retries resend the
    task plus the validation failure.
    """
    reason = "; ".join(reasons[:3]) or "invalid output"
    bad = bad_answer if len(bad_answer) <= 500 else bad_answer[:500] + "…"
    task = spec.prompt if len(spec.prompt) <= 1800 else spec.prompt[:1800] + "..."
    fmt = {
        Category.NER: "Return corrected JSON only (keys: person, organization, location, date).",
        Category.MATH: "Return only the final line: ANSWER: <value>",
        Category.SENTIMENT: "Return the label only.",
        Category.CODE_DEBUG: "Return corrected code only.",
        Category.CODE_GEN: "Return corrected code only.",
        Category.SUMMARIZATION: f"Return the corrected summary only{_len_clause(spec)}.",
    }.get(spec.category, "Return the corrected answer only.")
    p = (f"TASK:\n{task}\n\n"
         f"Your previous answer failed validation: {reason}.\n"
         f"PREVIOUS:\n{bad}\n\n{fmt}")
    cap = {
        Category.NER: 400,
        Category.MATH: 384,
        Category.SENTIMENT: 60,
        Category.CODE_DEBUG: 900,
        Category.CODE_GEN: 900,
        Category.SUMMARIZATION: 500,
    }.get(spec.category, 500)
    return p, cap, spec.category == Category.NER


def attach_draft(prompt: str, draft: str) -> str:
    """Draft-conditional generation: ONE remote call that either costs 1-2
    output tokens (draft accepted) or returns the corrected answer (no second
    call, no resent payload). EV-positive whenever
    P(draft ok) * expected_output_tokens > draft_tokens."""
    return (f"{prompt}\n"
            f"PROPOSED: {draft}\n"
            "If PROPOSED is fully correct and properly formatted, reply "
            "exactly OK. Otherwise output the corrected answer only.")


def build_sentiment_batch(items: list, labels: list[str]) -> tuple[str, int]:
    """One remote call for many label-only sentiment residues.
    items: list of (number, text). Returns (prompt, max_tokens)."""
    label_str = "|".join(labels)
    lines = [f"Sentiment for each numbered text. Labels: {label_str}.",
             "For each numbered text output one line: <n>: <label>. "
             "No other text."]
    for n, text in items:
        one = " ".join(text.split())
        lines.append(f"{n}. {one}")
    return "\n".join(lines), 8 * len(items) + 8
