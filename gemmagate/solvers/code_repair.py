"""Oracle-verified mutation repair — automated program repair at 0 tokens.

Insight: debug prompts almost always STATE THE INTENT ("should return the
max of a list", "should sum every number strictly between a and b") or give
examples. Both are executable oracles:

  * examples in the prompt  -> assert-based oracle
  * recognized intent       -> a Python reference implementation (max, count,
                               cumulative mean, ...) run on generated inputs

Given an oracle, we enumerate small single-edit mutations of the buggy code
(off-by-one range bounds, comparator swaps, divide-by-index, zero-init to
first element, index +/-1, len()-1 tweaks) and accept the FIRST mutant that
agrees with the oracle on every test case. The returned fix is therefore
semantically verified, not guessed — the same guarantee our brute-force
logic solver gives, applied to code debugging.
"""
from __future__ import annotations

import ast
import math
import re
from typing import Callable, Optional

from ..schemas import TaskSpec

_MAX_MUTANTS = 90


# ------------------------------------------------------------- fn parsing

def _fn_info(code: str) -> Optional[tuple[str, list[str]]]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    fns = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    if len(fns) != 1:
        return None
    return fns[0].name, [a.arg for a in fns[0].args.args]


def _extract_examples(prompt: str, fname: str) -> list[str]:
    tests = []
    for m in re.finditer(
            re.escape(fname) + r"\(([^()]*)\)\s*"
            r"(?:should\s+)?(?:returns?|->|==|=>|gives?|is)\s*"
            r"(\[[^\]]*\]|[\w\.\'\"\-]+)", prompt):
        tests.append((m.group(1), m.group(2)))
    return tests[:5]


# ------------------------------------------------------------- oracles

_L = [[3, 9, 2], [-5, -2, -9], [7], [0, 0, 4], [10, 3, 10], [1, 2, 3, 4, 5]]
_LT = [([5, 2, 5, 5], 5), ([1], 1), ([2, 3], 9), ([4, 4, 4], 4), ([0, 1], 0)]
_P = [(1, 5), (2, 3), (0, 4), (3, 9), (2, 4)]
_S = ["hello", "A", "xyz", "Level", "aa bb"]

_ORACLES: list[tuple[re.Pattern, list, Callable]] = [
    (re.compile(r"\b(largest|max(?:imum)?)\b.{0,50}\b(list|numbers?|nums)\b", re.I | re.S),
     [(x,) for x in _L], lambda nums: max(nums)),
    (re.compile(r"\b(smallest|min(?:imum)?)\b.{0,50}\b(list|numbers?|nums)\b", re.I | re.S),
     [(x,) for x in _L], lambda nums: min(nums)),
    (re.compile(r"count.{0,50}\b(times|occurrenc\w*|appears?)\b", re.I | re.S),
     _LT, lambda nums, t: nums.count(t)),
    (re.compile(r"\brunning[- ]average|cumulative (average|mean)\b", re.I),
     [(x,) for x in _L],
     lambda nums: [sum(nums[:i + 1]) / (i + 1) for i in range(len(nums))]),
    (re.compile(r"\bsum\b.{0,80}\b(strictly between|between\b.{0,40}exclusive)", re.I | re.S),
     [(a, b) for a, b in _P], lambda a, b: sum(range(min(a, b) + 1, max(a, b)))),
    (re.compile(r"\bsum\b.{0,40}\b(list|numbers|elements)\b", re.I | re.S),
     [(x,) for x in _L], lambda nums: sum(nums)),
    (re.compile(r"\breverse\b.{0,25}\bstring\b", re.I),
     [(s,) for s in _S], lambda s: s[::-1]),
    (re.compile(r"\bvowels?\b", re.I),
     [(s,) for s in _S], lambda s: sum(c in "aeiou" for c in s.lower())),
    (re.compile(r"\bfactorial\b", re.I),
     [(n,) for n in (0, 1, 4, 6)], math.factorial),
]


def _pick_oracle(prompt: str, fname: str, code: str):
    ex = _extract_examples(prompt, fname)
    if ex:
        def check(fn) -> bool:
            ns2: dict = {}
            for args, expected in ex:
                try:
                    got = fn(*eval(f"({args},)", ns2))      # noqa: S307 fixed literals
                    exp = eval(expected, ns2)                # noqa: S307
                except Exception:
                    return False
                if not _eq(got, exp):
                    return False
            return True
        return check
    for pat, cases, ref in _ORACLES:
        if pat.search(prompt):
            def check(fn, cases=cases, ref=ref) -> bool:
                for args in cases:
                    try:
                        got = fn(*args)
                        exp = ref(*args)
                    except Exception:
                        return False
                    if not _eq(got, exp):
                        return False
                return True
            return check
    return None


def _eq(a, b) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        try:
            return math.isclose(float(a), float(b), rel_tol=1e-9)
        except (TypeError, ValueError):
            return False
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_eq(x, y) for x, y in zip(a, b))
    return a == b


# ------------------------------------------------------------- mutations

def _pm(expr: str, delta: int) -> str:
    expr = expr.strip()
    if delta == 1 and expr.endswith("- 1"):
        return expr[:-3].strip()
    if delta == -1 and expr.endswith("+ 1"):
        return expr[:-3].strip()
    return f"{expr} {'+' if delta > 0 else '-'} 1"


def _mutants(code: str, list_params: list[str]):
    lines = code.split("\n")
    seen = {code}

    def emit(i: int, newline: str):
        cand = "\n".join(lines[:i] + [newline] + lines[i + 1:])
        if cand not in seen:
            seen.add(cand)
            yield cand

    def _split_args(s: str) -> list[str]:
        out, depth, cur = [], 0, ""
        for ch in s:
            if ch == "," and depth == 0:
                out.append(cur.strip()); cur = ""
            else:
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                cur += ch
        if cur.strip():
            out.append(cur.strip())
        return out

    for i, l in enumerate(lines):
        # one nesting level allowed: range(1, len(nums))
        for m in re.finditer(r"range\(((?:[^()]|\([^()]*\))*)\)", l):
            args = _split_args(m.group(1))
            variants = set()
            if len(args) == 1:
                variants |= {f"1, {args[0]}", _pm(args[0], -1), _pm(args[0], 1)}
            elif len(args) == 2:
                a, b = args
                for na in {a, _pm(a, 1), _pm(a, -1), "0"}:
                    for nb in {b, _pm(b, 1), _pm(b, -1)}:
                        if (na, nb) != (a, b):
                            variants.add(f"{na}, {nb}")
            for v in variants:
                yield from emit(i, l[:m.start()] + f"range({v})" + l[m.end():])
        for a, b in (("<=", "<"), ("<", "<="), (">=", ">"), (">", ">=")):
            if a in l and "=" + a not in l:
                yield from emit(i, l.replace(a, b, 1))
        if re.search(r"/\s*i\b", l):
            yield from emit(i, re.sub(r"/\s*i\b", "/ (i + 1)", l, count=1))
        m = re.match(r"(\s*)(\w+)\s*=\s*0\s*$", l)
        if m:
            for p in list_params:
                yield from emit(i, f"{m.group(1)}{m.group(2)} = {p}[0]")
        for m in re.finditer(r"\[(\w+)\]", l):
            for d in (" - 1", " + 1"):
                yield from emit(i, l[:m.start()] + f"[{m.group(1)}{d}]" + l[m.end():])
        for m in re.finditer(r"len\((\w+)\)\s*-\s*1", l):
            yield from emit(i, l[:m.start()] + f"len({m.group(1)})" + l[m.end():])


# ------------------------------------------------------------- entrypoint

def attempt(spec: TaskSpec) -> Optional[str]:
    from .code_tools import extract_code, looks_like_python

    code = extract_code(spec.payload or spec.prompt)
    if not code or not looks_like_python(code):
        return None
    info = _fn_info(code)
    if info is None:
        return None
    fname, params = info
    check = _pick_oracle(spec.prompt, fname, code)
    if check is None:
        return None

    def load(src: str):
        ns: dict = {}
        try:
            exec(src, ns)                                   # noqa: S102 in-container
            return ns.get(fname)
        except Exception:
            return None

    # if the original already satisfies the oracle, it isn't broken the way
    # we can prove — don't "fix" it
    orig_fn = load(code)
    if orig_fn is not None and check(orig_fn):
        return None

    tried = 0
    for cand in _mutants(code, params):
        tried += 1
        if tried > _MAX_MUTANTS:
            break
        fn = load(cand)
        if fn is None:
            continue
        if check(fn):
            if spec.wants_justification:
                old_l = next(a for a, b in zip(code.split("\n"), cand.split("\n"))
                             if a != b)
                new_l = next(b for a, b in zip(code.split("\n"), cand.split("\n"))
                             if a != b)
                return (f"Bug: `{old_l.strip()}` should be `{new_l.strip()}`."
                        f"\n```python\n{cand}\n```")
            return cand
    return None
