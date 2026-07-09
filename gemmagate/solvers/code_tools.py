"""Code utilities — syntax validation plus a few surgical local bug fixes.

Local code debugging is limited to high-confidence, mechanical patterns
(mutable default arguments, assignment-in-condition). Everything else goes
remote: code is judged on correctness, and guessing is how you fail the
accuracy gate. Syntax checking via ast.parse is used by the validator for
every Python code answer regardless of route.
"""
from __future__ import annotations

import ast
import re
from typing import Optional

from ..schemas import TaskSpec


def python_syntax_ok(code: str) -> tuple[bool, str]:
    try:
        ast.parse(code)
        return True, ""
    except SyntaxError as e:
        return False, f"line {e.lineno}: {e.msg}"


def extract_code(text: str) -> str:
    """Pull code out of markdown fences if present, else return stripped text."""
    m = re.search(r"```(?:\w+)?\n(.*?)```", text, re.S)
    return (m.group(1) if m else text).strip()


def looks_like_python(code: str) -> bool:
    return bool(re.search(r"\b(def |import |print\(|return |elif |lambda )", code))


_ASSIGN_IN_IF = re.compile(r"^(\s*(?:if|while|elif)\b[^=\n]*[^=!<>])=([^=][^\n]*:)", re.M)


def try_local_fix(spec: TaskSpec) -> Optional[str]:
    """Only two mechanical, provable fixes. Anything else -> None (escalate)."""
    code = extract_code(spec.payload or spec.prompt)
    if not code or not looks_like_python(code):
        return None
    fixed = code
    changed = False

    # 1) assignment in condition: `if x = 5:` -> `if x == 5:`
    ok_before, _ = python_syntax_ok(fixed)
    if not ok_before and _ASSIGN_IN_IF.search(fixed):
        fixed = _ASSIGN_IN_IF.sub(r"\1==\2", fixed)
        changed = True

    # 2) mutable default argument: rewrite to the None-sentinel idiom.
    mdef = re.search(r"^(\s*)def\s+\w+\(([^)]*?)(\w+)\s*=\s*(\[\]|\{\})([^)]*)\):\s*$",
                     fixed, re.M)
    if mdef and fixed.count("def ") == 1:
        indent, arg, empty = mdef.group(1), mdef.group(3), mdef.group(4)
        header_new = fixed[mdef.start():mdef.end()].replace(
            f"{arg}={empty}", f"{arg}=None").replace(
            f"{arg} = {empty}", f"{arg}=None")
        body_guard = (f"{indent}    if {arg} is None:\n"
                      f"{indent}        {arg} = {empty}")
        fixed = (fixed[:mdef.start()] + header_new + "\n" + body_guard
                 + fixed[mdef.end():])
        changed = True

    if not changed:
        from . import code_repair
        return code_repair.attempt(spec)
    ok_after, _ = python_syntax_ok(fixed)
    if not ok_after:
        return None
    if spec.wants_justification:
        return ("Bug: assignment (=) was used instead of comparison (==) in a "
                "condition.\n```python\n" + fixed + "\n```")
    return fixed
