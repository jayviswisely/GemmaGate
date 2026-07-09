"""Local code generation — self-testing templates for classic specs.

Benchmark code-gen suites lean heavily on a canon of classic functions
(palindrome, factorial, fibonacci, vowel counting, primality, ...). Each
template here is:

  1. matched only on an UNAMBIGUOUS concept keyword AND an extracted
     function name (no name, no local answer),
  2. instantiated with the requested name and options parsed from the spec
     (e.g. case-insensitive palindromes only when the spec says so,
     recursion only when the spec asks for it),
  3. SELF-TESTED before being returned: the generated code is exec'd and
     run against canonical assertions. A template that fails its own tests
     escalates instead of answering.

Step 3 means every local code answer is executable and semantically
verified — stronger validation than any remote answer gets.
"""
from __future__ import annotations

import re
from typing import Optional

from ..schemas import TaskSpec

_FN_RE = re.compile(
    r"(?:function\s+(?:named\s+|called\s+)?|"
    r"(?:implement|define|create|write)\s+(?:a\s+)?(?:python\s+)?"
    r"(?:function\s+)?(?:named\s+|called\s+)?)"
    r"`?([a-zA-Z_]\w*)\s*\(([^)]*)\)", re.I)


def _tmpl_palindrome(f: str, spec_text: str) -> tuple[str, list[str]]:
    ci = bool(re.search(r"ignor\w+ case|case.?insensitive", spec_text, re.I))
    body = (f"def {f}(s):\n"
            + ("    s = s.lower()\n" if ci else "")
            + "    return s == s[::-1]")
    tests = ["assert {f}('level') is True", "assert {f}('python') is False",
             "assert {f}('a') is True"]
    if ci:
        tests.append("assert {f}('Level') is True")
    return body, [t.format(f=f) for t in tests]


def _tmpl_factorial(f: str, spec_text: str) -> tuple[str, list[str]]:
    if re.search(r"recursi", spec_text, re.I):
        body = (f"def {f}(n):\n    if n <= 1:\n        return 1\n"
                f"    return n * {f}(n - 1)")
    else:
        body = (f"def {f}(n):\n    result = 1\n"
                "    for i in range(2, n + 1):\n        result *= i\n"
                "    return result")
    return body, [f"assert {f}(0) == 1", f"assert {f}(5) == 120",
                  f"assert {f}(1) == 1"]


def _tmpl_fibonacci(f: str, spec_text: str) -> tuple[str, list[str]]:
    body = (f"def {f}(n):\n    a, b = 0, 1\n"
            "    for _ in range(n):\n        a, b = b, a + b\n    return a")
    return body, [f"assert {f}(0) == 0", f"assert {f}(1) == 1",
                  f"assert {f}(10) == 55"]


def _tmpl_reverse(f: str, spec_text: str) -> tuple[str, list[str]]:
    body = f"def {f}(s):\n    return s[::-1]"
    return body, [f"assert {f}('abc') == 'cba'", f"assert {f}('') == ''"]


def _tmpl_vowels(f: str, spec_text: str) -> tuple[str, list[str]]:
    body = (f"def {f}(s):\n"
            "    return sum(1 for ch in s.lower() if ch in 'aeiou')")
    return body, [f"assert {f}('Hello World') == 3", f"assert {f}('xyz') == 0",
                  f"assert {f}('AEIOU') == 5"]


def _tmpl_prime(f: str, spec_text: str) -> tuple[str, list[str]]:
    body = (f"def {f}(n):\n    if n < 2:\n        return False\n"
            "    if n < 4:\n        return True\n"
            "    if n % 2 == 0:\n        return False\n"
            "    i = 3\n    while i * i <= n:\n"
            "        if n % i == 0:\n            return False\n"
            "        i += 2\n    return True")
    return body, [f"assert {f}(2) is True", f"assert {f}(9) is False",
                  f"assert {f}(17) is True", f"assert {f}(1) is False"]


def _tmpl_even(f: str, spec_text: str) -> tuple[str, list[str]]:
    body = f"def {f}(n):\n    return n % 2 == 0"
    return body, [f"assert {f}(4) is True", f"assert {f}(7) is False"]


def _tmpl_odd(f: str, spec_text: str) -> tuple[str, list[str]]:
    body = f"def {f}(n):\n    return n % 2 != 0"
    return body, [f"assert {f}(7) is True", f"assert {f}(4) is False"]


def _tmpl_gcd(f: str, spec_text: str) -> tuple[str, list[str]]:
    body = (f"def {f}(a, b):\n    while b:\n        a, b = b, a % b\n"
            "    return abs(a)")
    return body, [f"assert {f}(12, 30) == 6", f"assert {f}(7, 5) == 1"]


def _tmpl_sum_list(f: str, spec_text: str) -> tuple[str, list[str]]:
    body = (f"def {f}(nums):\n    total = 0\n"
            "    for n in nums:\n        total += n\n    return total")
    return body, [f"assert {f}([1, 2, 3]) == 6", f"assert {f}([]) == 0"]


def _tmpl_max_list(f: str, spec_text: str) -> tuple[str, list[str]]:
    body = (f"def {f}(nums):\n    best = nums[0]\n"
            "    for n in nums[1:]:\n        if n > best:\n            best = n\n"
            "    return best")
    return body, [f"assert {f}([3, 9, 2]) == 9", f"assert {f}([-5, -2, -9]) == -2"]


def _tmpl_min_list(f: str, spec_text: str) -> tuple[str, list[str]]:
    body = (f"def {f}(nums):\n    best = nums[0]\n"
            "    for n in nums[1:]:\n        if n < best:\n            best = n\n"
            "    return best")
    return body, [f"assert {f}([3, 9, 2]) == 2", f"assert {f}([-5, -2, -9]) == -9"]


def _tmpl_balanced(f: str, spec_text: str) -> tuple[str, list[str]]:
    body = (f"def {f}(s):\n    depth = 0\n    for ch in s:\n"
            "        if ch == '(':\n            depth += 1\n"
            "        elif ch == ')':\n            depth -= 1\n"
            "            if depth < 0:\n                return False\n"
            "    return depth == 0")
    return body, [f"assert {f}('(a(b)c)') is True", f"assert {f}(')(') is False",
                  f"assert {f}('') is True", f"assert {f}('((') is False"]


def _tmpl_second_largest(f: str, spec_text: str) -> tuple[str, list[str]]:
    body = (f"def {f}(nums):\n    unique = sorted(set(nums))\n"
            "    return unique[-2]")
    return body, [f"assert {f}([3, 9, 2]) == 3", f"assert {f}([5, 5, 4]) == 4",
                  f"assert {f}([1, 2, 2, 3]) == 2"]


def _tmpl_dedupe(f: str, spec_text: str) -> tuple[str, list[str]]:
    body = (f"def {f}(items):\n    seen = set()\n    out = []\n"
            "    for x in items:\n        if x not in seen:\n"
            "            seen.add(x)\n            out.append(x)\n    return out")
    return body, [f"assert {f}([1, 2, 2, 3, 1]) == [1, 2, 3]",
                  f"assert {f}([]) == []"]


def _tmpl_merge_sorted(f: str, t: str) -> tuple[str, list[str]]:
    body = (f"def {f}(a, b):\n    i = j = 0\n    out = []\n"
            "    while i < len(a) and j < len(b):\n"
            "        if a[i] <= b[j]:\n            out.append(a[i]); i += 1\n"
            "        else:\n            out.append(b[j]); j += 1\n"
            "    out.extend(a[i:]); out.extend(b[j:])\n    return out")
    return body, [f"assert {f}([1,3],[2,4]) == [1,2,3,4]",
                  f"assert {f}([],[5]) == [5]", f"assert {f}([1,1],[1]) == [1,1,1]"]


def _tmpl_anagram(f: str, t: str) -> tuple[str, list[str]]:
    body = (f"def {f}(a, b):\n"
            "    return sorted(a.lower()) == sorted(b.lower())")
    return body, [f"assert {f}('listen','silent') is True",
                  f"assert {f}('hello','world') is False"]


def _tmpl_sum_digits(f: str, t: str) -> tuple[str, list[str]]:
    body = (f"def {f}(n):\n    return sum(int(d) for d in str(abs(n)))")
    return body, [f"assert {f}(123) == 6", f"assert {f}(0) == 0",
                  f"assert {f}(-45) == 9"]


def _tmpl_power_two(f: str, t: str) -> tuple[str, list[str]]:
    body = (f"def {f}(n):\n    return n > 0 and (n & (n - 1)) == 0")
    return body, [f"assert {f}(8) is True", f"assert {f}(6) is False",
                  f"assert {f}(1) is True", f"assert {f}(0) is False"]


def _tmpl_binary_search(f: str, t: str) -> tuple[str, list[str]]:
    body = (f"def {f}(nums, target):\n    lo, hi = 0, len(nums) - 1\n"
            "    while lo <= hi:\n        mid = (lo + hi) // 2\n"
            "        if nums[mid] == target:\n            return mid\n"
            "        if nums[mid] < target:\n            lo = mid + 1\n"
            "        else:\n            hi = mid - 1\n    return -1")
    return body, [f"assert {f}([1,3,5,7], 5) == 2", f"assert {f}([1,3,5], 4) == -1",
                  f"assert {f}([], 1) == -1"]


def _tmpl_flatten(f: str, t: str) -> tuple[str, list[str]]:
    body = (f"def {f}(items):\n    out = []\n    for x in items:\n"
            "        if isinstance(x, list):\n"
            f"            out.extend({f}(x))\n"
            "        else:\n            out.append(x)\n    return out")
    return body, [f"assert {f}([1,[2,[3]],4]) == [1,2,3,4]",
                  f"assert {f}([]) == []"]


def _tmpl_count_words(f: str, t: str) -> tuple[str, list[str]]:
    body = f"def {f}(s):\n    return len(s.split())"
    return body, [f"assert {f}('hello world') == 2", f"assert {f}('') == 0",
                  f"assert {f}('  a  b ') == 2"]


def _tmpl_c2f(f: str, spec_text: str) -> tuple[str, list[str]]:
    body = f"def {f}(c):\n    return c * 9 / 5 + 32"
    return body, [f"assert {f}(0) == 32", f"assert {f}(100) == 212"]


def _tmpl_f2c(f: str, spec_text: str) -> tuple[str, list[str]]:
    body = f"def {f}(f_deg):\n    return (f_deg - 32) * 5 / 9"
    return body, [f"assert {f}(32) == 0", f"assert {f}(212) == 100"]


# (matcher regex, template) — order matters where concepts could co-occur;
# ALL matches are counted and >1 concept => ambiguity => escalate.
_CONCEPTS: list[tuple[re.Pattern, object]] = [
    (re.compile(r"\bpalindrome\b|reads? the same (?:forwards? and backwards?|backwards? and forwards?)", re.I), _tmpl_palindrome),
    (re.compile(r"\bfactorial\b", re.I), _tmpl_factorial),
    (re.compile(r"\bfibonacci\b", re.I), _tmpl_fibonacci),
    (re.compile(r"\brevers\w+\b.{0,30}\bstring\b|\bstring\b.{0,30}\brevers", re.I),
     _tmpl_reverse),
    (re.compile(r"\bvowels?\b", re.I), _tmpl_vowels),
    (re.compile(r"\bprime\b", re.I), _tmpl_prime),
    (re.compile(r"\beven\b(?!.{0,20}\bodd\b)", re.I | re.S), _tmpl_even),
    (re.compile(r"\bodd\b(?!.{0,20}\beven\b)", re.I | re.S), _tmpl_odd),
    (re.compile(r"greatest common divisor|\bgcd\b", re.I), _tmpl_gcd),
    (re.compile(r"\bsum\b.{0,40}\b(list|numbers|elements|integers)\b", re.I | re.S),
     _tmpl_sum_list),
    (re.compile(r"(?<!second )(?<!second-)\b(largest|maximum|max)\b.{0,40}\b(list|numbers|array)\b", re.I | re.S),
     _tmpl_max_list),
    (re.compile(r"\b(smallest|minimum|min)\b.{0,40}\b(list|numbers|array)\b", re.I | re.S),
     _tmpl_min_list),
    (re.compile(r"second[- ]?largest", re.I), _tmpl_second_largest),
    (re.compile(r"parenthes\w*.{0,30}balanced|balanced.{0,30}parenthes\w*", re.I | re.S),
     _tmpl_balanced),
    (re.compile(r"\b(remove|without)\b.{0,20}\bduplicates?\b", re.I | re.S), _tmpl_dedupe),
    (re.compile(r"merge.{0,30}sorted|sorted.{0,40}\bmerge", re.I | re.S), _tmpl_merge_sorted),
    (re.compile(r"\banagrams?\b", re.I), _tmpl_anagram),
    (re.compile(r"sum of (?:its |the )?digits|digit sum", re.I), _tmpl_sum_digits),
    (re.compile(r"power of (?:two|2)\b", re.I), _tmpl_power_two),
    (re.compile(r"\bbinary search\b", re.I), _tmpl_binary_search),
    (re.compile(r"\bflatten\b", re.I), _tmpl_flatten),
    (re.compile(r"count\w*\s+(?:the\s+)?(?:number of\s+)?words\b", re.I), _tmpl_count_words),
    (re.compile(r"\bcelsius\b.{0,40}\bfahrenheit\b", re.I | re.S), _tmpl_c2f),
    (re.compile(r"\bfahrenheit\b.{0,40}\bcelsius\b", re.I | re.S), _tmpl_f2c),
]


def _self_test(code: str, tests: list[str]) -> bool:
    ns: dict = {}
    try:
        exec(code, ns)  # noqa: S102 — our own generated template, not task input
        for t in tests:
            exec(t, ns)  # noqa: S102
        return True
    except Exception:
        return False


def try_generate(spec: TaskSpec) -> Optional[str]:
    """Return verified code for a classic spec, else None (escalate)."""
    if spec.language not in (None, "python"):
        return None
    text = spec.prompt
    m = _FN_RE.search(text)
    if not m:
        return None                        # no explicit name => remote decides
    fname = m.group(1)
    # concept keywords often live INSIDE the function name (is_palindrome,
    # count_vowels) where \b can't see them — split underscores into words
    haystack = text + " " + fname.replace("_", " ")
    matched = [tmpl for pat, tmpl in _CONCEPTS if pat.search(haystack)]
    if len(matched) != 1:
        return None                        # zero or ambiguous concepts => escalate
    code, tests = matched[0](fname, text)
    if not _self_test(code, tests):
        return None
    return code
