#!/usr/bin/env python3
"""GemmaGate generalization fuzzer.

The eval sets were authored alongside the solvers — they measure memorized
phrasings. This fuzzer measures GENERALIZATION: it programmatically builds
hundreds of tasks with independently computed ground truth, using randomized
values, names, and surface wordings deliberately different from anything in
the gold sets, then routes them all through the real Router (dry-run).

The metric hierarchy:
  1. LOCAL-WRONG  — a local route producing an incorrect answer. This is the
                    accuracy-gate killer. Target: ZERO. Every instance prints.
  2. local-correct coverage — how much of the free work we capture.
  3. escalations  — always acceptable (remote/failsafe answers are simulated
                    in dry-run, so they are excluded from accuracy scoring).

Usage:  python3 eval/fuzz.py [seed]
Exit code 1 if any LOCAL-WRONG exists.
"""
from __future__ import annotations

import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("GEMMAGATE_DRY_RUN", "1")
os.environ.setdefault("ALLOWED_MODELS", "sim-8b,sim-70b")
os.environ.setdefault("FIREWORKS_API_KEY", "sim")
os.environ.setdefault("FIREWORKS_BASE_URL", "https://sim.invalid/v1")

from run_benchmark import score  # noqa: E402  (same scoring as the gold sets)

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42
R = random.Random(SEED)

NAMES = ["Priya", "Omar", "Nadia", "Liam", "Kira", "Dev", "Tessa", "Yuri",
         "Ana", "Boris", "Chen", "Dana", "Sam", "Jo", "Lee", "Maya", "Felix",
         "Ines", "Ravi", "Zoe"]
FIRSTLAST = [("Maria", "Sanchez"), ("Wei", "Zhang"), ("Elena", "Petrova"),
             ("Kofi", "Mensah"), ("Lena", "Berg"), ("Diego", "Rossi")]
ORGS = ["Acme Corp", "Vertex Solutions", "Nakamura Industries",
        "University of Toronto", "Helios Bank", "Ministry of Finance",
        "Quantum Labs", "Borealis Group"]
LOCS = ["Tokyo", "Geneva", "Berlin", "Singapore", "Toronto", "Paris",
        "Seoul", "California"]
DATES = ["March 3, 2024", "2025-11-03", "14th of February 2025",
         "last Friday", "next Monday", "July 2019"]
NOUNS = ["ticket", "book", "widget", "chair", "lamp", "mug"]
ITEMS = ["jacket", "laptop", "bicycle", "camera", "sofa"]
CUR = ["$", "$", "$", "£", "€"]

CASES: list[dict] = []


def add(cat, prompt, check, allow_escalate=False):
    CASES.append({"task_id": f"{cat}-{len(CASES)}", "prompt": prompt,
                  "check": check, "cat": cat, "allow_escalate": allow_escalate})


def num(expected):
    return {"mode": "number", "expected": round(float(expected), 6)}


# ================================================================== MATH

def gen_math(k=110):
    for _ in range(k):
        fam = R.randrange(14)
        lead = R.choice(["What is", "Calculate", "Find", "Work out", "Determine"])
        c = R.choice(CUR)
        if fam == 0:
            p, n = R.choice([5, 10, 12, 15, 20, 25, 40]), R.randrange(40, 900)
            add("math", f"{lead} {p}% of {n}.", num(p / 100 * n))
        elif fam == 1:
            up = R.random() < 0.5
            verb = R.choice(["increased", "raised", "marked up"] if up else
                            ["decreased", "reduced", "marked down", "slashed", "cut"])
            n, p = R.randrange(60, 600), R.choice([10, 15, 20, 25, 30])
            add("math", f"A {R.choice(ITEMS)} priced at {c}{n} is {verb} by {p}%. "
                        f"What is the new price?",
                num(n * (1 + (p if up else -p) / 100)))
        elif fam == 2:
            n, a, b = R.randrange(100, 500), R.choice([10, 20, 25]), R.choice([10, 20, 30])
            add("math", f"A price of {c}{n} is increased by {a}% and then "
                        f"reduced by {b}%. {lead} the final price.",
                num(n * (1 + a / 100) * (1 - b / 100)))
        elif fam == 3:
            n, a, b = R.randrange(100, 400), R.choice([10, 20]), R.choice([5, 15, 25])
            add("math", f"A {R.choice(ITEMS)} costing {c}{n} is discounted by "
                        f"{a}%, then a further {b}% is taken off. "
                        f"What does it cost now?",
                num(n * (1 - a / 100) * (1 - b / 100)))
        elif fam == 4:
            a, b = R.choice([2, 3]), R.choice([3, 4, 5])
            total = (a + b) * R.randrange(20, 120)
            big = R.random() < 0.5
            add("math", f"{c}{total} is split between two partners in the "
                        f"ratio {a}:{b}. What is the "
                        f"{'larger' if big else 'smaller'} share?",
                num(total * (max(a, b) if big else min(a, b)) / (a + b)))
        elif fam == 5:
            q, p = R.randrange(3, 12), R.randrange(4, 30)
            add("math", f"{R.choice(NAMES)} buys {q} {R.choice(NOUNS)}s at "
                        f"{c}{p} each. How much does that cost in total?",
                num(q * p))
        elif fam == 6:
            b = R.randrange(40, 200); s = b + R.randrange(10, 90)
            add("math", f"A trader bought a {R.choice(ITEMS)} for {c}{b} and "
                        f"later sold it for {c}{s}. {lead} the profit.",
                num(s - b))
        elif fam == 7:
            a = R.randrange(40, 200); b = a + R.choice([a // 4, a // 2, a])
            add("math", f"A share price moved from {c}{a} to {c}{b}. "
                        f"What is the percentage increase?",
                num((b - a) / a * 100))
        elif fam == 8:
            n, p, y = R.choice([1000, 2000, 500]), R.choice([5, 8, 10]), R.choice([2, 3])
            unit = R.choice(["per year", "each year", "annually"])
            add("math", f"An investment of {c}{n} grows by {p}% {unit}. "
                        f"What is it worth after {y} years?",
                num(n * (1 + p / 100) ** y))
        elif fam == 9:
            p, r, y = R.choice([400, 500, 800]), R.choice([3, 4, 5]), R.choice([2, 3, 4])
            add("math", f"{lead} the simple interest on {c}{p} at {r}% for {y} years.",
                num(p * r / 100 * y))
        elif fam == 10:
            r, h = R.randrange(9, 30), R.randrange(3, 10)
            add("math", f"A courier earns {c}{r} per hour and works {h} hours. "
                        f"How much does the courier earn in total?", num(r * h))
        elif fam == 11:
            start = R.randrange(200, 600)
            p = R.choice([10, 20, 25, 50])
            m = R.randrange(20, 90); k2 = R.randrange(10, int(start * 0.3))
            add("math", f"A depot has {start} boxes. It ships {p}% on Monday, "
                        f"receives {m} boxes on Tuesday, then ships {k2} more "
                        f"on Wednesday. How many boxes remain?",
                num(start * (1 - p / 100) + m - k2))
        elif fam == 12:
            vals = [R.randrange(3, 60) for _ in range(R.randrange(4, 7))]
            op = R.choice(["average", "mean", "sum", "median"])
            lst = ", ".join(map(str, vals[:-1])) + f" and {vals[-1]}"
            import statistics
            exp = {"average": statistics.fmean(vals), "mean": statistics.fmean(vals),
                   "sum": sum(vals), "median": statistics.median(vals)}[op]
            add("math", f"{lead} the {op} of the numbers {lst}.", num(exp))
        else:
            a, b = R.choice([30, 40, 60]), R.choice([20, 60, 80])
            if a == b:
                b += 20
            add("math", f"A van drives to a depot at {a} km/h and returns along "
                        f"the same road at {b} km/h. What is its average speed "
                        f"for the whole journey?", num(2 * a * b / (a + b)))


# ================================================================= LOGIC

_ORDER_VERBS = [("is taller than", "tallest", "shortest"),
                ("is older than", "oldest", "youngest"),
                ("is faster than", "fastest", "slowest"),
                ("finished before", "first", "last")]


def gen_logic(k=55):
    for _ in range(k):
        if R.random() < 0.6:
            names = R.sample(NAMES, R.randrange(3, 6))
            order = names[:]                       # index 0 = least / earliest
            R.shuffle(order)
            verb, hi_word, lo_word = R.choice(_ORDER_VERBS)
            facts = []
            for i in range(len(order) - 1):
                a, b = order[i + 1], order[i]      # a "greater" than b
                if verb == "finished before":
                    facts.append(f"{b} finished before {a}.")
                else:
                    facts.append(f"{a} {verb} {b}.")
            R.shuffle(facts)
            hi = R.random() < 0.5
            q = R.choice([f"Who is the {hi_word if hi else lo_word}?",
                          f"Which of them is the {hi_word if hi else lo_word}?"]) \
                if verb != "finished before" else \
                f"Who finished {'last' if hi else 'first'}?"
            expected = order[-1] if hi else order[0]
            add("logic", " ".join(facts) + " " + q,
                {"mode": "exact", "expected": expected})
        else:
            k2 = R.randrange(3, 5)
            people = R.sample(NAMES, k2)
            pool = R.choice([["cat", "dog", "bird", "fish"],
                             ["coffee", "tea", "juice", "cocoa"],
                             ["guitar", "piano", "drums", "violin"]])[:k2]
            attrs = pool[:]
            R.shuffle(attrs)
            mapping = dict(zip(people, attrs))
            verb = {"cat": "own", "coffee": "drink", "guitar": "play"}[pool[0]]
            noun = {"cat": "pet", "coffee": "beverage", "guitar": "instrument"}[pool[0]]
            target_p = R.choice(people)
            facts = []
            others = [p for p in people if p != target_p]
            for p in others[:-1]:
                facts.append(f"{p} {verb}s the {mapping[p]}.")
            last = others[-1]
            facts.append(f"{last} does not {verb} the {mapping[target_p]}.")
            R.shuffle(facts)
            people_str = ", ".join(people[:-1]) + f", and {people[-1]}"
            attr_str = ", ".join(pool[:-1]) + f", and {pool[-1]}"
            add("logic", f"{k2} friends, {people_str}, each {verb} a different "
                         f"{noun}: {attr_str}. " + " ".join(facts) +
                         f" Who {verb}s the {mapping[target_p]}?",
                {"mode": "exact", "expected": target_p})


# ============================================================= SENTIMENT

_POS_W = ["excellent", "wonderful", "flawless", "fantastic", "delightful",
          "reliable", "amazing", "superb"]
_NEG_W = ["terrible", "awful", "broken", "useless", "disappointing",
          "faulty", "horrible", "defective"]


def gen_sentiment(k=55):
    frames = ["Classify the sentiment of this review as positive or negative:\n{r}",
              "Is the following feedback positive or negative?\n{r}",
              "Label the sentiment (positive/negative/neutral):\nReview: {r}",
              "What is the sentiment of this comment (positive, negative, or neutral)?\n{r}"]
    for _ in range(k):
        f = R.choice(frames)
        kind = R.randrange(5)
        if kind == 0:
            r = (f"Absolutely {R.choice(_POS_W)}, truly {R.choice(_POS_W)} "
                 f"and completely {R.choice(_POS_W)}.")
            exp = "positive"
        elif kind == 1:
            r = (f"Utterly {R.choice(_NEG_W)}, arrived {R.choice(_NEG_W)} "
                 f"and the service was {R.choice(_NEG_W)}.")
            exp = "negative"
        elif kind == 2:
            good, bad = R.choice(_POS_W), R.choice(_NEG_W)
            if R.random() < 0.5:
                r = f"The design is {bad}, but the performance is absolutely {good} and {R.choice(_POS_W)}."
                exp = "positive"
            else:
                r = f"The packaging looked {good}, but the device itself is {bad} and {R.choice(_NEG_W)}."
                exp = "negative"
        elif kind == 3:
            r = R.choice(["The parcel arrived on Thursday at the office.",
                          "The manual is printed in four languages.",
                          "The unit weighs about two kilograms."])
            if "neutral" not in f:
                continue
            exp = "neutral"
        else:
            r = (f"Oh great, another update that leaves it {R.choice(_NEG_W)}. "
                 f"Just what I needed.")
            exp = "negative"
        add("sentiment", f.format(r=r), {"mode": "exact", "expected": exp})
    # out-of-lexicon probes: escalation is the CORRECT behavior
    for r, exp in [("The ergonomics are sublime and the craftsmanship exquisite.", "positive"),
                   ("An abysmal, lamentable excuse for a product.", "negative")]:
        add("sentiment", frames[0].format(r=r),
            {"mode": "exact", "expected": exp}, allow_escalate=True)


# =================================================================== NER

def gen_ner(k=35):
    frames = [
        "Extract all named entities (person, organization, location, date) as JSON from: {s}",
        "Identify the person, organization, location and date entities. Output JSON.\nText: {s}",
        "List every named entity and its type from the text below as JSON.\nText: {s}",
    ]
    for _ in range(k):
        fn, ln = R.choice(FIRSTLAST)
        org, loc, date = R.choice(ORGS), R.choice(LOCS), R.choice(DATES)
        title = R.choice(["Dr.", "Prof.", "Ms.", "Mr."])
        s = R.choice([
            f"{title} {fn} {ln} of {org} met officials in {loc} on {date}.",
            f"{org} announced that {title} {fn} {ln} will visit {loc} on {date}.",
            f"On {date}, {title} {fn} {ln} joined {org} in {loc}.",
        ])
        add("ner", R.choice(frames).format(s=s),
            {"mode": "json_subset",
             "expected": {"person": [f"{fn} {ln}"], "organization": [org],
                          "location": [loc], "date": [date]}})


# ============================================================== CODE GEN

_GEN_SPECS = [
    ("pal", "returns True if the string s is a palindrome, ignoring case",
     ["assert {f}('Level') is True", "assert {f}('abc') is False"]),
    ("fact", "returns the factorial of n",
     ["assert {f}(5) == 120", "assert {f}(0) == 1"]),
    ("vcount", "returns the number of vowels in the string s",
     ["assert {f}('Hello') == 2"]),
    ("prime", "returns True if n is a prime number",
     ["assert {f}(7) is True", "assert {f}(9) is False"]),
    ("sec", "returns the second-largest distinct value in the list nums",
     ["assert {f}([4, 4, 1]) == 1"]),
    ("digsum", "returns the sum of its digits for the integer n",
     ["assert {f}(123) == 6"]),
    ("bsearch", "performs binary search on the sorted list nums and returns the index of target or -1",
     ["assert {f}([1,3,5], 5) == 2", "assert {f}([1,3,5], 4) == -1"]),
    ("flat", "flattens a nested list into a single flat list",
     ["assert {f}([1,[2,[3]]]) == [1,2,3]"]),
    ("anag", "returns True if the two strings a and b are anagrams",
     ["assert {f}('listen','silent') is True"]),
]
_ARGSIG = {"pal": "(s)", "fact": "(n)", "vcount": "(s)", "prime": "(n)",
           "sec": "(nums)", "digsum": "(n)", "bsearch": "(nums, target)",
           "flat": "(items)", "anag": "(a, b)"}
_HINT = {"pal": "palindrome", "fact": "factorial", "vcount": "vowels",
         "prime": "prime", "sec": "second-largest", "digsum": "sum of its digits",
         "bsearch": "binary search", "flat": "flatten", "anag": "anagrams"}


def gen_codegen(k=45):
    leads = ["Write a Python function", "Implement a function called",
             "Define", "Create a Python function named",
             "Please write a function"]
    for _ in range(k):
        key, spec, tests = R.choice(_GEN_SPECS)
        fname = R.choice([key, f"my_{key}", f"{key}_fn", f"do_{key}"])
        prompt = (f"{R.choice(leads)} {fname}{_ARGSIG[key]} that {spec}. "
                  f"It must handle the {_HINT[key]} case correctly.")
        add("gen", prompt,
            {"mode": "code_test", "tests": [t.format(f=fname) for t in tests]},
            allow_escalate=True)   # unmatched phrasings may escalate — fine


# ============================================================ CODE DEBUG

_DBG = [
    ("should return the largest value in the list",
     "def {f}(nums):\n    best = 0\n    for n in nums:\n        if n > best:\n            best = n\n    return best",
     ["assert {f}([-5, -2, -9]) == -2", "assert {f}([3, 9]) == 9"]),
    ("should count how many times target occurs in nums",
     "def {f}(nums, target):\n    c = 0\n    for i in range(1, len(nums)):\n        if nums[i] == target:\n            c += 1\n    return c",
     ["assert {f}([5, 2, 5], 5) == 2"]),
    ("should compute the running average of the values",
     "def {f}(values):\n    total = 0\n    out = []\n    for i, v in enumerate(values):\n        total += v\n        out.append(total / i)\n    return out",
     ["assert {f}([2, 4]) == [2.0, 3.0]"]),
    ("should sum every number strictly between a and b, exclusive on both ends",
     "def {f}(a, b):\n    t = 0\n    for n in range(a, b + 1):\n        t += n\n    return t",
     ["assert {f}(1, 5) == 9"]),
]


def gen_debug(k=30):
    frames = ["This function {spec}, but it has a bug. Fix it.\n```python\n{code}\n```",
              "There is a bug in the code below — it {spec}. Provide the corrected implementation.\n```python\n{code}\n```",
              "The following snippet is broken. It {spec}. Spot and correct the error.\n```python\n{code}\n```"]
    for _ in range(k):
        spec, code, tests = R.choice(_DBG)
        fname = R.choice(["fx", "calc", "proc", "run_it"])
        add("dbg", R.choice(frames).format(spec=spec, code=code.format(f=fname)),
            {"mode": "code_test", "tests": [t.format(f=fname) for t in tests]},
            allow_escalate=True)


# ============================================================ SUMMARIZE

_TOPIC = ["transit plan", "school budget", "harbour upgrade", "energy audit"]


def gen_summary(k=25):
    for _ in range(k):
        topic = R.choice(_TOPIC)
        sents = [f"The council reviewed the {topic} on {R.choice(['Monday','Tuesday'])}.",
                 f"Supporters said the {topic} would cut costs across the region.",
                 "Opponents raised concerns about the projected overruns.",
                 f"A final vote on the {topic} is expected next spring.",
                 "Officials promised quarterly progress reports."]
        R.shuffle(sents)
        passage = " ".join(sents[:R.randrange(4, 6)])
        style = R.randrange(3)
        if style == 0:
            n = R.choice([1, 2])
            c = R.choice([f"in exactly {n} sentence" + ("s" if n > 1 else ""),
                          f"in no more than {n} sentence" + ("s" if n > 1 else "")])
            check = {"mode": "summary", "max_sentences": n, "contains_any": [topic.split()[0]]}
        elif style == 1:
            w = R.choice([12, 15, 20])
            c = R.choice([f"in at most {w} words", f"within {w} words", f"using {w} words"])
            check = {"mode": "summary", "max_sentences": 3, "contains_any": [topic.split()[0]]}
        else:
            n = R.choice([2, 3])
            c = f"in {n} bullet points"
            check = {"mode": "contains_all", "expected": ["- "]}
        add("sum", f"Summarize the following {c}:\nText: {passage}", check)


# ================================================================== RUN

def main() -> int:
    gen_math(); gen_logic(); gen_sentiment(); gen_ner()
    gen_codegen(); gen_debug(); gen_summary()

    from gemmagate.router import Router
    import time
    router = Router()
    solved = router.solve_all(
        [{"task_id": c["task_id"], "prompt": c["prompt"]} for c in CASES],
        time.time() + 480)
    by_id = {s.task_id: s for s in solved}

    stats: dict[str, dict] = {}
    wrong_local: list[tuple] = []
    for case in CASES:
        s = by_id[case["task_id"]]
        st = stats.setdefault(case["cat"], {"n": 0, "local": 0, "ok": 0,
                                            "wrong_local": 0, "escalated": 0})
        st["n"] += 1
        is_local = s.remote_tokens == 0 and s.route.value.startswith("local")
        if not is_local:
            st["escalated"] += 1
            continue
        st["local"] += 1
        if score(case["check"], s.answer):
            st["ok"] += 1
        else:
            st["wrong_local"] += 1
            wrong_local.append((case["task_id"], case["prompt"], s.answer,
                                case["check"]))

    print(f"\nFUZZ seed={SEED}  tasks={len(CASES)}")
    print(f"{'category':<12}{'n':>5}{'local':>7}{'ok':>5}{'WRONG':>7}{'escal':>7}{'cov%':>7}")
    for cat, st in sorted(stats.items()):
        cov = 100 * st["ok"] / st["n"]
        print(f"{cat:<12}{st['n']:>5}{st['local']:>7}{st['ok']:>5}"
              f"{st['wrong_local']:>7}{st['escalated']:>7}{cov:>6.0f}%")
    total_wrong = len(wrong_local)
    print(f"\nLOCAL-WRONG total: {total_wrong}")
    for tid, prompt, ans, chk in wrong_local[:15]:
        print(f"\n--- {tid}\nPROMPT: {prompt[:180]}\nGOT: {ans[:120]}\nWANT: {chk}")
    return 1 if total_wrong else 0


if __name__ == "__main__":
    sys.exit(main())
