"""GemmaGate test suite — runs fully offline (dry-run remote, no network).

Covers the two things that decide the competition:
  A. HARNESS CONTRACT — results.json always valid, every task_id present,
     exit-safe under malformed input, empty input, and crashing tasks.
  B. ROUTING INVARIANTS — deterministic categories cost 0 remote tokens,
     validation failures escalate, failsafes never return empty.

Run:  python3 tests/test_all.py     (or python -m pytest tests/ -v)
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("GEMMAGATE_DRY_RUN", "1")
os.environ.setdefault("ALLOWED_MODELS", "test-8b-instruct,test-34b,test-70b-instruct")
os.environ.setdefault("FIREWORKS_API_KEY", "test")
os.environ.setdefault("FIREWORKS_BASE_URL", "https://example.invalid/v1")

import time

from gemmagate.classifier import classify, estimate_risk
from gemmagate.model_select import plan_tiers
from gemmagate.router import Router
from gemmagate.schemas import Category, Route
from gemmagate.solvers import logic, math_solver, ner, sentiment, summarize
from gemmagate.validator import validate

DEADLINE = time.time() + 300


def _spec(prompt, tid="t"):
    s = classify(tid, prompt)
    return estimate_risk(s)


# ------------------------------------------------------- A. harness contract

def test_output_always_covers_all_ids():
    import subprocess
    with tempfile.TemporaryDirectory() as d:
        inp, out = os.path.join(d, "tasks.json"), os.path.join(d, "results.json")
        json.dump([{"task_id": "a", "prompt": "What is 2+2?"},
                   {"task_id": "b"},                       # missing prompt
                   {"task_id": "c", "prompt": None}],       # null prompt
                  open(inp, "w"))
        env = dict(os.environ, INPUT_PATH=inp, OUTPUT_PATH=out)
        p = subprocess.run([sys.executable, "main.py"], env=env,
                           cwd=os.path.join(os.path.dirname(__file__), ".."),
                           capture_output=True, timeout=120)
        assert p.returncode == 0, p.stderr.decode()[-500:]
        data = json.load(open(out))
        assert {r["task_id"] for r in data} == {"a", "b", "c"}
        assert all(isinstance(r["answer"], str) for r in data)


def test_malformed_input_still_exits_zero():
    import subprocess
    with tempfile.TemporaryDirectory() as d:
        inp, out = os.path.join(d, "tasks.json"), os.path.join(d, "results.json")
        open(inp, "w").write('[{"task_id": "x", "prompt": "hi"')  # broken JSON
        env = dict(os.environ, INPUT_PATH=inp, OUTPUT_PATH=out)
        p = subprocess.run([sys.executable, "main.py"], env=env,
                           cwd=os.path.join(os.path.dirname(__file__), ".."),
                           capture_output=True, timeout=120)
        assert p.returncode == 0
        data = json.load(open(out))          # salvaged the task_id
        assert data and data[0]["task_id"] == "x"


def test_tier_planning_from_names():
    plan = plan_tiers(["big-70b-instruct", "tiny-3b", "mid-34b-chat"])
    assert plan.cheap == "tiny-3b"
    assert plan.strong == "big-70b-instruct"
    assert plan.mid == "mid-34b-chat"
    single = plan_tiers(["only-model-x"])
    assert single.cheap == single.mid == single.strong == "only-model-x"


# --------------------------------------------------- B. routing invariants

def test_math_zero_tokens():
    r = Router()
    out = r.solve_all([{"task_id": "m1", "prompt": "Calculate 19 * 21."},
                       {"task_id": "m2", "prompt": "What is 15% of 80?"},
                       {"task_id": "m3", "prompt": "A store increases the price of a $240 jacket by 15%. What is the new price?"}],
                      DEADLINE)
    answers = {s.task_id: s for s in out}
    assert "399" in answers["m1"].answer and answers["m1"].remote_tokens == 0
    assert "12" in answers["m2"].answer and answers["m2"].remote_tokens == 0
    assert "276" in answers["m3"].answer and answers["m3"].remote_tokens == 0


def test_sentiment_gating():
    clear = _spec("Sentiment of this review: positive or negative?\n"
                  "Review: Absolutely terrible, broken on arrival, total waste.")
    assert sentiment.solve(clear).startswith("negative")
    # contrast: post-"but" clause carries the verdict -> answered locally
    contrast = _spec("Sentiment?\nReview: The screen is amazing but the battery is terrible.")
    assert sentiment.solve(contrast).startswith("negative")
    # genuinely balanced (no contrast dominance) => escalate
    balanced = _spec("Sentiment?\nReview: The screen is amazing and the battery is terrible.")
    assert sentiment.solve(balanced) is None
    # sarcasm + concrete damage evidence => decidable negative
    sarcasm = _spec("Sentiment?\nReview: Oh great, another update that breaks "
                    "everything. Just what I needed.")
    assert sentiment.solve(sarcasm).startswith("negative")
    # sarcasm with NO negative lexicon evidence => still escalate
    pure = _spec("Sentiment?\nReview: Oh great, just what I needed.")
    assert sentiment.solve(pure) is None


def test_ner_coverage_guard():
    covered = _spec("Extract entities as JSON.\nText: Dr. Maria Chen joined "
                    "Acme Corp in Tokyo on March 3, 2024.")
    out = ner.solve(covered)
    assert out and json.loads(out)["person"] == ["Maria Chen"]
    uncovered = _spec("Extract entities as JSON.\nText: The Zorblax Initiative "
                      "met with Kremvia Holdings yesterday.")
    assert ner.solve(uncovered) is None         # unknown spans => escalate


def test_logic_brute_force_and_contradiction_check():
    s = _spec("Alice is taller than Bob. Carol is shorter than Bob. Who is the tallest?")
    assert logic.solve(s) == "Alice"
    v = validate(s, "Carol")                    # contradicts provable solution
    assert not v.passed and v.repaired == "Alice"


def test_code_validator_and_local_fix():
    s = _spec("Fix the bug.\n```python\ndef f(n):\n    if n % 2 = 0:\n"
              "        return True\n    return False\n```")
    assert s.category == Category.CODE_DEBUG
    from gemmagate.solvers import code_tools
    fixed = code_tools.try_local_fix(s)
    assert fixed and "==" in fixed
    assert validate(s, fixed).passed
    assert not validate(s, s.payload).passed    # unchanged code rejected


def test_summary_length_enforced():
    s = _spec("Summarize in 2 sentences.\nText: " +
              " ".join(f"Sentence number {i} talks about the plan." for i in range(8)))
    long_ans = " ".join("This is sentence %d." % i for i in range(5))
    v = validate(s, long_ans)                   # over-long => FREE truncation repair
    assert v.passed and v.repaired.count(".") == 2


def test_math_new_patterns_zero_tokens():
    cases = {
        "Simplify the ratio of 12 to 30.": "2:5",
        "Sarah buys 7 tickets at $15 each. What is the total cost?": "105",
        "A trader bought a watch for $80 and sold it for $110. What was the profit?": "30",
        "A stock went from $50 to $65. What is the percentage increase?": "30%",
        "An investment of $1000 grows by 10% per year. What is it worth after 3 years?": "1331",
        "Calculate the simple interest on $500 at 4% for 3 years.": "60",
    }
    for prompt, expected in cases.items():
        assert math_solver.solve(prompt) == expected, prompt


def test_ner_recall_and_article_trim():
    s = _spec("Extract entities as JSON.\nText: Dr. Alan Wu of the University "
              "of Melbourne spoke in Geneva on the 3rd of March 2024.")
    out = json.loads(ner.solve(s))
    assert out["person"] == ["Alan Wu"]
    assert out["organization"] == ["University of Melbourne"]
    assert out["location"] == ["Geneva"]
    assert out["date"] == ["3rd of March 2024"]


def test_custom_sentiment_labels():
    s = _spec("Label this review as good or bad.\nReview: Absolutely terrible, "
              "broken on arrival, total waste of money.")
    assert sentiment.solve(s).split(" \u2014")[0].split(":")[0].strip() == "bad"


def test_validator_label_normalization():
    s = _spec('Classify the sentiment. Answer with one of "positive" or "negative".\n'
              "Text: whatever")
    v = validate(s, "Positive.")
    assert v.passed and v.repaired == "positive"


def test_validator_math_units_stripped():
    s = _spec("A store increases the price of a $240 jacket by 15%. "
              "What is the new price?")
    v = validate(s, "ANSWER: approximately $276.00 dollars")
    assert v.passed and float(v.repaired.replace("$", "")) == 276.0


def test_validator_ner_key_synonyms():
    s = _spec("Extract entities. Output JSON.\nText: Maria Chen visited Tokyo.")
    v = validate(s, '{"people": ["Maria Chen"], "locations": ["Tokyo"]}')
    assert v.passed
    assert "person" in json.loads(v.repaired)


def test_validator_required_function_name():
    s = _spec("Write a Python function is_palindrome(s) that returns True if s "
              "reads the same forwards and backwards.")
    good = "def is_palindrome(s):\n    s = s.lower()\n    return s == s[::-1]"
    assert validate(s, good).passed
    wrong = "def check(s):\n    return s == s[::-1]"
    assert not validate(s, wrong).passed        # named function missing => escalate


def test_low_confidence_raises_risk():
    from gemmagate.schemas import Risk
    s = _spec("zorp the fleem?")                # nothing matches any rule
    assert s.cls_confidence <= 0.3
    assert s.risk in (Risk.MEDIUM, Risk.HIGH)


def test_duplicate_task_ids_both_answered():
    r = Router()
    out = r.solve_all([{"task_id": "dup", "prompt": "Calculate 2 + 3."},
                       {"task_id": "dup", "prompt": "Calculate 10 * 4."}],
                      DEADLINE)
    assert len(out) == 2
    joined = " | ".join(o.answer for o in out)
    assert "5" in joined and "40" in joined


def test_identical_prompts_single_remote_call():
    r = Router()
    r.client.set_dry_responses({"photosynthesis": "Plants convert sunlight into "
                                "glucose and oxygen via chlorophyll."})
    prompt = "Explain how photosynthesis works."
    out = r.solve_all([{"task_id": "a", "prompt": prompt},
                       {"task_id": "b", "prompt": prompt},
                       {"task_id": "c", "prompt": prompt}], DEADLINE)
    assert all(o.answer.startswith("Plants convert") for o in out)
    assert r.ledger.calls == 1                  # in-run dedup: pay once


def test_escalation_on_bad_remote_then_repair():
    r = Router()
    r.client.set_dry_responses({"photosynthesis": "Plants convert sunlight, "
                                "water and CO2 into glucose and oxygen using "
                                "chlorophyll in their chloroplasts."})
    out = r.solve_all([{"task_id": "f1",
                        "prompt": "Explain how photosynthesis works."}], DEADLINE)
    assert out[0].route == Route.REMOTE_CHEAP
    assert out[0].answer.startswith("Plants convert")


def test_failsafe_never_empty():
    r = Router()
    r.tiers = {}                                # simulate no remote available
    r.controller.tiers = {}
    out = r.solve_all([{"task_id": "x1", "prompt": "Explain quantum tunneling."}],
                      DEADLINE)
    assert out[0].answer != ""


def test_sentiment_batching_one_call():
    r = Router()
    r.client.set_dry_responses({"numbered text": "1: neutral\n2: neutral\n3: positive"})
    out = r.solve_all([
        {"task_id": "s1", "prompt": "Sentiment?\nText: The camera is good but the battery is bad."},
        {"task_id": "s2", "prompt": "Sentiment?\nText: Nice design but slow delivery."},
        {"task_id": "s3", "prompt": "Sentiment?\nText: Genuinely delightful, exceeded expectations, absolutely wonderful."},
    ], DEADLINE)
    by_id = {o.task_id: o for o in out}
    # s3 is answered by the free lexicon; s1+s2 share ONE batched call
    assert by_id["s3"].remote_tokens == 0
    assert by_id["s1"].answer == "neutral" and by_id["s2"].answer == "neutral"
    assert r.ledger.calls == 1


def test_batch_bad_line_falls_back_individually():
    r = Router()
    r.client.set_dry_responses({
        "numbered text": "1: neutral\n2: banana\n3: neutral",   # item 2 invalid
        "label only": "negative",                                # individual fallback
    })
    out = r.solve_all([
        {"task_id": "b1", "prompt": "Sentiment?\nText: The camera is good but the battery is bad."},
        {"task_id": "b2", "prompt": "Sentiment?\nText: Great screen but poor sound."},
        {"task_id": "b3", "prompt": "Sentiment?\nText: Nice design but slow delivery."},
    ], DEADLINE)
    by_id = {o.task_id: o for o in out}
    assert by_id["b1"].answer == "neutral" and by_id["b3"].answer == "neutral"
    assert by_id["b2"].answer == "negative"        # solved via individual ladder
    assert r.ledger.calls == 2                     # 1 batch + 1 fallback


def test_local_llm_consistent_factual_zero_tokens():
    from gemmagate.local_model import StubLocal
    r = Router()
    draft = ("Photosynthesis is the process by which plants use chlorophyll to "
             "convert sunlight, water and carbon dioxide into glucose and oxygen.")
    r.controller.local_model = StubLocal({"photosynthesis": draft})
    out = r.solve_all([{"task_id": "d1",
                        "prompt": "Explain how photosynthesis works."}], DEADLINE)
    assert out[0].answer == draft
    assert out[0].route == Route.LOCAL_MODEL
    assert r.ledger.calls == 0                 # ZERO remote: rule-sanctioned


def test_local_llm_inconsistent_escalates_with_draft():
    from gemmagate.local_model import StubLocal
    r = Router()
    r.controller.local_model = StubLocal({"gravity": [
        "Gravity is a magnetic force between metals in the ground.",
        "Gravity is what makes apples taste better in autumn weather."]})
    corrected = ("Gravity is the attraction between masses; on Earth it pulls "
                 "objects toward the planet's center at about 9.8 m/s squared.")
    r.client.set_dry_responses({"proposed": corrected})
    out = r.solve_all([{"task_id": "d2",
                        "prompt": "Explain what gravity is."}], DEADLINE)
    assert out[0].answer == corrected          # samples disagreed -> remote fixed it
    assert r.ledger.calls == 1                 # draft attached, single call


def test_local_llm_codegen_needs_passing_examples():
    from gemmagate.local_model import StubLocal
    r = Router()
    good = "def double(n):\n    return n * 2"
    r.controller.local_model = StubLocal({"double": good})
    out = r.solve_all([{"task_id": "c1", "prompt":
        "Write a Python function double(n) that doubles a number. "
        "For example double(4) should return 8."}], DEADLINE)
    assert out[0].answer == good and r.ledger.calls == 0
    r2 = Router()
    r2.controller.local_model = StubLocal(
        {"triple": "def triple(n):\n    return n + 3"})
    r2.client.set_dry_responses({"triple": "def triple(n):\n    return n * 3"})
    out2 = r2.solve_all([{"task_id": "c2", "prompt":
        "Write a Python function triple(n) that triples a number. "
        "For example triple(2) should return 6."}], DEADLINE)
    assert "n * 3" in out2[0].answer and r2.ledger.calls >= 1




def test_payload_numbers_dont_outvote_summarize():
    s = _spec("Summarize the following passage in 2 sentences.\nText: The board "
              "voted 7-2 after twenty percent gains over five years of planning.")
    assert s.category == Category.SUMMARIZATION      # not math!


def test_neutral_zero_signal():
    s = _spec("What is the sentiment (positive, negative, or neutral)?\n"
              "Text: The package arrived on Tuesday as scheduled.")
    assert sentiment.solve(s).startswith("neutral")


def test_multi_rate_distance():
    assert math_solver.solve("A car travels at 60 km per hour for 4 hours and "
                             "then 80 km per hour for 2 hours. What is the "
                             "total distance traveled?") == "400"


def test_logic_question_word_not_a_name():
    s = _spec("Dan finished the race before Erin. Erin finished before Frank. "
              "Who finished last?")
    assert logic.solve(s) == "Frank"


def test_math_track1_patterns():
    cases = {
        "A $200 item is increased by 10% and then decreased by 20%. What is the final price?": "176",
        "Divide $600 in the ratio 2:3. How much is each share?": "240 and 360",
        "Split 600 in the ratio 2:3. What is the larger share?": "360",
        "A laptop costs $50 plus 8% sales tax. What is the total price?": "54",
        "A shop item costs $60 and sells for $80. What is the profit margin?": "25%",
        "A shop item costs $60 and sells for $80. What is the markup?": "33.333333%",
    }
    for prompt, expected in cases.items():
        assert math_solver.solve(prompt) == expected, (prompt, math_solver.solve(prompt))


def test_sentiment_never_liked():
    s = _spec("Sentiment (positive or negative)?\nText: I never liked this brand "
              "and this purchase confirmed it, truly disappointing quality.")
    assert sentiment.solve(s).startswith("negative")


def test_ner_entities_list_schema():
    s = _spec('Extract entities. Return JSON: {"entities": [{"text": "...", '
              '"label": "PERSON|ORG|LOCATION|DATE"}]}.\n'
              "Text: Maria Chen visited Acme Corp in Tokyo yesterday.")
    assert s.ner_list
    out = json.loads(ner.solve(s))
    labels = {(e["text"], e["label"]) for e in out["entities"]}
    assert ("Maria Chen", "PERSON") in labels
    assert ("Acme Corp", "ORG") in labels
    assert ("Tokyo", "LOCATION") in labels
    assert ("yesterday", "DATE") in labels


def test_validator_converts_ner_schemas_both_ways():
    # remote answered keyed, task wants entities-list
    s1 = _spec('Extract entities. Return JSON: {"entities": [...]}.\n'
               "Text: Maria Chen visited Tokyo.")
    v1 = validate(s1, '{"person": ["Maria Chen"], "location": ["Tokyo"]}')
    assert v1.passed and "entities" in json.loads(v1.repaired)
    # remote answered entities-list, task wants keyed
    s2 = _spec("Extract entities as JSON with keys person, organization, "
               "location, date.\nText: Maria Chen visited Tokyo.")
    v2 = validate(s2, '{"entities": [{"text": "Maria Chen", "label": "PERSON"},'
                      '{"text": "Tokyo", "label": "LOCATION"}]}')
    assert v2.passed
    keyed = json.loads(v2.repaired)
    assert keyed["person"] == ["Maria Chen"] and keyed["location"] == ["Tokyo"]


def test_summary_bullets_local_and_repair():
    passage = ("Text: The council approved the transit plan on Tuesday. "
               "Supporters cited faster commutes across the region. "
               "Opponents warned about budget overruns. "
               "Construction begins next spring with phased funding.")
    s = _spec("Summarize in 3 bullet points.\n" + passage)
    assert s.category == Category.SUMMARIZATION and s.max_bullets == 3
    out = summarize.solve(s)
    assert out and len([l for l in out.splitlines() if l.startswith("- ")]) == 3
    # repair: 5 bullets truncated to 3; 2 bullets rejected (can't invent content)
    five = "\n".join(f"- point {i}" for i in range(5))
    v = validate(s, five)
    assert v.passed and v.repaired.count("- ") == 3
    assert not validate(s, "- only\n- two").passed


def test_code_gen_templates_self_tested():
    from gemmagate.solvers import code_gen
    s = _spec("Write a Python function fib(n) that returns the nth Fibonacci "
              "number, where fib(0) is 0.")
    code = code_gen.try_generate(s)
    assert code and "def fib(" in code
    ns = {}; exec(code, ns)
    assert ns["fib"](10) == 55
    # ambiguous / unknown concepts must escalate, not guess
    assert code_gen.try_generate(_spec(
        "Write a Python function solve(x) that inverts a matrix.")) is None
    assert code_gen.try_generate(_spec(
        "Write a function f(s) that checks if s is a palindrome and counts "
        "vowels.")) is None                      # two concepts => escalate


def test_code_gen_zero_tokens_end_to_end():
    r = Router()
    out = r.solve_all([{"task_id": "g1", "prompt":
        "Write a Python function is_palindrome(s) that returns True if s reads "
        "the same forwards and backwards, ignoring case."}], DEADLINE)
    assert out[0].remote_tokens == 0 and "def is_palindrome" in out[0].answer
    assert r.ledger.calls == 0


def test_debug_with_explanation():
    s = _spec("Identify the bug and explain your fix, then give corrected code."
              "\n```python\ndef f(n):\n    if n % 2 = 0:\n        return True"
              "\n    return False\n```")
    assert s.wants_justification
    from gemmagate.solvers import code_tools
    out = code_tools.try_local_fix(s)
    assert out and out.startswith("Bug:") and "==" in out
    assert validate(s, out).passed


def test_relative_dates():
    s = _spec("Extract entities as JSON.\nText: Maria Chen flies to Tokyo "
              "next Monday.")
    out = json.loads(ner.solve(s))
    assert "next Monday" in out["date"]


def test_fireworks_unavailable_failsafe():
    from gemmagate.schemas import LLMResult
    r = Router()
    r.client.complete = lambda *a, **k: LLMResult(   # simulate total outage
        text="", model="down", is_remote=True, error="connection refused")
    out = r.solve_all([
        {"task_id": "f1", "prompt": "Explain how photosynthesis works."},
        {"task_id": "f2", "prompt": "Calculate 6 * 7."},
    ], DEADLINE)
    by_id = {o.task_id: o for o in out}
    assert "42" in by_id["f2"].answer            # local path unaffected
    assert by_id["f1"].answer.strip() != ""      # non-empty failsafe


def test_router_classification_table():
    table = {
        "Explain how vaccines work.": Category.FACTUAL,
        "What is 15% of 80?": Category.MATH,
        "Classify the sentiment of this tweet as positive or negative.\nText: meh":
            Category.SENTIMENT,
        "Summarize this article in 2 sentences.\nText: Lorem ipsum dolor.":
            Category.SUMMARIZATION,
        "Extract all named entities from the text. Output JSON.\nText: Bob.":
            Category.NER,
        "Find the bug in this function.\n```python\ndef f(): pass\n```":
            Category.CODE_DEBUG,
        "Amy is older than Ben. Ben is older than Cal. Who is the youngest?":
            Category.LOGIC,
        "Write a Python function add(a, b) that returns the sum.":
            Category.CODE_GEN,
    }
    for prompt, expected in table.items():
        assert _spec(prompt).category == expected, prompt


def test_no_empty_answers_full_suite():
    r = Router()
    tasks = [{"task_id": f"x{i}", "prompt": p} for i, p in enumerate([
        "Calculate 2+2.", "", "???", "Summarize.\nText: Hi.", "zorp"])]
    out = r.solve_all(tasks, DEADLINE)
    assert len(out) == len(tasks)
    assert all(isinstance(o.answer, str) and o.answer.strip() != "" for o in out)


def test_divide_ratio_classifies_math_and_solves_free():
    r = Router()
    out = r.solve_all([{"task_id": "m6", "prompt":
        "Divide $600 between two partners in the ratio 2:3. "
        "What is the larger share?"}], DEADLINE)
    assert out[0].category == Category.MATH
    assert "360" in out[0].answer and out[0].remote_tokens == 0


def test_hard_math_patterns():
    cases = {
        "A jacket priced at $180 is discounted by 20%, then a further 15% is taken off the reduced price. What is the final price?": "122.4",
        "Profits of $840 are split among three partners in the ratio 3:2:2. What is the largest share?": "360",
        "A shop buys a lamp for $48 and sells it at a 25% markup, then adds 10% sales tax at checkout. What does the customer pay?": "66",
        "A warehouse has 480 units. It ships 25% on Monday, receives 90 units on Tuesday, then ships 30 more on Wednesday. How many units remain?": "420",
    }
    for prompt, expected in cases.items():
        got = math_solver.solve(prompt)
        assert got == expected, (prompt, got)


def test_hard_assignment_and_ner():
    s = _spec("Three colleagues, Ana, Boris, and Chen, each drink a different "
              "beverage: coffee, tea, juice. Boris does not drink tea. "
              "Chen drinks the juice. Who drinks the tea?")
    assert logic.solve(s) == "Ana"
    s2 = _spec("Extract entities as JSON.\nText: Executives from Nakamura "
               "Industries met in Singapore yesterday.")
    out = json.loads(ner.solve(s2))
    assert "Nakamura Industries" in out["organization"]


def test_ai_trap_math():
    assert math_solver.solve(
        "A cyclist rides to town at 60 km/h and returns along the same road "
        "at 40 km/h. What is her average speed for the whole trip?") == "48"
    assert math_solver.solve(
        "Alice can paint a fence in 6 hours and Bob can paint it in 3 hours. "
        "Working together, how long will it take them?") == "2"


def test_extreme_batch_regressions():
    # double-discount illusion: 150 *1.2 *0.8 = 144, not 150
    assert math_solver.solve(
        "A shop raises a $150 price by 20%, then applies a 20% discount to "
        "the new price. What is the actual final price?") == "144"
    # narrative with baking verb
    assert math_solver.solve(
        "A bakery has 360 rolls. It sells 25% in the morning, bakes 40 more "
        "at noon, then sells 130 in the afternoon. How many rolls remain?") == "180"
    # chained comparative + 5 names
    s = _spec("Five sprinters raced. Kira finished before Dev. Dev finished "
              "before Mo. Tessa finished after Mo but before Yuri. "
              "Who finished last?")
    assert logic.solve(s) == "Yuri"
    # 4-person assignment
    s2 = _spec("Four flatmates, Ana, Boris, Chen, and Dana, each play a "
               "different instrument: guitar, piano, drums, violin. Ana does "
               "not play the drums. Chen plays the piano. Dana does not play "
               "the guitar. Boris plays the drums. Who plays the violin?")
    assert logic.solve(s2) == "Dana"
    # snark lexicon
    s3 = _spec("Classify the sentiment (positive, negative, neutral, or mixed):"
               "\nFive stars for the courier, I suppose. Shame the actual "
               "product lasted a whole two days before dying.")
    assert sentiment.solve(s3).startswith("negative")
    # mutable-default mechanical fix, executable
    from gemmagate.solvers import code_tools
    s4 = _spec("Fix the bug caused by the mutable default argument.\n"
               "```python\ndef add_item(item, bag=[]):\n    bag.append(item)\n"
               "    return bag\n```")
    fixed = code_tools.try_local_fix(s4)
    assert fixed and "None" in fixed
    ns = {}; exec(fixed, ns)
    assert ns["add_item"]("a") == ["a"] and ns["add_item"]("a") == ["a"]
    # balanced-parens template
    from gemmagate.solvers import code_gen
    s5 = _spec("Write a Python function balanced(s) that returns True if the "
               "parentheses in s are balanced.")
    code = code_gen.try_generate(s5)
    assert code and "depth" in code


def test_oracle_repair_engine():
    from gemmagate.solvers import code_repair
    # intent oracle: "max of a list" with best=0 bug (fails on negatives)
    s = _spec("This function should return the max of a list but has a bug: "
              "def get_max(nums): return nums[0]. Find and fix it.")
    # off-by-one range with intent oracle
    s2 = _spec("This function should count how many times target appears in "
               "nums, but it has a bug. Fix it.\n```python\n"
               "def count_target(nums, target):\n    count = 0\n"
               "    for i in range(1, len(nums)):\n"
               "        if nums[i] == target:\n            count += 1\n"
               "    return count\n```")
    fixed = code_repair.attempt(s2)
    assert fixed
    ns = {}; exec(fixed, ns)
    assert ns["count_target"]([5, 2, 5, 5], 5) == 3
    # running-average divide-by-index
    s3 = _spec("Fix the bug in this running-average function.\n```python\n"
               "def running_avg(values):\n    total = 0\n    avgs = []\n"
               "    for i, v in enumerate(values):\n        total += v\n"
               "        avgs.append(total / i)\n    return avgs\n```")
    fixed3 = code_repair.attempt(s3)
    assert fixed3
    ns3 = {}; exec(fixed3, ns3)
    assert ns3["running_avg"]([2, 4]) == [2.0, 3.0]
    # oracle satisfied by original => no "fix" invented
    ok = _spec("This should return the max of a list.\n```python\n"
               "def get_max(nums):\n    return max(nums)\n```")
    assert code_repair.attempt(ok) is None


def test_double_negation_positive():
    s = _spec("Classify the sentiment as positive, negative, neutral, or mixed:"
              "\nI wouldn't say it's not worth the money — honestly, not bad "
              "at all, and the support team never disappointed me once.")
    assert sentiment.solve(s).startswith("positive")


def test_new_codegen_templates():
    from gemmagate.solvers import code_gen
    cases = {
        "Write a Python function merge_sorted(a, b) that merges two sorted lists.": "merge_sorted",
        "Write a Python function is_anagram(a, b) that checks if two words are anagrams.": "is_anagram",
        "Write a Python function sum_digits(n) that returns the sum of its digits.": "sum_digits",
        "Write a Python function is_power(n) that returns True if n is a power of two.": "is_power",
        "Write a Python function bsearch(nums, target) that performs binary search on a sorted list and returns the index or -1.": "bsearch",
        "Write a Python function flatten(items) that flattens a nested list.": "flatten",
    }
    for prompt, fname in cases.items():
        code = code_gen.try_generate(_spec(prompt))
        assert code and f"def {fname}(" in code, prompt


def test_hard_set_final_gaps():
    # fenced buggy code + "how many times" must be CODE_DEBUG, not math
    s = _spec("This function should count how many times target appears in "
              "nums, but it has a bug. Provide the corrected implementation."
              "\n```python\ndef count_target(nums, target):\n    count = 0\n"
              "    for i in range(1, len(nums)):\n        if nums[i] == target:"
              "\n            count += 1\n    return count\n```")
    assert s.category == Category.CODE_DEBUG
    # "exactly one sentence" parses as a hard constraint
    s2 = _spec("Summarize the following in exactly one sentence.\nText: "
               "First fact here. Second point follows. Third item ends it.")
    assert s2.max_sentences == 1
    out = summarize.solve(s2)
    assert out and out.count(".") <= 1
    # sarcasm + damage word answered locally
    s3 = _spec("Classify the sentiment (positive, negative, neutral, or mixed):"
               "\nOh great, another firmware update that bricks the camera. "
               "Just what I needed this week.")
    assert sentiment.solve(s3).startswith("negative")


def test_word_trim_keeps_main_clause():
    s = _spec("Summarize the following in at most 12 words:\nText: After "
              "months of negotiation, the two rail unions and the transport "
              "ministry reached a tentative agreement on pay, scheduling and "
              "safety staffing, averting a strike that analysts warned could "
              "have halted freight across three countries.")
    out = summarize.solve(s)
    assert out and len(out.split()) <= 12
    assert "reached" in out          # the claim survives the trim


def test_judge_facing_presentation():
    from gemmagate.present import polish
    m = _spec("A shop splits $600 in the ratio 2:3. What is the larger share?")
    assert polish(m, "360") == "The answer is $360."
    lg = _spec("Kira finished before Dev. Dev finished before Mo. Who finished last?")
    assert polish(lg, "Mo") == "The answer is Mo."
    bare = _spec("What is 15% of 200? Answer with only the number.")
    assert polish(bare, "30") == "30"
    # sentiment default justification, bare guard
    sj = _spec("Classify the sentiment: The build is flawless and reliable.")
    out = sentiment.solve(sj)
    assert out.startswith("positive") and len(out) > len("positive")
    sb = _spec("Classify the sentiment, answer with only the label: "
               "The build is flawless and reliable.")
    assert sentiment.solve(sb) == "positive"


def test_short_answer_batch_parse():
    from gemmagate.batcher import ShortAnswerBatcher
    class _C:
        def complete(self, model, prompt, max_tokens):
            class R: text = "1. Paris is the capital of France.\n2) Jane Austen wrote it."; total_tokens = 60
            return R()
    class _S:
        def __init__(self, tid, p):
            sp = _spec(p); sp.task_id = tid
            self.__dict__ = sp.__dict__
    import time as _t
    b = ShortAnswerBatcher(_C(), "m")
    s1 = _spec("What is the capital of France?"); s1.task_id = "f1"
    s2 = _spec("Who wrote Pride and Prejudice?"); s2.task_id = "f2"
    out = b.solve([(0, s1), (1, s2)], _t.time() + 60)
    assert len(out) == 2 and "Paris" in out[0].answer and "Austen" in out[1].answer
    assert out[0].remote_tokens == 30


def test_qualifier_mode_routes_risky_categories_strong_first():
    from gemmagate.escalation import EscalationController
    from gemmagate.remote import FireworksClient, Ledger
    old_q = os.environ.get("GEMMAGATE_QUALIFIER_MODE")
    try:
        os.environ["GEMMAGATE_QUALIFIER_MODE"] = "1"
        c = EscalationController(
            FireworksClient(Ledger()),
            {"cheap": "m-8b", "mid": "m-34b", "strong": "m-70b"})
        s = _spec("Summarize this in one sentence. Text: Alpha beta gamma.")
        assert c._ladder(s)[0] == Route.REMOTE_STRONG
        m = _spec("What is 12 * 12?")
        assert c._ladder(m)[0] == Route.LOCAL_RULE
    finally:
        if old_q is None:
            os.environ.pop("GEMMAGATE_QUALIFIER_MODE", None)
        else:
            os.environ["GEMMAGATE_QUALIFIER_MODE"] = old_q


def _run():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    ok = 0
    for f in fns:
        try:
            f()
            print(f"PASS  {f.__name__}")
            ok += 1
        except AssertionError as e:
            print(f"FAIL  {f.__name__}: {e}")
        except Exception as e:
            print(f"ERROR {f.__name__}: {type(e).__name__}: {e}")
    print(f"\n{ok}/{len(fns)} passed")
    return ok == len(fns)


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
