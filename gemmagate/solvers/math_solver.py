"""Deterministic math solver — zero tokens, exact when it applies.

Handles: direct arithmetic expressions (AST-whitelisted safe eval),
aggregates (sum/mean/median/min/max/product), percentage patterns
("15% of 80", "increase 240 by 15%", "20% discount on 50"), and simple
unit-rate projections ("$12 per hour for 8 hours").

Precision rule: bail out (return None) on any ambiguity — a wrong
deterministic answer is worse than an escalation. Multi-step word problems
fall through to the remote layer by design.
"""
from __future__ import annotations

import ast
import math
import operator
import re
import statistics
from typing import Optional

_BINOPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
           ast.Mod: operator.mod, ast.Pow: operator.pow}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {"sqrt": math.sqrt, "abs": abs, "round": round, "floor": math.floor,
          "ceil": math.ceil, "log": math.log, "log2": math.log2,
          "log10": math.log10, "exp": math.exp, "factorial": math.factorial,
          "gcd": math.gcd, "min": min, "max": max, "pow": pow}


def safe_eval(expr: str) -> Optional[float]:
    expr = expr.replace("^", "**").replace("×", "*").replace("÷", "/").replace(",", "")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return None

    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _BINOPS:
            return _BINOPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _UNARY:
            return _UNARY[type(n.op)](ev(n.operand))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) \
                and n.func.id in _FUNCS and not n.keywords:
            return _FUNCS[n.func.id](*[ev(a) for a in n.args])
        raise ValueError("disallowed")

    try:
        v = ev(tree)
        return v if isinstance(v, (int, float)) and math.isfinite(v) else None
    except Exception:
        return None


def fmt(x: float) -> str:
    if isinstance(x, float) and x.is_integer():
        return str(int(x))
    return f"{x:.10g}" if isinstance(x, float) else str(x)


_FUNC_EXPR = re.compile(
    r"(?:sqrt|abs|floor|ceil|log2|log10|log|exp|factorial|gcd|min|max|pow)"
    r"\([\d\s\.,\+\-\*/\^%\(\)]*\)", re.I)
_ARITH_EXPR = re.compile(r"[\d\.\(][\d\s\.\+\-\*/×÷\^%\(\)]*[\d\)]")
_HAS_OP = re.compile(r"\d\s*[\+\-\*/×÷\^%]\s*[\d\(]|\)\s*[\+\-\*/×÷\^%]")


def _find_expression(text: str) -> Optional[float]:
    best = None
    cands = [m.group(0) for m in _FUNC_EXPR.finditer(text)]
    cands += [m.group(0) for m in _ARITH_EXPR.finditer(text) if _HAS_OP.search(m.group(0))]
    for c in cands:
        v = safe_eval(c.strip())
        if v is not None and (best is None or len(c) > len(best[0])):
            best = (c, v)
    return best[1] if best else None


_NUM = r"(-?\d+(?:,\d{3})*(?:\.\d+)?)"


def _f(s: str) -> float:
    return float(s.replace(",", ""))


def solve(prompt: str) -> Optional[str]:
    text = prompt.replace("\u00a3", "$").replace("\u20ac", "$")

    # AI-trap: round-trip average speed = HARMONIC mean (LLMs answer the
    # arithmetic mean). "60 km/h there, 40 km/h back, average speed?" -> 48
    if re.search(r"\baverage speed\b", text, re.I) and \
            re.search(r"\b(returns?|back|round.?trips?|comes? back|heads? back)\b", text, re.I):
        sp = re.findall(_NUM + r"\s*(?:km/?h|kph|mph|km per hour|miles per hour)", text, re.I)
        if len(sp) == 2:
            a, b = _f(sp[0]), _f(sp[1])
            if a > 0 and b > 0:
                return fmt(round(2 * a * b / (a + b), 6))

    # AI-trap: combined work rate. "A paints in 6 hours, B in 3 hours,
    # together?" -> 1/(1/6+1/3) = 2
    if re.search(r"\btogether\b", text, re.I) and \
            re.search(r"\bhow (?:long|many (?:hours?|minutes?|days?))\b", text, re.I):
        rates = re.findall(r"\bin\s+" + _NUM + r"\s+(hours?|minutes?|days?)\b", text, re.I)
        if 2 <= len(rates) <= 3 and len({u.rstrip("s").lower() for _, u in rates}) == 1:
            vals = [_f(v) for v, _ in rates]
            if all(v > 0 for v in vals):
                return fmt(round(1 / sum(1 / v for v in vals), 6))

    # narrative sequential arithmetic: "A store has 240 items. It sells 15%
    # on Monday and 60 more on Tuesday. How many remain?"
    if re.search(r"\bhow (?:many|much)\b.{0,40}\b(remain|left|now|are there|does .{1,25} have)\b",
                 text, re.I):
        ms = re.search(r"\b(?:has|have|had|starts? with|there (?:are|were)|begins? with)\s+"
                       + _NUM + r"\b", text, re.I)
        if ms:
            val = _f(ms.group(1))
            ops = 0
            _SUB = r"(?:sells?|sold|loses?|lost|removes?|gives? away|gave away|uses?|used|eats?|ate|breaks?|broke|donates?|ships?|shipped)"
            _ADD = r"(?:buys?|bought|adds?|added|receives?|received|gains?|gained|restocks?|gets?|got|makes?|made|produces?|finds?|found|bakes?|baked|brews?|builds?|built|creates?|prints?|delivers? in|stocks?)"
            _ANYV = _SUB[:-1] + "|" + _ADD[3:]      # merged alternation
            # each clause ends at the NEXT action verb or sentence end, so a
            # later verb's numbers can't be swallowed under the wrong sign
            for vm in re.finditer(r"\b(" + _SUB + r"|" + _ADD + r")\b"
                                  r"((?:(?!\b" + _ANYV + r")[^.!?])*)",
                                  text[ms.end():], re.I):
                sign = -1 if re.match(_SUB, vm.group(1), re.I) else 1
                clause = vm.group(2)
                for tok in re.finditer(_NUM + r"\s*(%|percent)?", clause):
                    amt = _f(tok.group(1))
                    if tok.group(2):
                        val += sign * val * amt / 100
                    else:
                        val += sign * amt
                    ops += 1
            if ops >= 1:
                return fmt(round(val, 6))

    # successive percentage changes: "increased by 10% and then decreased by 20%"
    changes = re.findall(r"(increas|decreas|ris|rais|drop|fall|discount|reduc|grow)\w*"
                         r"[^%]{0,30}?by\s+" + _NUM + r"\s*(?:%|percent)", text, re.I)
    changes += [("decreas", pct) for pct in re.findall(
        r"(?:a further|another|an additional|then)\D{0,20}?" + _NUM +
        r"\s*(?:%|percent)\s*(?:is|are)?\s*(?:taken off|off|discount\w*|deducted|removed)",
        text, re.I)]
    if len(changes) >= 2:
        mb = re.search(r"\$?" + _NUM, text)
        if mb:
            val = _f(mb.group(1))
            for verb, pct in changes:
                sign = 1 if verb.lower() in ("increas", "ris", "rais", "grow") else -1
                val *= (1 + sign * _f(pct) / 100)
            return fmt(round(val, 6))


    # percentages first (an expression scan would misread "15% of 80")
    m = re.search(_NUM + r"\s*(?:%|percent)\s*of\s*\$?" + _NUM, text, re.I)
    if m:
        return fmt(_f(m.group(1)) / 100 * _f(m.group(2)))
    def _chg_sign(stem: str, direction: str) -> int:
        if direction:
            return 1 if direction.lower() == "up" else -1
        return 1 if stem.lower().startswith(("increas", "rais", "ris", "grow")) else -1

    _CHG = r"(increas\w*|decreas\w*|rais\w*|reduc\w*|discount\w*|mark\w*|slash\w*|cut)"
    m = re.search(_CHG + r"(?:\s+(up|down))?\D{0,40}?\$?" + _NUM +
                  r"(?!\s*%)[^%\d]{0,25}?by\s+" + _NUM + r"\s*(?:%|percent)", text, re.I)
    if m:
        base, pct = _f(m.group(3)), _f(m.group(4))
        return fmt(base * (1 + _chg_sign(m.group(1), m.group(2)) * pct / 100))
    # base BEFORE the verb: "a jacket priced at $180 is marked down by 15%"
    m = re.search(r"\$?" + _NUM + r"\D{0,30}?(?:is|was|are|were|gets?|got)\s+" +
                  _CHG + r"(?:\s+(up|down))?\s*by\s+" + _NUM +
                  r"\s*(?:%|percent)", text, re.I)
    if m:
        base, pct = _f(m.group(1)), _f(m.group(4))
        return fmt(base * (1 + _chg_sign(m.group(2), m.group(3)) * pct / 100))
    m = re.search(_NUM + r"\s*(?:%|percent)\s*(discount|off)\s*(?:on|of)?\s*\$?" + _NUM, text, re.I)
    if m:
        return fmt(_f(m.group(3)) * (1 - _f(m.group(1)) / 100))
    m = re.search(r"what percent(?:age)? (?:of|is)\s*\$?" + _NUM + r"\s*(?:is|of)\s*\$?" + _NUM, text, re.I)
    if m:
        a, b = _f(m.group(1)), _f(m.group(2))
        if "of" in m.group(0).lower().split("percent")[1][:20]:
            a, b = b, a
        if b:
            return fmt(round(a / b * 100, 6)) + "%"

    # ratio split: "divide $600 in the ratio 2:3" / "split 600 into the ratio 2 to 3"
    m = re.search(r"\$?" + _NUM + r"\D{0,40}?(?:divided|split|shared|distributed)"
                  r"\D{0,40}?ratio\s*(?:of\s*)?(\d+)\s*(?::|to)\s*(\d+)"
                  r"(?:\s*(?::|to)\s*(\d+))?", text, re.I) or \
        re.search(r"(?:divide|split|share)\D{0,20}?\$?" + _NUM +
                  r"\D{0,60}?ratio\s*(?:of\s*)?(\d+)\s*(?::|to)\s*(\d+)"
                  r"(?:\s*(?::|to)\s*(\d+))?", text, re.I)
    if m:
        total = _f(m.group(1))
        parts = [int(m.group(2)), int(m.group(3))] + ([int(m.group(4))] if m.group(4) else [])
        denom = sum(parts)
        shares = [total * p / denom for p in parts]
        if re.search(r"\b(larg|great|bigg|most)\w*\b", text, re.I):
            return fmt(max(shares))
        if re.search(r"\b(small|least|lowest)\w*\b", text, re.I):
            return fmt(min(shares))
        return " and ".join(fmt(x) for x in shares)

    # cost -> markup -> tax chain: "buys for $48 ... 25% markup ... 10% tax"
    if re.search(r"\b(customer pay|final price|total price|checkout|pays?)\b", text, re.I):
        mc = re.search(r"(?:buys?|bought|costs?|cost price)\D{0,30}?\$?" + _NUM, text, re.I)
        mk = re.search(_NUM + r"\s*%\s*markup", text, re.I)
        if mc and mk:
            price = _f(mc.group(1)) * (1 + _f(mk.group(1)) / 100)
            mt = re.search(_NUM + r"\s*%\s*(?:sales\s+)?tax", text, re.I)
            if mt:
                price *= (1 + _f(mt.group(1)) / 100)
            return fmt(round(price, 6))

    # sales tax: "$50 plus 8% tax" -> 54
    if re.search(r"\btax\b", text, re.I):
        m = re.search(r"\$?" + _NUM + r"\D{0,40}?" + _NUM + r"\s*%\s*(?:sales\s+)?tax",
                      text, re.I)
        if m:
            base, rate = _f(m.group(1)), _f(m.group(2))
            if re.search(r"\b(tax alone|just the tax|how much tax|amount of tax)\b", text, re.I):
                return fmt(round(base * rate / 100, 6))
            return fmt(round(base * (1 + rate / 100), 6))

    # profit margin / markup: cost + selling price
    if re.search(r"\b(margin|markup)\b", text, re.I):
        mc = re.search(r"(?:costs?|bought|cost price(?: of)?)\D{0,30}?\$?" + _NUM, text, re.I)
        ms = re.search(r"(?:sells?|sold|selling price(?: of)?)\D{0,30}?\$?" + _NUM, text, re.I)
        if mc and ms:
            cost, sell = _f(mc.group(1)), _f(ms.group(1))
            if sell and cost:
                if re.search(r"\bmarkup\b", text, re.I):
                    return fmt(round((sell - cost) / cost * 100, 6)) + "%"
                return fmt(round((sell - cost) / sell * 100, 6)) + "%"

    # aggregates over explicit lists
    m = re.search(r"\b(sum|total|average|mean|median|max(?:imum)?|min(?:imum)?|product)\s+of\s+"
                  r"(?:the\s+)?(?:numbers?|values|list)?\s*:?\s*"
                  r"((?:" + _NUM + r"[,\s]+(?:and\s+)?)+" + _NUM + r")", text, re.I)
    if m:
        nums = [_f(n) for n in re.findall(_NUM, m.group(2))]
        if len(nums) >= 2:
            op = m.group(1).lower()
            try:
                if op in ("sum", "total"):
                    return fmt(sum(nums))
                if op in ("average", "mean"):
                    return fmt(statistics.fmean(nums))
                if op == "median":
                    return fmt(statistics.median(nums))
                if op.startswith("max"):
                    return fmt(max(nums))
                if op.startswith("min"):
                    return fmt(min(nums))
                if op == "product":
                    return fmt(math.prod(nums))
            except statistics.StatisticsError:
                return None

    # ratio simplification: "simplify the ratio 12 to 30" / "ratio of 12:30"
    if re.search(r"\b(simplif\w+|simplest|lowest|reduce\w*)\b", text, re.I):
        m = re.search(r"ratio (?:of )?" + _NUM + r"\s*(?:to|:)\s*" + _NUM, text, re.I)
        if m:
            a, b = _f(m.group(1)), _f(m.group(2))
            if a > 0 and b > 0 and a.is_integer() and b.is_integer():
                g = math.gcd(int(a), int(b))
                return f"{int(a) // g}:{int(b) // g}"

    # item totals: "7 tickets at $15 each" / "buys 3 books for $12 each"
    m = re.search(_NUM + r"\s+\w+?s?\s+(?:at|for|costing)\s+\$?" + _NUM +
                  r"\s*(?:each|apiece|per\s+\w+)\b", text, re.I)
    if m and re.search(r"\b(total|cost|pay|spend|how much|revenue|earn)\b", text, re.I):
        return fmt(_f(m.group(1)) * _f(m.group(2)))

    # profit/loss: "bought ... for $X ... sold ... for $Y" + profit question
    if re.search(r"\b(profit|loss|gain)\b", text, re.I):
        mb = re.search(r"(?:bought|buys?|purchased?|acquired?)\D{0,50}?\$?" + _NUM, text, re.I)
        ms = re.search(r"(?:sold|sells?)\D{0,50}?\$?" + _NUM, text, re.I)
        if mb and ms:
            return fmt(abs(_f(ms.group(1)) - _f(mb.group(1))))

    # percent change: "from $50 to $65, what is the percentage increase"
    if re.search(r"\bpercent(age)?\s+(increase|decrease|change|growth)\b", text, re.I):
        m = re.search(r"from\s+\$?" + _NUM + r"\s+to\s+\$?" + _NUM, text, re.I)
        if m:
            a, b = _f(m.group(1)), _f(m.group(2))
            if a:
                return fmt(round(abs(b - a) / a * 100, 6)) + "%"

    # compound growth: "$1000 grows by 10% per year ... after 3 years"
    m = re.search(r"\$?" + _NUM + r"\D{0,60}?(?:grows?|increases?|appreciates?)\s+"
                  r"(?:by\s+)?" + _NUM +
                  r"\s*%\s*(?:(?:per|each|every|a)\s+(year|month)|annually)"
                  r"\D{0,60}?(?:after|for|in)\s+" + _NUM + r"\s+(year|month)s?", text, re.I)
    if m and (m.group(3) or "year").lower() == m.group(5).lower():
        base, pct, n = _f(m.group(1)), _f(m.group(2)), _f(m.group(4))
        if n.is_integer() and 0 < n <= 50:
            return fmt(round(base * (1 + pct / 100) ** int(n), 6))

    # simple interest: "simple interest on $500 at 4% for 3 years"
    if re.search(r"\bsimple interest\b", text, re.I):
        m = re.search(r"\$?" + _NUM + r"\D{0,30}?at\s+" + _NUM +
                      r"\s*%\D{0,30}?for\s+" + _NUM + r"\s+year", text, re.I)
        if m:
            p, r, t = _f(m.group(1)), _f(m.group(2)), _f(m.group(3))
            interest = p * r / 100 * t
            if re.search(r"\b(total|amount|worth|balance)\b", text, re.I):
                return fmt(p + interest)
            return fmt(interest)

    # multi-segment rate totals: "60 km per hour for 4 hours and then
    # 80 km per hour for 2 hours ... total distance"
    if re.search(r"\btotal (distance|cost|amount|earnings?|pay)\b", text, re.I):
        pairs = re.findall(_NUM + r"\s*(?:km|miles?|mi|\$|dollars?)?\s*per\s*"
                           r"(hour|day|minute)\s+for\s+" + _NUM +
                           r"\s*(hour|day|minute)s?", text, re.I)
        if len(pairs) >= 1 and all(p[1].lower() == p[3].lower() for p in pairs):
            return fmt(sum(_f(p[0]) * _f(p[2]) for p in pairs))

    # unit-rate: "$12 per hour ... 8 hours" (single rate, single quantity only)
    rates = re.findall(r"\$?" + _NUM + r"\s*per\s*(hour|day|week|month|year|item|unit|mile|km)", text, re.I)
    if len(rates) == 1:
        unit = rates[0][1].lower()
        qty = re.findall(_NUM + r"\s*" + unit + r"s?\b", text, re.I)
        if len(qty) == 1:
            return fmt(_f(rates[0][0]) * _f(qty[0]))

    # bare expression
    v = _find_expression(text)
    if v is not None:
        return fmt(v)
    return None
