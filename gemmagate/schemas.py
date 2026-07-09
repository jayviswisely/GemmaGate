"""Core data structures for GemmaGate (competition harness edition)."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Category(str, Enum):
    FACTUAL = "factual_knowledge"
    MATH = "mathematical_reasoning"
    SENTIMENT = "sentiment_classification"
    SUMMARIZATION = "text_summarization"
    NER = "named_entity_recognition"
    CODE_DEBUG = "code_debugging"
    LOGIC = "logical_reasoning"
    CODE_GEN = "code_generation"
    UNKNOWN = "unknown"


class Risk(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Route(str, Enum):
    LOCAL_RULE = "local_rule"        # deterministic / rule-based solver
    LOCAL_MODEL = "local_model"      # small local LLM (free tokens)
    REMOTE_CHEAP = "remote_cheap"
    REMOTE_MID = "remote_mid"
    REMOTE_STRONG = "remote_strong"
    FAILSAFE = "failsafe"            # best-effort answer under failure/deadline


@dataclass
class TaskSpec:
    task_id: str
    prompt: str
    category: Category = Category.UNKNOWN
    cls_confidence: float = 1.0
    risk: Risk = Risk.MEDIUM
    # extracted structure
    payload: str = ""                # passage / code / text-to-analyze portion
    instruction: str = ""            # the ask portion
    wants_json: bool = False
    wants_justification: bool = False
    allowed_labels: list[str] = field(default_factory=list)   # sentiment etc.
    max_words: Optional[int] = None
    max_sentences: Optional[int] = None
    max_bullets: Optional[int] = None
    ner_list: bool = False        # entities-list schema requested
    meta: dict = field(default_factory=dict)
    language: Optional[str] = None   # for code tasks


@dataclass
class LLMResult:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    is_remote: bool = False
    error: Optional[str] = None


@dataclass
class Validation:
    passed: bool
    score: float = 0.0
    reasons: list[str] = field(default_factory=list)
    repaired: Optional[str] = None


@dataclass
class Attempt:
    route: Route
    model: str
    answer: str
    validation: Validation
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class Solved:
    task_id: str
    answer: str
    route: Route = Route.FAILSAFE
    category: Category = Category.UNKNOWN
    cls_confidence: float = 1.0
    risk: Risk = Risk.MEDIUM
    confidence: float = 0.0
    attempts: list[Attempt] = field(default_factory=list)
    remote_tokens: int = 0
    wall_time_s: float = 0.0
    ts: float = field(default_factory=time.time)
