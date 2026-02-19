import re
from typing import List


NEGATIONS = {"no", "not", "never", "none", "without", "cannot", "can't"}


def detect_contradictions(claims: List[str]) -> List[str]:
    contradictions: List[str] = []
    for i in range(len(claims)):
        for j in range(i + 1, len(claims)):
            left = claims[i].strip()
            right = claims[j].strip()
            if _negation_conflict(left, right) or _numeric_conflict(left, right):
                contradictions.append(f"Potential contradiction between: '{left[:80]}' and '{right[:80]}'")
    return contradictions


def _negation_conflict(a: str, b: str) -> bool:
    a_tokens = _tokens(a)
    b_tokens = _tokens(b)
    shared = a_tokens & b_tokens
    if len(shared) < 3:
        return False
    a_neg = bool(a_tokens & NEGATIONS)
    b_neg = bool(b_tokens & NEGATIONS)
    return a_neg != b_neg


def _numeric_conflict(a: str, b: str) -> bool:
    a_nums = _numbers(a)
    b_nums = _numbers(b)
    if not a_nums or not b_nums:
        return False
    shared_context = len(_tokens(a) & _tokens(b)) >= 3
    return shared_context and set(a_nums) != set(b_nums)


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z']+", text.lower()))


def _numbers(text: str) -> List[str]:
    return re.findall(r"\b\d+(?:\.\d+)?%?\b", text)

