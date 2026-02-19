from research_assistant.utils.contradictions import detect_contradictions


def test_detects_negation_conflict():
    claims = [
        "AI tutoring significantly improves exam scores in most schools.",
        "AI tutoring does not improve exam scores in most schools.",
    ]
    found = detect_contradictions(claims)
    assert found


def test_detects_numeric_conflict():
    claims = [
        "Renewable energy adoption reached 45% in 2025 in the region.",
        "Renewable energy adoption reached 30% in 2025 in the region.",
    ]
    found = detect_contradictions(claims)
    assert found

