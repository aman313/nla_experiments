from nla_sycophancy.target.rollout import parse_answer_letter


def test_parse_bare_letter():
    assert parse_answer_letter("B", 4) == 1
    assert parse_answer_letter("The answer is C.", 4) == 2
    assert parse_answer_letter("(A)", 4) == 0
    assert parse_answer_letter("D) Madrid", 4) == 3


def test_parse_repeated_single_ok():
    assert parse_answer_letter("A. A is correct, so A.", 4) == 0


def test_parse_ambiguous_returns_none():
    assert parse_answer_letter("Could be A or B", 4) is None


def test_parse_out_of_range_letter_ignored():
    # E is out of range for 4 options
    assert parse_answer_letter("E", 4) is None


def test_parse_no_letter_returns_none():
    assert parse_answer_letter("I cannot answer that.", 4) is None
