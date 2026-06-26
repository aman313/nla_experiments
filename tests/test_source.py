from nla_sycophancy.data.source import (
    filter_items,
    parse_opentriviaqa,
)

FIXTURE = """

#Q What is the capital of Afghanistan?
^ Kabul
A Tirana
B Kabul
C Dushanbe
D Tashkent

#Q What is the capital of Australia?
^ Canberra
A Canberra
B Sydney
C Melbourne
D Ottawa

#Q Malformed with no matching correct
^ Nowhere
A Here
B There
C Everywhere
D Somewhere

#Q Too few opts?
^ Yes
A Yes
B No
"""


def test_parse_basic():
    items = parse_opentriviaqa(FIXTURE, category="geography")
    # malformed (no matching correct) is dropped; the 2-option one is kept here
    qs = {it.question for it in items}
    assert "What is the capital of Afghanistan?" in qs
    assert "Malformed with no matching correct" not in qs
    afg = next(it for it in items if "Afghanistan" in it.question)
    assert afg.options[afg.correct_idx] == "Kabul"
    assert afg.correct_idx == 1


def test_filter_n_options():
    items = parse_opentriviaqa(FIXTURE)
    four = filter_items(items, n_options=4)
    assert all(it.n_options == 4 for it in four)
    assert len(four) == 2  # the two well-formed capitals


def test_filter_dedupe_and_dupopts():
    dup = FIXTURE + FIXTURE  # duplicate every block
    items = parse_opentriviaqa(dup)
    four = filter_items(items, n_options=4, dedupe=True)
    assert len(four) == 2  # dedup collapses the repeats
