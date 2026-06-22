"""exact_match / token_f1 scorer 与注册表。"""
from __future__ import annotations

from eval_vlm.data.schema import Sample
from eval_vlm.scoring import get_scorer, available_scorers
from eval_vlm.scoring.exact_match import normalize


def _sample():
    # scorer 只用 sample.id;reference 通过参数显式传入。
    return Sample(id="x")


def test_registry_has_exact_match():
    assert "exact_match" in available_scorers()


def test_normalize():
    assert normalize("  Red.  ") == "red"
    assert normalize("A CAT!") == "a cat"


def test_exact_match_hit():
    sc = get_scorer("exact_match")
    r = sc.score_one("red", "Red", _sample())
    assert r.score == 1.0
    assert r.detail["exact_match"] == 1.0


def test_exact_match_miss_but_contains():
    sc = get_scorer("exact_match")
    r = sc.score_one("The car is red.", "red", _sample())
    assert r.score == 0.0          # 非完全匹配
    assert r.detail["contains"] == 1.0


def test_aggregate_accuracy():
    sc = get_scorer("exact_match")
    s = _sample()
    results = [
        sc.score_one("red", "red", s),
        sc.score_one("blue", "red", s),
    ]
    agg = sc.aggregate(results)
    assert agg["accuracy"] == 0.5
    assert agg["num_scored"] == 2


def test_skipped_when_no_reference():
    sc = get_scorer("exact_match")
    r = sc.score_one("anything", None, Sample(id="y"))
    assert r.detail.get("skipped") is True


# ---- token_f1(开放式回答,如轮1描述) ----

def test_registry_has_token_f1():
    assert "token_f1" in available_scorers()


def test_token_f1_perfect():
    sc = get_scorer("token_f1")
    r = sc.score_one("画面是一只猫", "画面是一只猫", _sample())
    assert r.score == 1.0
    assert r.detail["precision"] == 1.0
    assert r.detail["recall"] == 1.0


def test_token_f1_partial():
    sc = get_scorer("token_f1")
    # 部分重叠:0 < f1 < 1
    r = sc.score_one("一只黑猫", "一只白猫", _sample())
    assert 0.0 < r.score < 1.0


def test_token_f1_no_overlap():
    sc = get_scorer("token_f1")
    r = sc.score_one("abc", "xyz", _sample())
    assert r.score == 0.0


def test_token_f1_aggregate():
    sc = get_scorer("token_f1")
    s = _sample()
    results = [
        sc.score_one("猫", "猫", s),       # f1 = 1.0
        sc.score_one("狗", "猫", s),       # f1 = 0.0
    ]
    agg = sc.aggregate(results)
    assert agg["f1"] == 0.5
    assert "precision" in agg and "recall" in agg


def test_token_f1_skipped_when_no_reference():
    sc = get_scorer("token_f1")
    r = sc.score_one("anything", None, _sample())
    assert r.detail.get("skipped") is True
