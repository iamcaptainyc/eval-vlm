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


# ---- prefix_match(只比前 k 个字符,逻辑同 exact_match) ----

def test_registry_has_prefix_match():
    assert "prefix_match" in available_scorers()


def test_prefix_match_hit_ignores_trailing():
    """前 k 字符一致即命中,即使整串不同(与 exact_match 的关键区别)。"""
    sc = get_scorer("prefix_match:3")
    r = sc.score_one("红色的车厢", "红色的汽车", _sample())
    assert r.score == 1.0                      # 前3字 "红色的" 相同,后续不同不影响
    assert r.detail["prefix_match"] == 1.0
    assert r.detail["k"] == 3


def test_prefix_match_miss_on_early_diff():
    """前 k 个字符里就有差异 -> 未命中。"""
    sc = get_scorer("prefix_match:3")
    r = sc.score_one("蓝色的车", "红色的车", _sample())
    assert r.score == 0.0
    assert r.detail["prefix_match"] == 0.0


def test_prefix_match_normalizes_like_exact_match():
    """归一化(大小写/标点/空白)后再比:'Red, apple' 与 'red apple' 前 3 字符一致。"""
    sc = get_scorer("prefix_match:3")
    r = sc.score_one("Red, apple", "red apple", _sample())
    assert r.score == 1.0


def test_prefix_match_k_via_name_suffix():
    """名字后缀 prefix_match:K 指定 k,并体现在 detail 里。"""
    sc = get_scorer("prefix_match:4")
    r = sc.score_one("abcdef", "abcdZZ", _sample())
    assert r.score == 1.0                      # 前4字 "abcd" 相同
    assert r.detail["k"] == 4


def test_prefix_match_short_reference_degrades_to_exact():
    """reference 不足 k 个字符 -> 取其全部,退化为整串精确匹配。"""
    sc = get_scorer("prefix_match:10")
    assert sc.score_one("ab", "ab", _sample()).score == 1.0
    assert sc.score_one("abc", "ab", _sample()).score == 0.0   # pred前10="abc" != ref"ab"


def test_prefix_match_default_k():
    """不带后缀的 prefix_match 用默认 k=10。"""
    sc = get_scorer("prefix_match")
    assert sc.k == 10
    r = sc.score_one("x", "x", _sample())
    assert r.score == 1.0 and r.detail["k"] == 10


def test_prefix_match_bad_spec_raises():
    for bad in ("prefix_match:abc", "prefix_match:0", "prefix_match:-3"):
        try:
            get_scorer(bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad} 应报错但未报")


def test_prefix_match_aggregate_accuracy():
    sc = get_scorer("prefix_match:2")
    s = _sample()
    results = [
        sc.score_one("愤怒的表情", "愤怒", s),   # 前2字 "愤怒" == ref"愤怒" -> 命中
        sc.score_one("开心", "愤怒", s),         # 前2字不同 -> 未命中
    ]
    agg = sc.aggregate(results)
    assert agg["accuracy"] == 0.5
    assert agg["prefix_k"] == 2


def test_prefix_match_skipped_when_no_reference():
    sc = get_scorer("prefix_match:5")
    r = sc.score_one("anything", None, _sample())
    assert r.detail.get("skipped") is True


# ---- contain_acc(pred 包含 ref 即命中,不限位置、不要求整串相等) ----

def test_registry_has_contain_acc():
    assert "contain_acc" in available_scorers()


def test_contain_acc_hit_when_substring():
    """ref 作为子串出现在 pred 里(前后有解释)-> 命中,与 exact_match 的关键区别。"""
    sc = get_scorer("contain_acc")
    r = sc.score_one("这张图片表达的情绪是愤怒。", "愤怒", _sample())
    assert r.score == 1.0
    assert r.detail["contain_acc"] == 1.0


def test_contain_acc_miss_when_absent():
    sc = get_scorer("contain_acc")
    r = sc.score_one("这张图片表达的是开心", "愤怒", _sample())
    assert r.score == 0.0


def test_contain_acc_normalizes_before_match():
    """归一化(大小写/标点/空白)后再判子串:'Red apple' 含 'red'。"""
    sc = get_scorer("contain_acc")
    assert sc.score_one("A Red, apple.", "red", _sample()).score == 1.0


def test_contain_acc_empty_reference_not_hit():
    """归一化后 ref 为空串不算命中(否则空串是任何串的子串会误判满分)。"""
    sc = get_scorer("contain_acc")
    assert sc.score_one("anything", "", _sample()).score == 0.0


def test_contain_acc_aggregate_accuracy():
    sc = get_scorer("contain_acc")
    s = _sample()
    results = [
        sc.score_one("答案是第三车道", "第三车道", s),   # 包含 -> 命中
        sc.score_one("第二车道", "第三车道", s),         # 不含 -> 未命中
    ]
    agg = sc.aggregate(results)
    assert agg["accuracy"] == 0.5
    assert agg["num_scored"] == 2


def test_contain_acc_skipped_when_no_reference():
    sc = get_scorer("contain_acc")
    r = sc.score_one("anything", None, _sample())
    assert r.detail.get("skipped") is True


# ---- 混淆矩阵 (Confusion Matrix) ----

def test_compute_confusion_matrix_basic():
    from eval_vlm.scoring.confusion_matrix import compute_confusion_matrix
    pairs = [
        ("car", "car"),
        ("car", "car"),
        ("car", "bus"),
        ("bus", "bus"),
        ("bus", "car"),
        ("bus", "truck"),  # truck 在已知类别中
        ("truck", "truck"),
        ("truck", "bike"),  # bike 不在真值类别中 -> 归入 (其他)
    ]
    cm = compute_confusion_matrix(pairs)
    assert cm is not None
    assert cm["ref_classes"] == ["bus", "car", "truck"]
    assert cm["classes"] == ["bus", "car", "truck", "(其他)"]
    # 行: bus, car, truck
    # bus行: [bus: 1, car: 1, truck: 1, 其他: 0] -> sum = 3
    assert cm["matrix"][0] == [1, 1, 1, 0]
    # car行: [bus: 1, car: 2, truck: 0, 其他: 0] -> sum = 3
    assert cm["matrix"][1] == [1, 2, 0, 0]
    # truck行: [bus: 0, car: 0, truck: 1, 其他: 1] -> sum = 2
    assert cm["matrix"][2] == [0, 0, 1, 1]

    # 指标验证
    assert cm["per_class"]["car"]["support"] == 3
    assert cm["per_class"]["car"]["recall"] == round(2 / 3, 4)
    assert cm["per_class"]["car"]["precision"] == round(2 / 3, 4)  # 预测car的有 bus(1) + car(2) = 3
    assert cm["per_class"]["truck"]["support"] == 2
    assert cm["per_class"]["truck"]["recall"] == 0.5  # 1/2
    assert cm["accuracy"] == 0.5  # 4/8


def test_confusion_matrix_thresholds():
    from eval_vlm.scoring.confusion_matrix import compute_confusion_matrix
    # 类别数 < 2 -> None
    assert compute_confusion_matrix([("single", "single")]) is None
    assert compute_confusion_matrix([]) is None

    # 类别数 > max_classes -> None
    many_pairs = [(f"class_{i}", f"class_{i}") for i in range(55)]
    assert compute_confusion_matrix(many_pairs, max_classes=50) is None


def test_confusion_matrix_formatters():
    from eval_vlm.scoring.confusion_matrix import (
        compute_confusion_matrix,
        format_confusion_matrix_text,
        format_confusion_matrix_markdown,
        format_confusion_matrix_html,
    )
    pairs = [("A", "A"), ("A", "B"), ("B", "B"), ("B", "C")]
    cm = compute_confusion_matrix(pairs)
    assert cm is not None

    text = format_confusion_matrix_text(cm)
    assert "真实 \\ 预测" in text
    assert "Macro Avg" in text
    assert "Classification Report" in text

    md = format_confusion_matrix_markdown(cm)
    assert "#### 混淆矩阵 (Confusion Matrix)" in md
    assert "| 真实 \\ 预测 |" in md
    assert "##### 分类指标详情" in md

    html_str = format_confusion_matrix_html(cm)
    assert '<table class="cm-table">' in html_str
    assert 'class="cm-diag"' in html_str
    assert '<table class="cm-report-table">' in html_str


def test_exact_match_scorer_generates_confusion_matrix():
    sc = get_scorer("exact_match")
    s = _sample()
    results = [
        sc.score_one("A", "A", s),
        sc.score_one("B", "A", s),
        sc.score_one("B", "B", s),
    ]
    agg = sc.aggregate(results)
    assert "confusion_matrix" in agg
    cm = agg["confusion_matrix"]
    assert cm["ref_classes"] == ["a", "b"]
    assert cm["accuracy"] == round(2 / 3, 4)


