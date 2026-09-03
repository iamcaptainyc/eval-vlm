"""tune.py 纯逻辑测试(不跑训练/评测子进程)。

共享 db / 命名 / leaderboard 列的改动验证:
  - trial 命名 = 基座名_study名_trial_序号_时间(全 [0-9A-Za-z_],eval-vlm 幂等)
  - 共享 db:同一 sqlite 容纳多个 study,签名防护警告
  - leaderboard 固定列含 study/model/started_at
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))     # 使 tune.py 可被 import
import tune


# ---------------------------------------------------------------------------
# safe_name / 命名
# ---------------------------------------------------------------------------
def test_safe_name_folds_special_chars():
    assert tune.safe_name("Qwen3.5-0.8B") == "Qwen3_5_0_8B"
    assert tune.safe_name("  ") == "model"
    assert tune.safe_name("A/B:c") == "A_B_c"


def test_trial_time_formats_local_and_safe():
    class FakeTrial:
        datetime_start = datetime(2026, 9, 1, 14, 30, 25, tzinfo=timezone.utc)

    ts = tune._trial_time(FakeTrial())
    # 全 [0-9A-Za-z_],14 位数字 + '_' 分隔
    assert len(ts) == 15 and ts.isalnum() is False and "_" in ts
    assert ts.startswith("20260901_")               # UTC 转本地后仍是同一天(本地=UTC+8 以内)
    assert tune.safe_name(ts) == ts                 # 与 safe_name 幂等

    # datetime_start 缺失回退 now,格式一致
    ts2 = tune._trial_time(object())
    assert len(ts2) == 15


def test_trial_artifact_tag_matches_expected_format():
    tag = tune.trial_artifact_tag("Qwen3_5_0_8B", "4dimroad_v2", 7, "20260901_143025")
    assert tag == "Qwen3_5_0_8B_4dimroad_v2_trial_0007_20260901_143025"
    # 全安全字符(eval-vlm safe_model_dirname 不再改写)
    assert tune.safe_name(tag) == tag
    # 序号 4 位补零、时间在尾
    assert tune.trial_artifact_tag("m", "s", 123, "t").endswith("_trial_0123_t")


# ---------------------------------------------------------------------------
# leaderboard 列
# ---------------------------------------------------------------------------
def test_leaderboard_has_study_model_started_at():
    assert tune._LEADERBOARD_FIELDS[:3] == ["study", "model", "started_at"]
    assert "trial" in tune._LEADERBOARD_FIELDS and "objective" in tune._LEADERBOARD_FIELDS


def test_append_leaderboard_writes_study_columns(tmp_path):
    p = tmp_path / "leaderboard.csv"
    tune.append_leaderboard(p, {"study": "s1", "model": "m", "started_at": "t",
                                "trial": 0, "objective": 0.5})
    txt = p.read_text(encoding="utf-8")
    head = txt.splitlines()[0]
    assert head == ",".join(tune._LEADERBOARD_FIELDS)
    assert "s1" in txt and "m" in txt


# ---------------------------------------------------------------------------
# 共享 db:create_or_load_study 的签名防护 + 多 study 共存
# ---------------------------------------------------------------------------
def _cfg(tmp_path: Path, study_name: str, base_model: str = "/m/Qwen3.5-0.8B",
         dataset: str = "/d/ds") -> dict:
    return {"optuna": {"study_name": study_name},
            "base_model": base_model, "eval_vlm_dataset": dataset}


def _storage(tmp_path: Path) -> str:
    return f"sqlite:///{(tmp_path / 'study.db').as_posix()}"


def test_shared_db_two_studies_coexist(tmp_path):
    storage = _storage(tmp_path)
    s1 = tune.create_or_load_study(_cfg(tmp_path, "study_a"), storage, None, "maximize")
    s2 = tune.create_or_load_study(_cfg(tmp_path, "study_b"), storage, None, "maximize")
    assert s1.study_name == "study_a" and s2.study_name == "study_b"
    names = {s.study_name for s in tune.optuna.get_all_study_summaries(storage)}
    assert names == {"study_a", "study_b"}


def test_signature_guard_warns_on_different_experiment(tmp_path, capsys):
    storage = _storage(tmp_path)
    tune.create_or_load_study(_cfg(tmp_path, "exp1", base_model="/m/A"), storage, None, "maximize")
    # 同 study 名、不同 base_model -> 大声警告
    tune.create_or_load_study(_cfg(tmp_path, "exp1", base_model="/m/B"), storage, None, "maximize")
    out = capsys.readouterr().out
    assert "⚠️ 警告" in out and "exp1" in out
    # 同 study 名、同 base_model 续跑 -> 不警告
    capsys.readouterr()
    tune.create_or_load_study(_cfg(tmp_path, "exp1", base_model="/m/A"), storage, None, "maximize")
    assert "⚠️ 警告" not in capsys.readouterr().out
