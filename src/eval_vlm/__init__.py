"""eval_vlm — 解耦的 VLM 测试集评测工具。

三步独立 CLI,通过文件产物交接:
    split  -> test_split.json
    run    -> predictions.jsonl
    score  -> metrics.json / scored.jsonl / summary.md
"""

__version__ = "0.1.0"
