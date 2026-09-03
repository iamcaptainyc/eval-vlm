"""纯 Python 分类混淆矩阵计算与格式化工具。

适用于 exact_match / 分类评测场景,无需额外依赖 (如 scikit-learn / numpy)。
支持:
  - 类别频次统计、未知预测列归并为 (其他)
  - 混淆矩阵生成 (带行/列合计)
  - 每类 Precision / Recall / F1 / Support 计算与 Macro-Average
  - 终端对齐文本 / Markdown 表格 / HTML 组件格式化输出
"""
from __future__ import annotations

from collections import Counter
import html
from typing import Optional


def compute_confusion_matrix(
    pairs: list[tuple[str, str]],
    max_classes: int = 50,
) -> Optional[dict]:
    """计算分类混淆矩阵。

    Args:
        pairs: [(reference, prediction), ...] 字符串列表
        max_classes: 最大允许的类别数。若真值独立类别数大于此值(如开放式文本),则视为非离散分类任务,返回 None。

    Returns:
        若符合分类任务条件,返回 dict 包含:
          - classes: 列标签列表(含 "(其他)" 若存在未知预测)
          - ref_classes: 行标签列表(全部真值类别)
          - matrix: 2D list[list[int]], matrix[i][j] 为真值 i 预测为 j 的频次
          - per_class: 各类指标 {c: {precision, recall, f1, support}}
          - macro_avg: {precision, recall, f1}
          - accuracy: float
          - total_samples: int
        否则返回 None。
    """
    if not pairs:
        return None

    # 统计真值中出现的所有类别
    clean_pairs = [(str(r).strip(), str(p).strip()) for r, p in pairs if r is not None]
    if not clean_pairs:
        return None

    ref_counts = Counter(r for r, _ in clean_pairs if r)
    unique_refs = sorted(ref_counts.keys())
    if len(unique_refs) < 2 or len(unique_refs) > max_classes:
        return None

    # 检查是否有预测值落在已知真值类别之外
    has_other = any(p not in ref_counts for _, p in clean_pairs)
    classes = list(unique_refs)
    if has_other:
        classes.append("(其他)")

    ref_to_row = {r: i for i, r in enumerate(unique_refs)}
    col_map = {c: j for j, c in enumerate(classes)}

    n_rows = len(unique_refs)
    n_cols = len(classes)
    matrix = [[0] * n_cols for _ in range(n_rows)]

    for r, p in clean_pairs:
        if r not in ref_to_row:
            continue
        row_idx = ref_to_row[r]
        col_idx = col_map.get(p, col_map.get("(其他)"))
        matrix[row_idx][col_idx] += 1

    # 各类别评估指标
    per_class: dict[str, dict[str, float | int]] = {}
    total_tp = 0
    total_samples = len(clean_pairs)

    for i, c in enumerate(unique_refs):
        tp = matrix[i][i]
        total_tp += tp
        support = sum(matrix[i])
        fp = sum(matrix[r][i] for r in range(n_rows)) - tp

        prec = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
        rec = round(tp / support, 4) if support > 0 else 0.0
        f1 = round(2 * prec * rec / (prec + rec), 4) if (prec + rec) > 0 else 0.0

        per_class[c] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support": support,
        }

    macro_prec = round(sum(pc["precision"] for pc in per_class.values()) / n_rows, 4) if n_rows else 0.0
    macro_rec = round(sum(pc["recall"] for pc in per_class.values()) / n_rows, 4) if n_rows else 0.0
    macro_f1 = round(sum(pc["f1"] for pc in per_class.values()) / n_rows, 4) if n_rows else 0.0
    acc = round(total_tp / total_samples, 4) if total_samples > 0 else 0.0

    return {
        "classes": classes,
        "ref_classes": unique_refs,
        "matrix": matrix,
        "per_class": per_class,
        "macro_avg": {
            "precision": macro_prec,
            "recall": macro_rec,
            "f1": macro_f1,
        },
        "accuracy": acc,
        "total_samples": total_samples,
    }


def format_confusion_matrix_text(
    cm_data: dict,
    title: str = "混淆矩阵 (Confusion Matrix)",
) -> str:
    """把混淆矩阵格式化为控制台对齐文本表格。"""
    classes = cm_data["classes"]
    ref_classes = cm_data["ref_classes"]
    matrix = cm_data["matrix"]
    per_class = cm_data.get("per_class", {})
    macro = cm_data.get("macro_avg", {})

    # 计算列宽
    col_w = max(10, max((len(c) for c in classes), default=4) + 2)
    label_w = max(16, max((len(r) for r in ref_classes), default=4) + 2)

    lines = [
        "=" * 78,
        title,
        "=" * 78,
        f"{'真实 \\ 预测':<{label_w}}" + "".join(f"{c:>{col_w}}" for c in classes) + f"{'合计':>{col_w}}",
        "-" * (label_w + col_w * (len(classes) + 1)),
    ]

    col_totals = [0] * len(classes)
    for i, ref_c in enumerate(ref_classes):
        row_vals = matrix[i]
        row_sum = sum(row_vals)
        for j, val in enumerate(row_vals):
            col_totals[j] += val
        row_str = f"{ref_c:<{label_w}}" + "".join(f"{v:>{col_w}}" for v in row_vals) + f"{row_sum:>{col_w}}"
        lines.append(row_str)

    lines.append("-" * (label_w + col_w * (len(classes) + 1)))
    row_tot = f"{'合计':<{label_w}}" + "".join(f"{ct:>{col_w}}" for ct in col_totals) + f"{sum(col_totals):>{col_w}}"
    lines.append(row_tot)
    lines.append("-" * (label_w + col_w * (len(classes) + 1)))

    # 分类性能指标表 (Precision / Recall / F1)
    lines.append("")
    lines.append("分类详细指标 (Classification Report):")
    p_w = 12
    lines.append(f"{'类别':<{label_w}}{'Precision':>{p_w}}{'Recall':>{p_w}}{'F1-score':>{p_w}}{'Support':>{p_w}}")
    lines.append("-" * (label_w + p_w * 4))
    for c in ref_classes:
        stats = per_class.get(c, {})
        p = f"{stats.get('precision', 0.0):.4f}"
        r = f"{stats.get('recall', 0.0):.4f}"
        f1 = f"{stats.get('f1', 0.0):.4f}"
        sup = str(stats.get('support', 0))
        lines.append(f"{c:<{label_w}}{p:>{p_w}}{r:>{p_w}}{f1:>{p_w}}{sup:>{p_w}}")
    lines.append("-" * (label_w + p_w * 4))
    mp = f"{macro.get('precision', 0.0):.4f}"
    mr = f"{macro.get('recall', 0.0):.4f}"
    mf1 = f"{macro.get('f1', 0.0):.4f}"
    lines.append(f"{'Macro Avg':<{label_w}}{mp:>{p_w}}{mr:>{p_w}}{mf1:>{p_w}}{cm_data.get('total_samples', 0):>{p_w}}")
    lines.append(f"{'Accuracy':<{label_w}}{'':>{p_w}}{'':>{p_w}}{cm_data.get('accuracy', 0.0):>{p_w}.4f}{cm_data.get('total_samples', 0):>{p_w}}")
    lines.append("=" * 78)

    return "\n".join(lines)


def format_confusion_matrix_markdown(cm_data: dict) -> str:
    """把混淆矩阵与每类指标格式化为 Markdown 表格。"""
    classes = cm_data["classes"]
    ref_classes = cm_data["ref_classes"]
    matrix = cm_data["matrix"]
    per_class = cm_data.get("per_class", {})
    macro = cm_data.get("macro_avg", {})

    lines = [
        "#### 混淆矩阵 (Confusion Matrix)",
        "",
        "| 真实 \\ 预测 | " + " | ".join(classes) + " | 合计 |",
        "| " + " | ".join(["---"] * (len(classes) + 2)) + " |",
    ]

    col_totals = [0] * len(classes)
    for i, ref_c in enumerate(ref_classes):
        row_vals = matrix[i]
        row_sum = sum(row_vals)
        for j, val in enumerate(row_vals):
            col_totals[j] += val
        row_strs = [str(v) for v in row_vals]
        lines.append(f"| **{ref_c}** | " + " | ".join(row_strs) + f" | {row_sum} |")

    lines.append("| **合计** | " + " | ".join(str(ct) for ct in col_totals) + f" | {sum(col_totals)} |")
    lines.append("")

    # 分类详细指标 Markdown 表格
    lines.append("##### 分类指标详情 (Classification Report)")
    lines.append("")
    lines.append("| 类别 | Precision | Recall | F1-score | Support |")
    lines.append("| --- | --- | --- | --- | --- |")
    for c in ref_classes:
        stats = per_class.get(c, {})
        lines.append(
            f"| **{c}** | {stats.get('precision', 0.0):.4f} | "
            f"{stats.get('recall', 0.0):.4f} | {stats.get('f1', 0.0):.4f} | "
            f"{stats.get('support', 0)} |"
        )
    lines.append(
        f"| **Macro Avg** | {macro.get('precision', 0.0):.4f} | "
        f"{macro.get('recall', 0.0):.4f} | {macro.get('f1', 0.0):.4f} | "
        f"{cm_data.get('total_samples', 0)} |"
    )
    lines.append(
        f"| **Accuracy** | - | - | **{cm_data.get('accuracy', 0.0):.4f}** | "
        f"{cm_data.get('total_samples', 0)} |"
    )
    lines.append("")

    return "\n".join(lines)


def format_confusion_matrix_html(cm_data: dict, title: str = "混淆矩阵 (Confusion Matrix)") -> str:
    """把混淆矩阵渲染为带样式的 HTML 卡片。"""
    classes = cm_data["classes"]
    ref_classes = cm_data["ref_classes"]
    matrix = cm_data["matrix"]
    per_class = cm_data.get("per_class", {})
    macro = cm_data.get("macro_avg", {})

    html_out = [
        '<div class="cm-section">',
        f'<h3>{html.escape(title)}</h3>',
        '<div class="cm-table-wrapper">',
        '<table class="cm-table">',
        '<thead><tr>',
        '<th>真实 \\ 预测</th>',
    ]
    for c in classes:
        html_out.append(f'<th>{html.escape(c)}</th>')
    html_out.append('<th>合计</th></tr></thead><tbody>')

    col_totals = [0] * len(classes)
    for i, ref_c in enumerate(ref_classes):
        row_vals = matrix[i]
        row_sum = sum(row_vals)
        html_out.append(f'<tr><th class="row-label">{html.escape(ref_c)}</th>')
        for j, val in enumerate(row_vals):
            col_totals[j] += val
            # 对角线高亮
            is_diag = (j < len(ref_classes) and classes[j] == ref_c)
            cls = 'class="cm-diag"' if is_diag else ('class="cm-zero"' if val == 0 else '')
            html_out.append(f'<td {cls}>{val}</td>')
        html_out.append(f'<td class="cm-total">{row_sum}</td></tr>')

    html_out.append('</tbody><tfoot><tr><th class="row-label">合计</th>')
    for ct in col_totals:
        html_out.append(f'<td class="cm-total">{ct}</td>')
    html_out.append(f'<td class="cm-total-all">{sum(col_totals)}</td></tr></tfoot></table></div>')

    # 分类详细指标表格
    html_out.append('<div class="cm-report-wrapper"><table class="cm-report-table">')
    html_out.append('<thead><tr><th>类别</th><th>Precision</th><th>Recall</th><th>F1-score</th><th>Support</th></tr></thead><tbody>')
    for c in ref_classes:
        stats = per_class.get(c, {})
        html_out.append(
            f'<tr><td class="cat-name">{html.escape(c)}</td>'
            f'<td>{stats.get("precision", 0.0):.4f}</td>'
            f'<td>{stats.get("recall", 0.0):.4f}</td>'
            f'<td>{stats.get("f1", 0.0):.4f}</td>'
            f'<td>{stats.get("support", 0)}</td></tr>'
        )
    html_out.append('</tbody><tfoot>')
    html_out.append(
        f'<tr><td class="cat-name">Macro Avg</td>'
        f'<td>{macro.get("precision", 0.0):.4f}</td>'
        f'<td>{macro.get("recall", 0.0):.4f}</td>'
        f'<td>{macro.get("f1", 0.0):.4f}</td>'
        f'<td>{cm_data.get("total_samples", 0)}</td></tr>'
    )
    html_out.append(
        f'<tr><td class="cat-name">Accuracy</td>'
        f'<td>-</td><td>-</td><td><strong>{cm_data.get("accuracy", 0.0):.4f}</strong></td>'
        f'<td>{cm_data.get("total_samples", 0)}</td></tr>'
    )
    html_out.append('</tfoot></table></div></div>')

    return "".join(html_out)
