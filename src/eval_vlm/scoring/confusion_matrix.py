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
    """把混淆矩阵渲染为学术论文级精美热力图卡片 (支持颜色深浅变化、百分比/数值模式切换与指标条形图)。"""
    classes = cm_data["classes"]
    ref_classes = cm_data["ref_classes"]
    matrix = cm_data["matrix"]
    per_class = cm_data.get("per_class", {})
    macro = cm_data.get("macro_avg", {})
    total_samples = cm_data.get("total_samples", 0)
    acc = cm_data.get("accuracy", 0.0)

    max_val = max((max(row) for row in matrix), default=1)
    if max_val == 0:
        max_val = 1

    # 列合计
    col_totals = [0] * len(classes)
    for row in matrix:
        for j, v in enumerate(row):
            col_totals[j] += v

    html_out = [
        '<div class="cm-section">',
        '<style>',
        '.cm-section { background: #ffffff; border: 1px solid #d0d7de; border-radius: 12px; padding: 20px 24px; margin-bottom: 24px; box-shadow: 0 2px 10px rgba(0,0,0,0.04); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; }',
        '.cm-header-row { display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px; margin-bottom: 16px; padding-bottom: 12px; border-bottom: 1px solid #e1e4e8; }',
        '.cm-header-row h3 { margin: 0; font-size: 17px; font-weight: 700; color: #1f2937; display: flex; align-items: center; gap: 8px; }',
        '.cm-header-meta { font-size: 13px; color: #64748b; }',
        '.cm-header-meta strong { color: #0969da; }',
        '.cm-toolbar { display: flex; align-items: center; gap: 14px; flex-wrap: wrap; margin-bottom: 14px; background: #f8fafc; padding: 10px 14px; border-radius: 10px; border: 1px solid #e2e8f0; }',
        '.cm-tool-item { display: flex; align-items: center; gap: 8px; }',
        '.cm-tool-title { font-size: 12.5px; font-weight: 700; color: #334155; }',
        '.cm-pill-group { display: inline-flex; border-radius: 20px; background: #e2e8f0; padding: 3px; gap: 2px; }',
        '.cm-pill-btn { border: none; background: transparent; padding: 5px 12px; font-size: 12px; font-weight: 600; color: #475569; border-radius: 16px; cursor: pointer; transition: all .18s ease; display: inline-flex; align-items: center; gap: 4px; }',
        '.cm-pill-btn:hover { color: #0f172a; background: rgba(255,255,255,0.6); }',
        '.cm-pill-btn.active { background: #2563eb; color: #ffffff; box-shadow: 0 2px 6px rgba(37,99,235,0.28); }',
        '.cm-colorbar-wrap { display: flex; align-items: center; gap: 8px; font-size: 11px; color: #64748b; margin-left: auto; }',
        '.cm-cb-gradient { width: 95px; height: 12px; border-radius: 3px; background: linear-gradient(to right, #f8fafc 0%, hsl(215, 85%, 85%) 25%, hsl(215, 85%, 55%) 70%, hsl(215, 85%, 35%) 100%); border: 1px solid #cbd5e1; }',
        '.cm-layout { display: flex; gap: 24px; flex-wrap: wrap; align-items: flex-start; }',
        '.cm-matrix-container { flex: 1 1 540px; min-width: 320px; }',
        '.cm-axis-top { text-align: center; font-size: 12px; font-weight: 700; letter-spacing: 0.5px; color: #475569; margin-bottom: 6px; }',
        '.cm-table-wrapper { overflow-x: auto; border-radius: 8px; border: 1px solid #d0d7de; }',
        '.cm-table { border-collapse: separate; border-spacing: 2px; width: 100%; font-size: 13px; background: #f1f5f9; }',
        '.cm-table th, .cm-table td { padding: 8px 10px; text-align: center; border-radius: 4px; }',
        '.cm-table thead th { background: #ffffff; color: #1e293b; font-weight: 600; border-bottom: 2px solid #cbd5e1; }',
        '.cm-table th.row-label { background: #ffffff; color: #1e293b; font-weight: 600; text-align: right; border-right: 2px solid #cbd5e1; white-space: nowrap; }',
        '.cm-cell, .cm-diag { position: relative; min-width: 50px; min-height: 48px; cursor: pointer; transition: transform .1s, box-shadow .1s; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }',
        '.cm-cell:hover, .cm-diag:hover { transform: scale(1.08); z-index: 5; box-shadow: 0 0 0 2px #0f172a, 0 4px 12px rgba(0,0,0,0.18); }',
        '.cm-v-count { display: block; font-size: 14px; font-weight: 700; line-height: 1.2; }',
        '.cm-v-pct { display: block; font-size: 10.5px; opacity: 0.9; line-height: 1.2; margin-top: 2px; }',
        '.cm-mode-count-only .cm-v-pct { display: none !important; }',
        '.cm-mode-count-only .cm-v-count { font-size: 15px; }',
        '.cm-mode-pct-only .cm-v-count { display: none !important; }',
        '.cm-mode-pct-only .cm-v-pct { font-size: 13.5px; font-weight: 700; opacity: 1; margin-top: 0; }',
        '.cm-total { background: #e2e8f0; color: #334155; font-weight: 600; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }',
        '.cm-total-all { background: #cbd5e1; color: #0f172a; font-weight: 700; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }',
        '.cm-report-container { flex: 1 1 380px; min-width: 320px; }',
        '.cm-report-table { width: 100%; border-collapse: collapse; font-size: 13px; background: #ffffff; border-radius: 8px; overflow: hidden; border: 1px solid #e2e8f0; }',
        '.cm-report-table th, .cm-report-table td { border-bottom: 1px solid #e2e8f0; padding: 8px 12px; text-align: right; }',
        '.cm-report-table th { background: #f8fafc; color: #475569; font-weight: 600; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; }',
        '.cm-report-table td.cat-name, .cm-report-table th.cat-name { text-align: left; font-weight: 600; color: #1e293b; }',
        '.cm-report-table tfoot td { background: #f8fafc; font-weight: 600; }',
        '.cm-bar-wrap { display: flex; align-items: center; justify-content: flex-end; gap: 8px; }',
        '.cm-bar { width: 42px; height: 6px; background: #e2e8f0; border-radius: 3px; overflow: hidden; }',
        '.cm-bar-fill { height: 100%; background: #2563eb; border-radius: 3px; }',
        '</style>',
        '<div class="cm-header-row">',
        f'<h3>📊 {html.escape(title)}</h3>',
        f'<div class="cm-header-meta">样本总数: <strong>{total_samples}</strong> &nbsp;|&nbsp; 总体准确率: <strong>{acc * 100:.2f}%</strong></div>',
        '</div>',
        '<div class="cm-toolbar">',
        '<div class="cm-tool-item">',
        '<span class="cm-tool-title">显示切换:</span>',
        '<div class="cm-pill-group cm-display-mode">',
        '<button type="button" class="cm-pill-btn active" data-mode="both">🔢+％ 数量与占比</button>',
        '<button type="button" class="cm-pill-btn" data-mode="count">🔢 仅看数值</button>',
        '<button type="button" class="cm-pill-btn" data-mode="pct">％ 仅看百分比</button>',
        '</div>',
        '</div>',
        '<div class="cm-tool-item">',
        '<span class="cm-tool-title">热力基准:</span>',
        '<div class="cm-pill-group cm-heat-mode">',
        '<button type="button" class="cm-pill-btn active" data-heat="pct">行归一化 % (召回率)</button>',
        '<button type="button" class="cm-pill-btn" data-heat="count">绝对样本数</button>',
        '</div>',
        '</div>',
        '<div class="cm-colorbar-wrap">',
        '<span>0% (低)</span>',
        '<div class="cm-cb-gradient"></div>',
        '<span>100% (高)</span>',
        '</div>',
        '</div>',
        '<div class="cm-layout">',
        '<div class="cm-matrix-container">',
        '<div class="cm-axis-top">预测类别 (Predicted Class) ➔</div>',
        '<div class="cm-table-wrapper">',
        '<table class="cm-table">',
        '<thead><tr>',
        '<th style="min-width:70px;">真实 \\ 预测</th>',
    ]

    for c in classes:
        html_out.append(f'<th>{html.escape(c)}</th>')
    html_out.append('<th style="background:#e2e8f0; color:#334155;">合计 (Support)</th></tr></thead><tbody>')

    for i, ref_c in enumerate(ref_classes):
        row_vals = matrix[i]
        row_sum = sum(row_vals)
        html_out.append(f'<tr><th class="row-label">{html.escape(ref_c)}</th>')
        for j, val in enumerate(row_vals):
            is_diag = (j < len(ref_classes) and classes[j] == ref_c)
            pct = (val / row_sum * 100) if row_sum > 0 else 0.0
            col_pct = (val / col_totals[j] * 100) if col_totals[j] > 0 else 0.0
            norm_pct = (val / row_sum) if row_sum > 0 else 0.0
            norm_cnt = (val / max_val) if max_val > 0 else 0.0

            # 默认行归一化色彩
            if val == 0:
                bg = "#f8fafc"
                fg = "#94a3b8"
            else:
                lightness = round(95 - norm_pct * 59)
                bg = f"hsl(215, 85%, {lightness}%)"
                fg = "#ffffff" if lightness < 64 else "#0f172a"

            cell_cls = "cm-diag" if is_diag else ("cm-zero" if val == 0 else "cm-cell")
            pred_name = classes[j]
            tooltip = (
                f"真实: {ref_c} &#10;"
                f"预测: {pred_name} &#10;"
                f"样本数: {val} &#10;"
                f"占该真值行 (Recall): {pct:.1f}% &#10;"
                f"占该预测列 (Precision): {col_pct:.1f}%"
            )

            html_out.append(
                f'<td class="{cell_cls}" '
                f'style="background-color: {bg}; color: {fg};" '
                f'data-val="{val}" data-pct="{pct:.1f}%" '
                f'data-norm-pct="{norm_pct:.3f}" data-norm-cnt="{norm_cnt:.3f}" '
                f'title="{tooltip}">'
                f'<span class="cm-v-count">{val}</span>'
                f'<span class="cm-v-pct">{pct:.1f}%</span>'
                f'</td>'
            )
        html_out.append(f'<td class="cm-total">{row_sum}</td></tr>')

    html_out.append('</tbody><tfoot><tr><th class="row-label">合计 (Pred)</th>')
    for ct in col_totals:
        html_out.append(f'<td class="cm-total">{ct}</td>')
    html_out.append(f'<td class="cm-total-all">{sum(col_totals)}</td></tr></tfoot></table></div></div>')

    # 分类详细指标表格
    html_out.append('<div class="cm-report-container">')
    html_out.append('<div style="font-size:12px; font-weight:700; color:#475569; margin-bottom:6px; letter-spacing:0.5px;">分类性能指标 (Classification Report)</div>')
    html_out.append('<table class="cm-report-table">')
    html_out.append('<thead><tr><th class="cat-name">类别</th><th>Precision</th><th>Recall</th><th>F1-score</th><th>Support</th></tr></thead><tbody>')
    for c in ref_classes:
        stats = per_class.get(c, {})
        p = stats.get("precision", 0.0)
        r = stats.get("recall", 0.0)
        f1 = stats.get("f1", 0.0)
        sup = stats.get("support", 0)
        html_out.append(
            f'<tr><td class="cat-name">{html.escape(c)}</td>'
            f'<td><div class="cm-bar-wrap"><div class="cm-bar"><div class="cm-bar-fill" style="width:{p*100:.1f}%;"></div></div><span>{p:.4f}</span></div></td>'
            f'<td><div class="cm-bar-wrap"><div class="cm-bar"><div class="cm-bar-fill" style="width:{r*100:.1f}%;"></div></div><span>{r:.4f}</span></div></td>'
            f'<td><div class="cm-bar-wrap"><div class="cm-bar"><div class="cm-bar-fill" style="width:{f1*100:.1f}%;"></div></div><span>{f1:.4f}</span></div></td>'
            f'<td>{sup}</td></tr>'
        )
    html_out.append('</tbody><tfoot>')
    mp = macro.get("precision", 0.0)
    mr = macro.get("recall", 0.0)
    mf1 = macro.get("f1", 0.0)
    html_out.append(
        f'<tr><td class="cat-name">Macro Avg</td>'
        f'<td><div class="cm-bar-wrap"><div class="cm-bar"><div class="cm-bar-fill" style="width:{mp*100:.1f}%;"></div></div><span>{mp:.4f}</span></div></td>'
        f'<td><div class="cm-bar-wrap"><div class="cm-bar"><div class="cm-bar-fill" style="width:{mr*100:.1f}%;"></div></div><span>{mr:.4f}</span></div></td>'
        f'<td><div class="cm-bar-wrap"><div class="cm-bar"><div class="cm-bar-fill" style="width:{mf1*100:.1f}%;"></div></div><span>{mf1:.4f}</span></div></td>'
        f'<td>{total_samples}</td></tr>'
    )
    html_out.append(
        f'<tr><td class="cat-name">Accuracy</td>'
        f'<td colspan="2">-</td>'
        f'<td><div class="cm-bar-wrap"><div class="cm-bar"><div class="cm-bar-fill" style="width:{acc*100:.1f}%; background:#10b981;"></div></div><span style="font-weight:700; color:#059669;">{acc:.4f}</span></div></td>'
        f'<td>{total_samples}</td></tr>'
    )
    html_out.append('</tfoot></table></div></div>')

    # 内置交互脚本 (作用于每个 .cm-section, 互不干扰)
    html_out.append("""<script>
(function() {
  document.querySelectorAll('.cm-section').forEach(function(sec) {
    var table = sec.querySelector('.cm-table');
    if (!table) return;

    // 切换数值/百分比显示模式
    var dispBtns = sec.querySelectorAll('.cm-display-mode .cm-pill-btn');
    dispBtns.forEach(function(btn) {
      btn.addEventListener('click', function() {
        dispBtns.forEach(function(b) { b.classList.remove('active'); });
        btn.classList.add('active');
        var mode = btn.getAttribute('data-mode');
        table.classList.remove('cm-mode-count-only', 'cm-mode-pct-only');
        if (mode === 'count') {
          table.classList.add('cm-mode-count-only');
        } else if (mode === 'pct') {
          table.classList.add('cm-mode-pct-only');
        }
      });
    });

    // 切换热力基准: pct (行归一化) vs count (绝对数量)
    var heatBtns = sec.querySelectorAll('.cm-heat-mode .cm-pill-btn');
    heatBtns.forEach(function(btn) {
      btn.addEventListener('click', function() {
        heatBtns.forEach(function(b) { b.classList.remove('active'); });
        btn.classList.add('active');
        var heat = btn.getAttribute('data-heat');
        sec.querySelectorAll('.cm-cell, .cm-diag, .cm-zero').forEach(function(cell) {
          var val = parseFloat(cell.getAttribute('data-val') || 0);
          if (val === 0) {
            cell.style.backgroundColor = '#f8fafc';
            cell.style.color = '#94a3b8';
            return;
          }
          var norm = parseFloat(cell.getAttribute(heat === 'count' ? 'data-norm-cnt' : 'data-norm-pct') || 0);
          var lightness = Math.round(95 - norm * 59);
          cell.style.backgroundColor = 'hsl(215, 85%, ' + lightness + '%)';
          cell.style.color = (lightness < 64) ? '#ffffff' : '#0f172a';
        });
      });
    });
  });
})();
</script></div>""")

    return "".join(html_out)

    return "".join(html_out)
