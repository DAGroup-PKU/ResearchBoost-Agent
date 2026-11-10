"""Regenerate the README result figures (English and Chinese):

- docs/figures/results-overview{,.zh-CN}.svg: every project, ranked by speedup
- docs/figures/results-time{,.zh-CN}.svg: baseline vs optimized time for selected projects

Edit PROJECTS (keep it sorted by speedup) or SHOWCASE when a result is added or changes, then run:

    python3 docs/figures/make_results_overview.py
"""
from pathlib import Path

FIGS = Path(__file__).resolve().parent

# Categorical domain colors, validated for color-vision deficiency on a light surface; every row also names its domain.
DOMAINS = {
    "llm": ("#2a78d6", "LLM", "语言模型"),
    "mllm": ("#1baf7a", "MLLM", "多模态模型"),
    "img": ("#eb6834", "Image generation", "图像生成"),
    "vid": ("#eda100", "Video generation", "视频生成"),
    "emb": ("#6250d6", "Embodied AI", "具身智能"),
    "sci": ("#e87ba4", "AI for Science", "AI for Science"),
    "3d": ("#333333", "3D reconstruction", "三维重建"),
}

# (name, domain, speedup that sets the bar length (a (low, high) pair for a range), label, extrapolated estimate?, validation)
PROJECTS = [
    ("Motus", "emb", 9.59, "9.59×", False, "bitwise"),
    ("RAE", "img", 5.32, "5.32×", False, "stat"),
    ("RoboTwin", "emb", 4.38, "4.38×", True, "bitwise"),
    ("PIDG (module)", "3d", 4.1749, "4.17×", False, "task"),
    ("FastWAM", "emb", 3.46, "3.46×", False, "task"),
    ("LLaVA-OneVision-2", "mllm", (2.2, 3.0), "2.2–3.0×", False, "stat"),
    ("OpenWAM", "emb", (2.54, 2.77), "2.54–2.77×", False, "task"),
    ("LLaVA", "mllm", 2.39, "2.39×", False, "stat"),
    ("ProteinMPNN", "sci", 2.25, "2.25×", False, "explore"),
    ("YOCO", "llm", 2.12, "2.12×", False, "stat"),
    ("Self-Forcing", "vid", 2.08, "2.08×", False, "stat"),
    ("R3", "mllm", 2.08, "2.08×", False, "bitwise"),
    ("MiniMax-H3", "vid", 1.857, "1.86×", False, "stat"),
    ("BAGEL", "mllm", 1.82, "1.82×", False, "stat"),
    ("DMD2", "img", 1.72, "1.72×", False, "stat"),
    ("LLM2Vec", "llm", 1.6009, "1.60×", False, "stat"),
    ("JiT", "img", 1.56, "1.56×", False, "stat"),
    ("Protenix", "sci", 1.5524, "1.55×", False, "manual"),
    ("FermiNet", "sci", 1.252, "1.25×", False, "bitwise"),
    ("3DGS", "3d", 1.1314, "1.13×", False, "task"),
    ("NanoChat", "llm", 1.0744, "1.07×", False, "stat"),
]

# Projects contributed by the community; the leaderboard marks them with a people icon.
COMMUNITY = {"NanoChat", "LLM2Vec", "MiniMax-H3", "OpenWAM", "LLaVA-OneVision-2", "ProteinMPNN", "Protenix", "FermiNet", "PIDG (module)", "3DGS"}

# (name, domain, baseline, optimized, unit-formatted labels, speedup label, estimate?, en/zh scope, en/zh quality)
SHOWCASE = [
    ("Motus", "emb", 7.468, 0.779, ("7.47 s", "0.78 s"), "9.59×", False,
     ("Fine-tuning · per iteration", "微调 · 每次迭代"),
     ("Bitwise identical over 1,100 iterations", "1,100 次迭代逐位一致")),
    ("RoboTwin", "emb", 73.5, 16.8, ("73.5 h", "16.8 h"), "4.38×", True,
     ("π0.5 policy evaluation · full benchmark", "π0.5 策略评测 · 完整测试集"),
     ("Compared episodes bitwise identical", "对比的 episode 逐位一致")),
    ("FastWAM", "emb", 1541.4, 445.7, ("1,541 ms", "446 ms"), "3.46×", False,
     ("LIBERO training recipe · per iteration", "LIBERO 训练配方 · 每次迭代"),
     ("LIBERO success 96.90% vs 97.15% released", "LIBERO 成功率 96.90%，官方 97.15%")),
    ("R3", "mllm", 24.72, 11.87, ("24.7 s", "11.9 s"), "2.08×", False,
     ("BAGEL Tree-RL · per RL step", "BAGEL Tree-RL · 每个 RL step"),
     ("Bitwise identical, 14 steps on 8 ranks", "8 卡 14 步逐位一致")),
    ("YOCO", "llm", 5345, 2799, ("89 min", "47 min"), "1.91×", False,
     ("3B pretraining · 20k-update run", "3B 预训练 · 20,000 次更新端到端"),
     ("Statistical equivalence", "统计等价")),
    ("DMD2", "img", 16.39, 9.79, ("16.4 h", "9.8 h"), "1.67×", False,
     ("One-step distillation · 2,000 iterations", "一步蒸馏 · 2,000 次迭代"),
     ("FID-30k 8.70 vs 8.65 baseline", "FID-30k 8.70，基线 8.65")),
]

TEXT = {
    "en": {
        "title": "Speedup over each project's own baseline",
        "sub": "{n} projects · {d} research domains · higher is faster",
        "cols": ("#", "Project", "Speedup", "Validation"),
        "valid": {"bitwise": "Bitwise", "stat": "Statistical", "task": "Task metric", "explore": "Exploratory", "manual": "Manual review"},
        "est": "est.",
        "baseline": "1× baseline",
        "community": "Community case",
        "foot": "Hatched: extrapolated estimate · LLM2Vec: SimCSE stage (MNTP 1.29×) · details in docs/results",
        "scope_foot": "PIDG: module group including PNG output; not complete-training time.",
        "t_title": "Same work, a fraction of the time",
        "t_sub": "Baseline vs optimized, from the project reports",
        "t_base": "baseline",
        "t_opt": "optimized",
        "t_foot": "YOCO and DMD2 rows are longer end-to-end runs, so their speedup differs from the headline number.",
    },
    "zh": {
        "title": "加速比（均相对各项目自身的基线）",
        "sub": "{n} 个项目 · {d} 个研究领域 · 越长越快",
        "cols": ("#", "项目", "加速比", "验证方式"),
        "valid": {"bitwise": "逐位一致", "stat": "统计等价", "task": "任务指标", "explore": "探索性", "manual": "人工复核"},
        "est": "估计",
        "baseline": "1× 基线",
        "community": "社区案例",
        "foot": "斜纹：外推估计值 · LLM2Vec：SimCSE 阶段（MNTP 为 1.29×） · 详见 docs/results",
        "scope_foot": "PIDG：包含 PNG 写出的模块组耗时，不是完整训练耗时。",
        "t_title": "同样的工作，只用几分之一的时间",
        "t_sub": "基线与优化后的耗时，数据来自各项目报告",
        "t_base": "基线",
        "t_opt": "优化后",
        "t_foot": "YOCO 与 DMD2 两行是更长的端到端运行，因此加速比与榜单数字不同。",
    },
}

INK, INK2, MUTED = "#241B1C", "#4E4244", "#867678"
ACCENT, GRAY_BAR = "#8C0000", "#D9CFCF"
GRID, STRIPE, SURFACE, BORDER = "#F1EAEA", "#FAF6F6", "#FFFFFF", "#EEE4E4"
FONT = ('-apple-system,BlinkMacSystemFont,"Segoe UI","Helvetica Neue",Helvetica,Arial,'
        '"PingFang SC","Hiragino Sans GB","Microsoft YaHei","Noto Sans CJK SC",sans-serif')
W = 1000


def bar_path(x0, y, x1, h, r=4):
    # square at the baseline, rounded at the data end
    r = min(r, (x1 - x0) / 2)
    return (f"M{x0:.1f} {y:.1f}H{x1 - r:.1f}Q{x1:.1f} {y:.1f} {x1:.1f} {y + r:.1f}"
            f"V{y + h - r:.1f}Q{x1:.1f} {y + h:.1f} {x1 - r:.1f} {y + h:.1f}H{x0:.1f}Z")


def head(H, title, desc, extra_defs=""):
    return [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-labelledby="t d">',
        f'<title id="t">{title}</title>',
        f'<desc id="d">{desc}</desc>',
        '<defs>' + extra_defs +
        f'<style>text{{font-family:{FONT}}}'
        f'.title{{font-size:26px;font-weight:700;fill:{INK}}}.sub{{font-size:15px;fill:{MUTED}}}'
        f'.col{{font-size:12.5px;font-weight:600;fill:{MUTED};letter-spacing:.04em}}'
        f'.rank{{font-size:13px;fill:{MUTED};font-variant-numeric:tabular-nums}}'
        f'.name{{font-size:15.5px;font-weight:600;fill:{INK}}}.dom{{font-size:13px;fill:{MUTED}}}'
        f'.val{{font-size:15.5px;font-weight:700;fill:{INK};font-variant-numeric:tabular-nums}}'
        f'.chip{{font-size:12.5px;font-weight:600}}.leg{{font-size:13px;fill:{INK2}}}'
        f'.tick{{font-size:12px;fill:{MUTED};font-variant-numeric:tabular-nums}}.foot{{font-size:12.5px;fill:{MUTED}}}'
        f'.big{{font-size:26px;font-weight:800;fill:{INK};font-variant-numeric:tabular-nums}}'
        f'.tl{{font-size:13px;fill:{INK2};font-variant-numeric:tabular-nums}}.tlb{{font-size:13px;font-weight:700;fill:{INK};font-variant-numeric:tabular-nums}}'
        f'.q{{font-size:13px;fill:{INK2}}}</style></defs>',
        f'<rect x="0.5" y="0.5" width="{W - 1}" height="{H - 1}" rx="16" fill="{SURFACE}" stroke="{BORDER}"/>',
    ]


def hatch(pid, color):
    return (f'<pattern id="{pid}" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
            f'<rect width="6" height="6" fill="{color}" fill-opacity=".28"/><rect width="2.6" height="6" fill="{color}"/></pattern>')


def people(x, cy, color):
    # two people, 14 × 12, left edge at x, centered on cy
    return (f'<g fill="{color}"><g fill-opacity=".5"><circle cx="{x + 9.6:.1f}" cy="{cy - 2.8:.1f}" r="2.1"/>'
            f'<path d="M{x + 6.4:.1f} {cy + 5.5:.1f}Q{x + 6.8:.1f} {cy + 0.6:.1f} {x + 9.6:.1f} {cy + 0.6:.1f}'
            f'Q{x + 12.6:.1f} {cy + 0.6:.1f} {x + 13.2:.1f} {cy + 5.5:.1f}Z"/></g>'
            f'<circle cx="{x + 5:.1f}" cy="{cy - 2.1:.1f}" r="2.5"/>'
            f'<path d="M{x + 0.6:.1f} {cy + 6:.1f}Q{x + 1:.1f} {cy + 1.4:.1f} {x + 5:.1f} {cy + 1.4:.1f}'
            f'Q{x + 9:.1f} {cy + 1.4:.1f} {x + 9.4:.1f} {cy + 6:.1f}Z"/></g>')


def leaderboard(lang):
    t = TEXT[lang]
    li = 1 if lang == "en" else 2
    n = len(PROJECTS)
    PAD, TOP, ROW_H, BAR_H = 36, 172, 32, 14
    X_RANK, X_NAME, X0, PLOT_W, X_VAL, X_CHIP = PAD + 14, PAD + 46, 290, 390, 776, 808
    sx = lambda v: X0 + v / 10.0 * PLOT_W
    # Include community provenance in the same wrapping legend as the domains.
    text_width = lambda label: sum(13.5 if ord(ch) > 0x2E80 else 7.1 for ch in label)
    legend, x, legend_y = [], PAD, 106
    for color, en, zh in DOMAINS.values():
        label = en if lang == "en" else zh
        width = 18 + text_width(label) + 24
        if x > PAD and x + width > W - PAD:
            x, legend_y = PAD, legend_y + 24
        legend.append((x, legend_y, color, label))
        x += width
    if x + 30 + text_width(t["community"]) > W - PAD:
        x, legend_y = PAD, legend_y + 24
    community_x, community_y = x, legend_y
    TOP += legend_y - 106
    bottom = TOP + n * ROW_H
    H = bottom + 90
    defs = "".join(hatch(f"h-{k}", c) for k, (c, *_rest) in DOMAINS.items())
    desc = "; ".join(f"{p[0]} ({DOMAINS[p[1]][li]}) {p[3]}{' ' + t['est'] if p[4] else ''}, {t['valid'][p[5]]}"
                     f"{', ' + t['community'] if p[0] in COMMUNITY else ''}" for p in PROJECTS)
    o = head(H, t["title"], desc, defs)
    a = o.append
    a(f'<text x="{PAD}" y="56" class="title">{t["title"]}</text>')
    a(f'<text x="{PAD}" y="84" class="sub">{t["sub"].format(n=n, d=len({p[1] for p in PROJECTS}))}</text>')
    # legend
    for x, y, color, label in legend:
        a(f'<rect x="{x}" y="{y}" width="12" height="12" rx="3" fill="{color}"/>')
        a(f'<text x="{x + 18}" y="{y + 11}" class="leg">{label}</text>')
    if community_x > PAD:
        a(f'<line x1="{community_x - 6:.1f}" y1="{community_y - 2}" x2="{community_x - 6:.1f}" y2="{community_y + 14}" stroke="{GRAY_BAR}" stroke-width="1"/>')
    a(people(community_x + 10, community_y + 6, MUTED))
    a(f'<text x="{community_x + 30:.1f}" y="{community_y + 11}" class="leg">{t["community"]}</text>')
    # header
    cols = t["cols"]
    a(f'<text x="{X_RANK}" y="{TOP - 14}" class="col" text-anchor="middle">{cols[0]}</text>')
    a(f'<text x="{X_NAME}" y="{TOP - 14}" class="col">{cols[1]}</text>')
    a(f'<text x="{X_VAL}" y="{TOP - 14}" class="col" text-anchor="end">{cols[2]}</text>')
    a(f'<text x="{X_CHIP}" y="{TOP - 14}" class="col">{cols[3]}</text>')
    for i in range(n):
        if i % 2 == 0:
            a(f'<rect x="{PAD - 8}" y="{TOP + i * ROW_H}" width="{W - 2 * PAD + 16}" height="{ROW_H}" rx="6" fill="{STRIPE}"/>')
    for v in (0, 2, 4, 6, 8, 10):
        a(f'<line x1="{sx(v):.1f}" y1="{TOP}" x2="{sx(v):.1f}" y2="{bottom}" stroke="{GRID}" stroke-width="1"/>')
        a(f'<text x="{sx(v):.1f}" y="{bottom + 18}" class="tick" text-anchor="middle">{v}×</text>')
    a(f'<line x1="{sx(1):.1f}" y1="{TOP - 6}" x2="{sx(1):.1f}" y2="{bottom}" stroke="{ACCENT}" stroke-width="1" stroke-dasharray="3 3"/>')
    a(f'<text x="{sx(1) + 5:.1f}" y="{TOP - 14}" class="tick" fill="{ACCENT}" style="fill:{ACCENT}">{t["baseline"]}</text>')
    chip_style = {"bitwise": ("#FBEAEA", "#E7C2C2", ACCENT), "stat": ("#F3F0F0", "#E4DCDC", INK2),
                  "task": ("#EEF4FA", "#D3E2F0", "#2F5F8A"), "explore": ("#FFFFFF", "#D9CFCF", MUTED),
                  "manual": ("#F8F1EA", "#E6D5C3", "#7A5634")}
    for i, (name, d, v, label, est, val) in enumerate(PROJECTS):
        cy = TOP + i * ROW_H + ROW_H / 2
        color = DOMAINS[d][0]
        a(f'<text x="{X_RANK}" y="{cy + 4.5:.1f}" class="rank" text-anchor="middle">{i + 1}</text>')
        a(f'<circle cx="{X_NAME + 5}" cy="{cy:.1f}" r="5" fill="{color}"/>')
        a(f'<text x="{X_NAME + 18}" y="{cy + 5:.1f}" class="name">{name}</text>')
        if name in COMMUNITY:
            a(people(X0 - 26, cy, MUTED))
        lo, hi = v if isinstance(v, tuple) else (v, v)
        if hi > lo:
            # a range: the solid bar ends at the low end, a lighter segment runs to the high end
            a(f'<path d="{bar_path(X0, cy - BAR_H / 2, sx(hi), BAR_H)}" fill="{color}" fill-opacity=".35"/>')
        a(f'<path d="{bar_path(X0, cy - BAR_H / 2, sx(lo), BAR_H, 0 if hi > lo else 4)}" fill="{"url(#h-" + d + ")" if est else color}"/>')
        a(f'<text x="{X_VAL}" y="{cy + 5:.1f}" class="val" text-anchor="end">{label}{"*" if est else ""}</text>')
        fill, stroke, ink = chip_style[val]
        chip = t["valid"][val]
        cw = len(chip) * (7.4 if lang == "en" else 13) + 20
        dash = ' stroke-dasharray="3 2"' if val == "explore" else ""
        a(f'<rect x="{X_CHIP}" y="{cy - 11:.1f}" width="{cw:.0f}" height="22" rx="11" fill="{fill}" stroke="{stroke}"{dash}/>')
        a(f'<text x="{X_CHIP + cw / 2:.1f}" y="{cy + 4.5:.1f}" class="chip" text-anchor="middle" style="fill:{ink}">{chip}</text>')
    a(f'<text x="{PAD}" y="{H - 42}" class="foot">* {t["foot"]}</text>')
    a(f'<text x="{PAD}" y="{H - 22}" class="foot">{t["scope_foot"]}</text>')
    a('</svg>')
    return "\n".join(o) + "\n"


def time_chart(lang):
    t = TEXT[lang]
    li = 0 if lang == "en" else 1
    PAD, TOP, ROW_H = 36, 128, 84
    X0, FULL = 300, 400
    n = len(SHOWCASE)
    H = TOP + n * ROW_H + 52
    defs = hatch("h-est", ACCENT)
    desc = "; ".join(f"{s[0]}: {s[4][0]} → {s[4][1]} ({s[5]})" for s in SHOWCASE)
    o = head(H, t["t_title"], desc, defs)
    a = o.append
    a(f'<text x="{PAD}" y="56" class="title">{t["t_title"]}</text>')
    a(f'<text x="{PAD}" y="84" class="sub">{t["t_sub"]}</text>')
    # key
    kx = W - PAD - 230
    a(f'<rect x="{kx}" y="72" width="22" height="10" rx="3" fill="{GRAY_BAR}"/><text x="{kx + 30}" y="81" class="leg">{t["t_base"]}</text>')
    a(f'<rect x="{kx + 110}" y="72" width="22" height="10" rx="3" fill="{ACCENT}"/><text x="{kx + 140}" y="81" class="leg">{t["t_opt"]}</text>')
    for i, (name, d, base, opt, (lb, lo), sp, est, scope, quality) in enumerate(SHOWCASE):
        y = TOP + i * ROW_H
        if i:
            a(f'<line x1="{PAD}" y1="{y - 10}" x2="{W - PAD}" y2="{y - 10}" stroke="{GRID}" stroke-width="1"/>')
        color = DOMAINS[d][0]
        a(f'<circle cx="{PAD + 5}" cy="{y + 14}" r="5" fill="{color}"/>')
        a(f'<text x="{PAD + 18}" y="{y + 19}" class="name">{name}</text>')
        a(f'<text x="{PAD}" y="{y + 42}" class="dom">{scope[li]}</text>')
        wo = FULL * opt / base
        a(f'<path d="{bar_path(X0, y + 4, X0 + FULL, 16)}" fill="{GRAY_BAR}"/>')
        a(f'<text x="{X0 + FULL + 8}" y="{y + 17}" class="tl">{lb}</text>')
        a(f'<path d="{bar_path(X0, y + 26, X0 + wo, 16)}" fill="{"url(#h-est)" if est else ACCENT}"/>')
        a(f'<text x="{X0 + wo + 8:.1f}" y="{y + 39}" class="tlb">{lo}</text>')
        a(f'<text x="{X0}" y="{y + 64}" class="q">✓ {quality[li]}</text>')
        a(f'<text x="{W - PAD}" y="{y + 32}" class="big" text-anchor="end">{sp}</text>')
        if est:
            a(f'<text x="{W - PAD}" y="{y + 50}" class="dom" text-anchor="end">{t["est"]}</text>')
    a(f'<text x="{PAD}" y="{H - 22}" class="foot">{t["t_foot"]}</text>')
    a('</svg>')
    return "\n".join(o) + "\n"


for lang, suffix in (("en", ""), ("zh", ".zh-CN")):
    (FIGS / f"results-overview{suffix}.svg").write_text(leaderboard(lang), encoding="utf-8")
    (FIGS / f"results-time{suffix}.svg").write_text(time_chart(lang), encoding="utf-8")
