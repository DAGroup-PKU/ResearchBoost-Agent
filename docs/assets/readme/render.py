"""Render the README banner, buttons and highlight cards (English and Chinese) with headless Chrome.

Edit the text below, then run from the repository root (optionally naming assets, e.g. `run` or `card-judge`):

    python3 docs/assets/readme/render.py

Needs Google Chrome (set CHROME to its binary if it is not at the macOS default path) and network access for
Google Fonts. The banner art is docs/assets/teaser.jpg.
"""
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

OUT = Path(__file__).resolve().parent
ART = (OUT.parent / "teaser.jpg").as_uri()
CHROME = os.environ.get("CHROME", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")

ICONS = {
    "chart": '<path d="M5 20V11M12 20V5M19 20v-7"/><path d="M3 21h18"/>',
    "terminal": '<rect x="3" y="4" width="18" height="16" rx="3"/><path d="M7 9l3 3-3 3M13 15h4"/>',
    "shield": '<path d="M12 3l7 3v5c0 4.5-3 8-7 10-4-2-7-5.5-7-10V6z"/><path d="M9 12l2 2 4-4"/>',
    "loop": '<path d="M20 12a8 8 0 1 1-2.3-5.7"/><path d="M20 4v5h-5"/>',
    "book": '<path d="M5 4h9a4 4 0 0 1 4 4v12a3 3 0 0 0-3-3H5z"/><path d="M5 4v13"/>',
    "rocket": '<path d="M5 15c-1.5 1.5-2 4-2 6 2 0 4.5-.5 6-2"/><path d="M9 15l-3-3c1-4 4.5-8 12-9-1 7.5-5 11-9 12z"/><circle cx="14.5" cy="9.5" r="1.5"/>',
    "arrow": '<path d="M5 12h14M13 6l6 6-6 6"/>',
}


def icon(name, size=24, stroke=2.2):
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            f'stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round">{ICONS[name]}</svg>')


# Domain colors match docs/figures/make_results_overview.py.
DOMAINS = [
    ("emb", "#6250d6", "Embodied AI", "具身智能", "Motus", "9.59×"),
    ("img", "#eb6834", "Image gen", "图像生成", "RAE", "5.32×"),
    ("mllm", "#1baf7a", "MLLM", "多模态模型", "LLaVA-OneVision-2", "3.0×"),
    ("sci", "#e87ba4", "AI for Science", "AI for Science", "ProteinMPNN", "2.25×"),
    ("llm", "#2a78d6", "LLM", "语言模型", "YOCO", "2.12×"),
    ("vid", "#eda100", "Video gen", "视频生成", "Self-Forcing", "2.08×"),
    ("3d", "#333333", "3D recon", "三维重建", "3DGS · 30k", "1.13×"),
]

T = {
    "en": {
        "lang": "en",
        "eyebrow": "AI4AI · OPEN-SOURCE TRAINING ACCELERATION",
        "tagline": "AI that accelerates the research that advances AI.",
        "steps": ["Profile", "Optimize", "Verify"],
        "buttons": [("blog", "book", "Read the blog", "crimson"), ("results", "chart", "Explore results", "ink"),
                    ("quickstart", "rocket", "Quick start", "light")],
        "cards": [
            ("results", "chart", "01 · RESULTS", "Up to 9.59× faster,<br>across seven domains",
             "21 projects, from LLM pretraining to 3D reconstruction, robot policies and protein design, each measured against its own baseline.",
             "See all results"),
            ("prompt", "terminal", "02 · ONE PROMPT", "Paste one prompt,<br>get a verified speedup",
             "Open Codex or Claude Code on a GPU host. The agent finds the code, asks only what is yours to decide, and does the rest.",
             "Quick start"),
            ("judge", "shield", "03 · FROZEN JUDGE", "Faster, never<br>different",
             "A hashed judge, frozen before round one, keeps a change only if it is bitwise identical or within the baseline's own seed noise.",
             "How the judge works"),
            ("loop", "loop", "04 · THE LOOP", "Profile, optimize,<br>verify, repeat",
             "One /goal session in a sandbox attacks one hotspot per round, guided by nine skill cards and supervised from outside.",
             "How a run works"),
        ],
        "domain_top": "featured speedup",
        "prompt_lines": ["Use ResearchBoost-Agent to speed up my training.",
                         "Clone it outside my repository and follow",
                         "goal/templates/FORMULATOR_BRIEF.md from section 0.5."],
        "asks": ["Which code and training command?", "Which GPUs, and where to work?", "Workload, precision, target speedup?"],
        "done": "Verified speedup reported",
        "gates": [("0", "Structure", "entry points unchanged"), ("1", "Benefit", "beats baseline spread"),
                  ("2", "Bitwise", "every tensor matches"), ("3", "Statistical", "within 3σ of seed noise")],
        "frozen": "Frozen before round one",
        "pass": "PASS",
        "rule": "Gates run in order and stop at the first failure; a bitwise match accepts at once.",
        "loop": ["Profile", "Edit", "Time", "Check"],
        "loop_center": ["/goal", "session · sandbox"],
        "skills": ["tiered profiling", "action ladder", "fusion patterns", "numerics validation",
                   "kernel utilization", "measurement", "FSDP multi-forward", "SGLang diffusion ×2"],
        "run": {
            "stages": [("1", "Frame and freeze", "before the agent starts"),
                       ("2", "Goal mode", "one agent, round after round"),
                       ("3", "Accept", "only what the judge accepts")],
            "frame": ["Fix workload, precision and training semantics with the owner",
                      "Build the judge: bench/ and a frozen copy of numerics/",
                      "Record the baseline twice, bitwise identical; freeze σ",
                      "Known-bad candidates: every gate must fire",
                      "Hash the judge, write GOAL.md"],
            "handoff": "GOAL.md · hashed judge",
            "clawo": ("claw-orchestrator", "keeps the thread across restarts and usage limits · runs the final check itself"),
            "sandbox": "bubblewrap sandbox · assigned GPUs only · judge read-only",
            "session": ("/goal", "Codex or Claude Code"),
            "loop": [("Profile", "stage → module → op → kernel"), ("Edit", "one hotspot, behind a switch"),
                     ("Time", "bench.py time"), ("Check", "bench.py check at target"),
                     ("Log", "WORK_LOG.md, failures too")],
            "supervisor": ("Supervisor", "watches from outside · rules and facts only, never direction"),
            "accept": ["check --final: PASS", "speedup ≥ the target in GOAL.md", "judge files byte for byte unchanged"],
            "confirm": "then a real end-to-end training or evaluation run",
            "result": "Verified speedup",
            "repair": "A failed final check goes back to the same thread once.",
        },
        "contrib": {
            "steps": [("Run", "Run the Quick start on your own training code."),
                      ("Submit", "A solid speedup or a trajectory worth reading? Open a case submission."),
                      ("Review", "We review it against the judge and label it."),
                      ("Merge", "Once accepted, open a pull request that adds your row to Results.")],
            "term": ["Use ResearchBoost-Agent to", "speed up my training."],
            "term_ok": "bench.py check: PASS",
            "issue": "[Case] Faster-YourProject",
            "fields": ["Upstream repository", "Domain", "Workload", "Optimized repository"],
            "criteria": ["Frozen judge PASS", "A real gain, or rounds that teach something", "Public repo, reproducible commands"],
            "pr": "docs: add Faster-YourProject",
            "files": [("README.md", "+1 row"), ("docs/results/…", "+1 section")],
            "merged": "Merged",
            "you": "You join the contributors",
        },
    },
    "zh": {
        "lang": "zh-CN",
        "eyebrow": "AI4AI · 开源训练加速",
        "tagline": "用 AI 加速 AI 研究。",
        "steps": ["分析", "优化", "验证"],
        "buttons": [("blog", "book", "阅读项目 Blog", "crimson"), ("results", "chart", "查看全部结果", "ink"),
                    ("quickstart", "rocket", "快速开始", "light")],
        "cards": [
            ("results", "chart", "01 · 结果", "最高 9.59× 加速，<br>覆盖七个领域",
             "21 个项目，覆盖大模型预训练、三维重建、机器人策略和蛋白质设计，每个都与自身的基线对比。",
             "查看全部结果"),
            ("prompt", "terminal", "02 · 一段提示词", "粘贴一段话，<br>拿到验证过的加速",
             "在 GPU 主机上打开 Codex 或 Claude Code。agent 自己找到代码，只问需要你决定的事，其余全部自己完成。",
             "快速开始"),
            ("judge", "shield", "03 · 固定评测", "更快，<br>但结果不变",
             "评测在第一轮之前固定并记录哈希。只有与基线逐位一致，或差异不超过基线自身随机种子波动的改动才会保留。",
             "评测如何工作"),
            ("loop", "loop", "04 · 优化循环", "分析、优化、<br>验证，循环往复",
             "一个 /goal 会话在沙箱里每轮攻克一个瓶颈，参考 9 份技能文档，由外部监督。",
             "运行流程"),
        ],
        "domain_top": "代表性加速比",
        "prompt_lines": ["用 ResearchBoost-Agent 加速我的训练。",
                         "把它克隆到我的仓库之外，从",
                         "goal/templates/FORMULATOR_BRIEF.md 第 0.5 节开始执行。"],
        "asks": ["用哪份代码、哪条训练命令？", "用哪些 GPU，在哪里工作？", "训练负载、精度、目标加速比？"],
        "done": "报告经过验证的加速比",
        "gates": [("0", "结构", "训练入口未被修改"), ("1", "收益", "超过基线计时波动"),
                  ("2", "逐位", "全部张量逐位一致"), ("3", "统计", "偏差在 3σ 以内")],
        "frozen": "第一轮之前固定",
        "pass": "通过",
        "rule": "各项检查依次执行，任一项失败即停止；逐位一致则直接通过。",
        "loop": ["分析", "修改", "计时", "评测"],
        "loop_center": ["/goal", "会话 · 沙箱"],
        "skills": ["分层性能分析", "分阶段优化", "算子融合", "数值验证",
                   "内核利用率", "计时方法", "FSDP 多次前向", "SGLang 扩散 ×2"],
        "run": {
            "stages": [("1", "定义并固定", "在 agent 启动之前"),
                       ("2", "goal 模式", "一个 agent，一轮接一轮"),
                       ("3", "验收", "只认评测的结论")],
            "frame": ["与负责人确定训练负载、精度和训练语义",
                      "搭建评测：bench/ 与固定版本的 numerics/",
                      "两次基线逐位一致，固定 σ",
                      "用已知有问题的候选，确认每项检查都会拒绝",
                      "记录评测哈希，编写 GOAL.md"],
            "handoff": "GOAL.md · 已哈希的评测",
            "clawo": ("claw-orchestrator", "重启或用量受限后接回原会话 · 亲自运行最终评测"),
            "sandbox": "bubblewrap 沙箱 · 只有分配的 GPU · 评测只读",
            "session": ("/goal", "Codex 或 Claude Code"),
            "loop": [("分析", "阶段→模块→算子→内核"), ("修改", "一个瓶颈，用开关控制"),
                     ("计时", "bench.py time"), ("评测", "达标后 bench.py check"),
                     ("记录", "写入 WORK_LOG.md")],
            "supervisor": ("监督者", "在外部观察 · 只管规则和事实，不指导优化方向"),
            "accept": ["check --final：通过", "加速比达到 GOAL.md 目标", "评测文件逐字节未改动"],
            "confirm": "再用真实的端到端训练或评测确认",
            "result": "经过验证的加速",
            "repair": "最终评测失败时，结果会退回同一会话，再给一次修复机会。",
        },
        "contrib": {
            "steps": [("运行", "在你自己的训练代码上跑一遍快速开始。"),
                      ("提交", "加速比不错，或优化过程值得分享？按模板提交 case。"),
                      ("审核", "我们对照评测结果审核，并打上标签。"),
                      ("合并", "审核通过后，提 PR 把你的结果加进已有结果。")],
            "term": ["用 ResearchBoost-Agent", "加速我的训练。"],
            "term_ok": "bench.py check：通过",
            "issue": "[Case] Faster-YourProject",
            "fields": ["上游仓库", "领域", "训练负载", "优化后的仓库"],
            "criteria": ["固定评测通过", "真实加速，或有价值的优化轨迹", "公开仓库，命令可复现"],
            "pr": "docs: add Faster-YourProject",
            "files": [("README.md", "+1 行"), ("docs/results/…", "+1 节")],
            "merged": "已合并",
            "you": "你成为 contributor",
        },
    },
}

BASE_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,400;0,500;0,600;0,700;0,800;1,500&family=Noto+Sans+SC:wght@400;500;700;800&display=block');
:root { --crimson:#8C0000; --crimson2:#B4182D; --ink:#241B1C; --ink2:#4E4244; --muted:#867678; --line:#EADCDC; --bg:#F5F0EE; }
* { box-sizing: border-box; margin: 0; padding: 0; }
html, body { background: transparent; }
body { font-family: 'Plus Jakarta Sans', 'Noto Sans SC', sans-serif; color: var(--ink); -webkit-font-smoothing: antialiased; }
"""


def banner(t):
    steps = '<span class="dot"></span>'.join(f"<span>{s}</span>" for s in t["steps"])
    return f"""<style>{BASE_CSS}
body {{ width:1360px; height:400px; background:var(--bg); position:relative; overflow:hidden; }}
.art {{ position:absolute; right:-10px; top:0; height:400px; }}
.fade {{ position:absolute; left:550px; top:0; width:130px; height:400px; background:linear-gradient(90deg,var(--bg),rgba(245,240,238,0)); }}
.txt {{ position:absolute; left:64px; top:0; height:400px; display:flex; flex-direction:column; justify-content:center; padding-top:64px; width:600px; }}
.eyebrow {{ font-size:12.5px; letter-spacing:.26em; color:var(--muted); font-weight:600; }}
h1 {{ font-size:50px; font-weight:800; letter-spacing:-.02em; margin:14px 0 12px; line-height:1.05; white-space:nowrap; }}
h1 b {{ color:var(--crimson); font-weight:800; }}
.tag {{ font-size:21px; font-weight:500; font-style:italic; color:var(--ink2); }}
:lang(zh-CN) .tag {{ font-style:normal; }}
.steps {{ margin-top:26px; display:flex; align-items:center; gap:14px; font-size:16px; font-weight:600; color:var(--ink2); }}
.steps .dot {{ width:6px; height:6px; border-radius:3px; background:var(--crimson); display:inline-block; }}
</style>
<img class="art" src="{ART}"><div class="fade"></div>
<div class="txt"><div class="eyebrow">{t['eyebrow']}</div><h1><b>ResearchBoost</b>-Agent</h1>
<div class="tag">{t['tagline']}</div><div class="steps">{steps}</div></div>"""


def button(t, ic, label, style):
    styles = {
        "crimson": "background:linear-gradient(100deg,#8C0000,#B4182D); color:#fff; box-shadow:0 10px 22px -8px rgba(140,0,0,.55);",
        "ink": "background:linear-gradient(100deg,#2B2324,#4A3F40); color:#fff; border:1.5px solid rgba(255,255,255,.16); box-shadow:0 10px 22px -8px rgba(36,27,28,.5);",
        "light": "background:#fff; color:var(--crimson); border:2px solid #E2C4C4; box-shadow:0 10px 22px -10px rgba(140,0,0,.35);",
    }
    return f"""<style>{BASE_CSS}
body {{ width:380px; height:96px; display:flex; align-items:center; justify-content:center; }}
.b {{ width:350px; height:66px; border-radius:33px; display:flex; align-items:center; justify-content:center; gap:14px;
      font-size:22px; font-weight:700; letter-spacing:-.005em; {styles[style]} }}
.b .a {{ opacity:.85; }}
</style><div class="b">{icon(ic, 24)}<span>{label}</span><span class="a">{icon('arrow', 22)}</span></div>"""


def card_right(kind, t):
    if kind == "results":
        tiles = "".join(
            f'<div class="tile"><div class="th"><span class="sw" style="background:{c}"></span>{en if t is T["en"] else zh}</div>'
            f'<div class="big">{x}</div><div class="pj{" long" if len(pj) > 14 else ""}">{pj}</div></div>'
            for _, c, en, zh, pj, x in DOMAINS)
        return f'<div class="grid">{tiles}</div><div class="note">{t["domain_top"]}</div>'
    if kind == "prompt":
        lines = "".join(f'<div class="ln">{"<span class=p>›</span> " if i == 0 else "<span class=p>&nbsp;</span> "}{l}</div>'
                        for i, l in enumerate(t["prompt_lines"]))
        asks = "".join(f'<div class="ask"><span class="q">?</span>{a}</div>' for a in t["asks"])
        return (f'<div class="term"><div class="bar"><i></i><i></i><i></i></div>{lines}<div class="asks">{asks}</div>'
                f'<div class="ok">{icon("shield", 16, 2.4)} {t["done"]}</div></div>')
    if kind == "judge":
        gates = '<div class="gar">›</div>'.join(
            f'<div class="gate"><div class="gn">{n}</div><div class="gt">{name}</div><div class="gd">{d}</div></div>'
            for n, name, d in t["gates"])
        return (f'<div class="gates">{gates}</div><div class="jfoot"><div class="frz">❄ {t["frozen"]}</div>'
                f'<div class="verd"><span class="vx">no_benefit</span><span class="vx">INCORRECT</span>'
                f'<div class="pass">{icon("shield", 18, 2.4)} {t["pass"]}</div></div></div><div class="rule">{t["rule"]}</div>')
    if kind == "loop":
        pos = [(50, 11), (89, 50), (50, 89), (11, 50)]
        nodes = "".join(f'<div class="node" style="left:{x}%;top:{y}%">{s}</div>' for s, (x, y) in zip(t["loop"], pos))
        skills = "".join(f'<span class="sk">{s}</span>' for s in t["skills"])
        return (f'<div class="loopwrap"><div class="ring"></div>{nodes}<div class="lc"><b>{t["loop_center"][0]}</b>'
                f'<span>{t["loop_center"][1]}</span></div></div><div class="skills">{skills}</div>')


def card(t, kind, ic, eyebrow, title, body, link):
    return f"""<style>{BASE_CSS}
body {{ width:1240px; height:420px; padding:18px; }}
.c {{ width:100%; height:100%; border-radius:28px; border:1.5px solid var(--line); position:relative; overflow:hidden;
      background:radial-gradient(circle at 92% 8%, rgba(180,24,45,.07), transparent 40%), linear-gradient(135deg,#FFFFFF 0%,#FCF9F8 55%,#F8F1F0 100%);
      box-shadow:0 14px 30px -18px rgba(90,30,30,.35); display:flex; align-items:center; padding:0 40px 0 46px; gap:36px; }}
.l {{ width:470px; flex:none; }}
.ic {{ width:54px; height:54px; border-radius:16px; background:linear-gradient(135deg,#8C0000,#C2263B); color:#fff;
       display:flex; align-items:center; justify-content:center; box-shadow:0 8px 18px -8px rgba(140,0,0,.7); }}
.eh {{ display:flex; align-items:center; gap:18px; }}
.ey {{ font-size:15.5px; font-weight:800; letter-spacing:.2em; color:var(--crimson); }}
h2 {{ font-size:39px; line-height:1.14; font-weight:800; letter-spacing:-.02em; margin:22px 0 14px; }}
p {{ font-size:18px; line-height:1.55; color:var(--muted); }}
.lk {{ margin-top:22px; font-size:17px; font-weight:700; color:var(--crimson); display:flex; align-items:center; gap:8px; }}
.r {{ flex:1; height:318px; border-radius:20px; background:rgba(255,255,255,.75); border:1px solid var(--line);
      box-shadow:0 6px 18px -12px rgba(90,30,30,.3); padding:22px 24px; position:relative; min-width:0; }}
/* results */
.grid {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:10px; }}
.tile {{ border-radius:14px; border:1px solid var(--line); background:#fff; padding:13px 10px 12px; }}
.th {{ display:flex; align-items:center; gap:6px; font-size:12px; font-weight:600; color:var(--muted); white-space:nowrap; }}
.sw {{ width:12px; height:12px; border-radius:4px; flex:none; }}
.big {{ font-size:30px; font-weight:800; letter-spacing:-.02em; margin-top:6px; }}
.pj {{ font-size:13.5px; color:var(--ink2); font-weight:600; white-space:nowrap; }}
.pj.long {{ font-size:12px; letter-spacing:-.015em; }}
.note {{ position:absolute; right:24px; bottom:12px; font-size:12px; color:var(--muted); letter-spacing:.06em; }}
/* prompt */
.r:has(.term) {{ background:#221B1C; border-color:#221B1C; padding:0; overflow:hidden; }}
.term {{ color:#EDE6E6; font-family:'SFMono-Regular',Menlo,Consolas,monospace; font-size:14px; }}
.bar {{ height:34px; background:#2E2627; display:flex; align-items:center; gap:7px; padding-left:16px; margin-bottom:16px; }}
.bar i {{ width:11px; height:11px; border-radius:6px; background:#5A4E4F; display:block; }}
.bar i:first-child {{ background:#C2263B; }}
.ln {{ padding:0 22px; line-height:1.7; white-space:nowrap; }}
.p {{ color:#FF8A8A; font-weight:700; }}
.asks {{ display:flex; flex-direction:column; gap:7px; margin:16px 22px 0; font-family:'Plus Jakarta Sans','Noto Sans SC',sans-serif; }}
.ask {{ align-self:flex-start; background:#3A3031; border:1px solid #4D4142; border-radius:10px; padding:6px 12px; font-size:14px; display:flex; gap:9px; align-items:center; }}
.q {{ width:18px; height:18px; border-radius:9px; background:#C2263B; color:#fff; font-size:12px; font-weight:800; display:flex; align-items:center; justify-content:center; }}
.ok {{ position:absolute; right:20px; bottom:16px; font-family:'Plus Jakarta Sans','Noto Sans SC',sans-serif; font-size:14px; font-weight:700; color:#fff;
       background:linear-gradient(100deg,#8C0000,#C2263B); border-radius:10px; padding:7px 12px; display:flex; align-items:center; gap:7px; }}
/* judge */
.gates {{ display:flex; align-items:stretch; gap:6px; margin-top:6px; }}
.gate {{ flex:1; border-radius:14px; background:#fff; border:1px solid var(--line); padding:20px 10px 18px; text-align:center; }}
.gn {{ width:30px; height:30px; margin:0 auto 8px; border-radius:15px; background:#F6E7E7; color:var(--crimson); font-weight:800; font-size:15px; display:flex; align-items:center; justify-content:center; }}
.gt {{ font-size:17px; font-weight:800; }}
.gd {{ font-size:12.5px; color:var(--muted); margin-top:4px; line-height:1.35; }}
.gar {{ color:var(--crimson); font-weight:800; font-size:22px; display:flex; align-items:center; }}
.jfoot {{ display:flex; align-items:center; justify-content:space-between; margin-top:22px; }}
.verd {{ display:flex; gap:8px; align-items:center; }}
.vx {{ font-size:13.5px; font-weight:700; color:var(--muted); background:#F3EEEE; border:1px solid var(--line); border-radius:10px; padding:8px 11px; font-family:'SFMono-Regular',Menlo,monospace; }}
.rule {{ margin-top:18px; font-size:13.5px; color:var(--muted); }}
.frz {{ white-space:nowrap; font-size:15px; font-weight:600; color:#3E6E8E; background:#EAF3F8; border:1px solid #CFE2EC; border-radius:10px; padding:9px 14px; }}
.pass {{ font-size:17px; font-weight:800; color:#fff; background:linear-gradient(100deg,#8C0000,#C2263B); border-radius:12px; padding:9px 18px; display:flex; align-items:center; gap:8px; letter-spacing:.06em; }}
/* loop */
.loopwrap {{ position:absolute; left:20px; top:22px; width:270px; height:272px; }}
.ring {{ position:absolute; left:50%; top:50%; width:212px; height:212px; transform:translate(-50%,-50%); border-radius:50%; border:3px dashed #D9B9B9; }}
.node {{ position:absolute; transform:translate(-50%,-50%); background:linear-gradient(135deg,#8C0000,#C2263B); color:#fff; font-weight:700; font-size:14.5px;
         border-radius:12px; padding:7px 13px; box-shadow:0 6px 14px -6px rgba(140,0,0,.6); white-space:nowrap; }}
.lc {{ position:absolute; left:50%; top:50%; transform:translate(-50%,-50%); text-align:center; font-size:13px; color:var(--muted); line-height:1.4; }}
.lc b {{ display:block; color:var(--ink); font-size:14px; font-family:'SFMono-Regular',Menlo,monospace; margin-bottom:2px; }}
.skills {{ position:absolute; left:306px; right:22px; top:26px; display:flex; flex-wrap:wrap; gap:8px; align-content:flex-start; }}
.sk {{ font-size:13.5px; font-weight:600; color:var(--ink2); background:#fff; border:1px solid var(--line); border-radius:999px; padding:6px 12px; }}
.sk:nth-child(-n+2) {{ border-color:#E2C4C4; color:var(--crimson); background:#FBF2F2; }}
</style>
<div class="c"><div class="l"><div class="eh"><div class="ic">{icon(ic, 28)}</div><div class="ey">{eyebrow}</div></div>
<h2>{title}</h2><p>{body}</p><div class="lk">{link} {icon('arrow', 18, 2.6)}</div></div>
<div class="r">{card_right(kind, t)}</div></div>"""


def run_figure(t):
    r = t["run"]

    def stage(i):
        n, title, sub = r["stages"][i]
        return f'<div class="sh"><span class="sn">{n}</span><div><b>{title}</b><span>{sub}</span></div></div>'

    check = '<span class="ck"><svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="3.4" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12l5 5 9-10"/></svg></span>'
    frame = "".join(f'<li>{check}<span>{x}</span></li>' for x in r["frame"])
    accept = "".join(f'<li>{check}<span>{x}</span></li>' for x in r["accept"])
    # the round loop: five nodes on an ellipse, clockwise from the top, with arcs between them
    W, H, cx, cy, rx, ry = 520, 330, 260, 166, 186, 122
    angles = [-90, -18, 54, 126, 198]
    pt = lambda a: (cx + rx * math.cos(math.radians(a)), cy + ry * math.sin(math.radians(a)))
    arcs = ""
    for i, a0 in enumerate(angles):
        a1 = angles[(i + 1) % 5] + (360 if i == 4 else 0)
        gap = 30 if i == 2 else 21  # the bottom arc runs under the two lower nodes
        (x0, y0), (x1, y1) = pt(a0 + gap), pt(a1 - gap)
        arcs += f'<path d="M{x0:.1f} {y0:.1f}A{rx} {ry} 0 0 1 {x1:.1f} {y1:.1f}" marker-end="url(#ah)"/>'
    nodes = "".join(
        f'<div class="nd" style="left:{pt(a)[0]:.0f}px;top:{pt(a)[1]:.0f}px"><b>{name}</b><span>{d}</span></div>'
        for a, (name, d) in zip(angles, r["loop"]))
    loop = (f'<div class="loop" style="width:{W}px;height:{H}px"><svg width="{W}" height="{H}">'
            f'<defs><marker id="ah" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto"><path d="M0 0L8 4L0 8Z" fill="#B4182D"/></marker></defs>'
            f'<g fill="none" stroke="#D9AEB0" stroke-width="2.4">{arcs}</g></svg>{nodes}'
            f'<div class="ctr"><b>{r["session"][0]}</b><span>{r["session"][1]}</span></div></div>')
    nxt = f'<div class="nx">{icon("arrow", 20, 2.8)}</div>'
    return f"""<style>{BASE_CSS}
body {{ width:1240px; height:660px; padding:18px; }}
.f {{ width:100%; height:100%; border-radius:28px; border:1.5px solid var(--line); padding:26px 28px;
      background:radial-gradient(circle at 50% 0%, rgba(180,24,45,.06), transparent 45%), linear-gradient(135deg,#FFFFFF 0%,#FCF9F8 55%,#F8F1F0 100%);
      box-shadow:0 14px 30px -18px rgba(90,30,30,.35); display:grid; grid-template-columns:292px 30px 1fr 30px 262px; align-items:stretch; }}
.col {{ display:flex; flex-direction:column; min-width:0; }}
.sh {{ display:flex; gap:11px; align-items:center; height:50px; margin-bottom:12px; }}
.sn {{ width:30px; height:30px; flex:none; border-radius:10px; background:linear-gradient(135deg,#8C0000,#C2263B); color:#fff; font-weight:800; font-size:16px;
       display:flex; align-items:center; justify-content:center; box-shadow:0 6px 12px -6px rgba(140,0,0,.7); }}
.sh b {{ display:block; font-size:20px; font-weight:800; letter-spacing:-.01em; }}
.sh div span {{ display:block; font-size:14px; color:var(--muted); margin-top:1px; }}
.pn {{ flex:1; border-radius:18px; background:rgba(255,255,255,.85); border:1px solid var(--line); padding:18px 18px 16px; display:flex; flex-direction:column;
       box-shadow:0 6px 18px -12px rgba(90,30,30,.3); }}
ul {{ list-style:none; display:flex; flex-direction:column; gap:13px; }}
li {{ display:flex; gap:10px; font-size:15.5px; line-height:1.4; color:var(--ink2); }}
.ck {{ width:20px; height:20px; flex:none; border-radius:10px; background:#B4182D; display:flex; align-items:center; justify-content:center; margin-top:1px; }}
.ho {{ margin-top:auto; align-self:flex-start; font-size:14px; font-weight:700; color:var(--crimson); background:#FBF0F0; border:1px solid #E7C2C2; border-radius:10px; padding:8px 12px; font-family:'SFMono-Regular',Menlo,monospace; }}
.nx {{ display:flex; align-items:center; justify-content:center; color:var(--crimson); padding-top:62px; }}
.bar {{ border-radius:12px; padding:9px 14px; font-size:14px; display:flex; gap:10px; align-items:center; }}
.bar b {{ font-size:15px; white-space:nowrap; }}
.cl {{ background:linear-gradient(100deg,#2B2324,#4A3F40); color:#EFE7E7; }}
.cl b {{ color:#fff; font-family:'SFMono-Regular',Menlo,monospace; }}
.sv {{ background:#fff; border:1px solid var(--line); color:var(--muted); }}
.sv b {{ color:var(--ink); }}
.sb {{ flex:1; margin:10px 0; border:2px dashed #D9B9B9; border-radius:18px; position:relative; background:rgba(255,255,255,.55); display:flex; align-items:center; justify-content:center; }}
.sbl {{ position:absolute; left:14px; top:-11px; background:#FBF6F5; padding:0 8px; font-size:13.5px; font-weight:700; color:#9C7D7F; letter-spacing:.02em; }}
.loop {{ position:relative; }}
.loop svg {{ position:absolute; left:0; top:0; }}
.nd {{ position:absolute; transform:translate(-50%,-50%); width:168px; background:#fff; border:1px solid #E7C2C2; border-radius:12px; padding:7px 8px 8px; text-align:center;
       box-shadow:0 6px 14px -8px rgba(140,0,0,.45); }}
.nd b {{ display:block; font-size:16.5px; font-weight:800; color:var(--crimson); }}
.nd span {{ display:block; font-size:13px; color:var(--muted); margin-top:2px; line-height:1.3; }}
.ctr {{ position:absolute; left:50%; top:50%; transform:translate(-50%,-50%); text-align:center; }}
.ctr b {{ display:block; font-size:24px; font-weight:800; font-family:'SFMono-Regular',Menlo,monospace; color:var(--ink); }}
.ctr span {{ display:block; font-size:14px; color:var(--muted); margin-top:3px; }}
.down {{ text-align:center; color:var(--crimson); margin:12px 0 4px; }}
.cf {{ font-size:14.5px; color:var(--ink2); text-align:center; line-height:1.4; }}
.rp {{ margin-top:auto; margin-bottom:12px; font-size:14px; color:var(--muted); text-align:center; line-height:1.4; }}
.res {{ font-size:17px; font-weight:800; color:#fff; background:linear-gradient(100deg,#8C0000,#C2263B); border-radius:14px; padding:13px 14px;
        display:flex; align-items:center; justify-content:center; gap:9px; box-shadow:0 10px 20px -10px rgba(140,0,0,.7); }}
</style>
<div class="f">
<div class="col">{stage(0)}<div class="pn"><ul>{frame}</ul><div class="ho">{r["handoff"]}</div></div></div>
{nxt}
<div class="col">{stage(1)}
<div class="bar cl"><b>{r["clawo"][0]}</b><span>{r["clawo"][1]}</span></div>
<div class="sb"><div class="sbl">{r["sandbox"]}</div>{loop}</div>
<div class="bar sv"><b>{r["supervisor"][0]}</b><span>{r["supervisor"][1]}</span></div></div>
{nxt}
<div class="col">{stage(2)}<div class="pn"><ul>{accept}</ul>
<div class="down"><svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"><path d="M12 5v14M6 13l6 6 6-6"/></svg></div>
<div class="cf">{r["confirm"]}</div>
<div class="rp">{r["repair"]}</div>
<div class="res">{icon("shield", 20, 2.4)} {r["result"]}</div></div></div>
</div>"""


def contribute_figure(t):
    c = t["contrib"]
    check = '<span class="ck"><svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="3.6" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12l5 5 9-10"/></svg></span>'
    term = "".join(f'<div class="tl">{"<i>›</i> " if i == 0 else "<i>&nbsp;</i> "}{x}</div>' for i, x in enumerate(c["term"]))
    visuals = [
        f'<div class="term"><div class="dots"><b></b><b></b><b></b></div>{term}<div class="tok">✓ {c["term_ok"]}</div></div>',
        f'<div class="iss"><div class="it"><span class="dot"></span>{c["issue"]}</div>'
        f'<div class="lbs"><span class="lb">case-submission</span><span class="lb nr">needs-review</span></div>'
        + "".join(f'<div class="fd"><span>{f}</span><i></i></div>' for f in c["fields"]) + '</div>',
        '<div class="rv"><div class="flow"><span class="lb nr">needs-review</span><span class="ar">→</span><span class="lb ac">accepted</span></div>'
        + "".join(f'<div class="cr">{check}<span>{x}</span></div>' for x in c["criteria"]) + '</div>',
        f'<div class="prc"><div class="pt"><span class="pri">⇄</span>{c["pr"]}</div>'
        + "".join(f'<div class="pf"><code>{f}</code><span>{d}</span></div>' for f, d in c["files"])
        + f'<div class="mg"><span class="mgl">{c["merged"]}</span></div>'
        f'<div class="you"><span class="av" style="background:#2a78d6"></span><span class="av" style="background:#1baf7a"></span>'
        f'<span class="av" style="background:#eda100"></span><span class="av me">+</span><b>{c["you"]}</b></div></div>',
    ]
    cols = []
    for i, ((title, desc), vis) in enumerate(zip(c["steps"], visuals)):
        cols.append(f'<div class="st"><div class="hd"><span class="n">{i + 1}</span><b>{title}</b></div><p>{desc}</p>{vis}</div>')
    arrow = f'<div class="nx">{icon("arrow", 20, 2.8)}</div>'
    return f"""<style>{BASE_CSS}
body {{ width:1240px; height:470px; padding:18px; }}
.f {{ width:100%; height:100%; border-radius:28px; border:1.5px solid var(--line); padding:26px 26px;
      background:radial-gradient(circle at 50% 0%, rgba(180,24,45,.06), transparent 45%), linear-gradient(135deg,#FFFFFF 0%,#FCF9F8 55%,#F8F1F0 100%);
      box-shadow:0 14px 30px -18px rgba(90,30,30,.35); display:grid; grid-template-columns:1fr 26px 1fr 26px 1fr 26px 1fr; }}
.st {{ display:flex; flex-direction:column; min-width:0; }}
.hd {{ display:flex; align-items:center; gap:10px; }}
.n {{ width:30px; height:30px; border-radius:10px; background:linear-gradient(135deg,#8C0000,#C2263B); color:#fff; font-weight:800; font-size:16px;
      display:flex; align-items:center; justify-content:center; box-shadow:0 6px 12px -6px rgba(140,0,0,.7); }}
.hd b {{ font-size:21px; font-weight:800; letter-spacing:-.01em; }}
p {{ font-size:14.5px; line-height:1.45; color:var(--muted); margin:10px 0 14px; min-height:63px; }}
.nx {{ display:flex; align-items:center; justify-content:center; color:var(--crimson); padding-top:150px; }}
.term, .iss, .rv, .prc {{ flex:1; border-radius:16px; padding:14px 14px; box-shadow:0 6px 18px -12px rgba(90,30,30,.35); }}
.term {{ background:#221B1C; color:#EDE6E6; font-family:'SFMono-Regular',Menlo,monospace; font-size:13px; display:flex; flex-direction:column; }}
.dots {{ display:flex; gap:6px; margin-bottom:12px; }}
.dots b {{ width:10px; height:10px; border-radius:5px; background:#5A4E4F; }}
.dots b:first-child {{ background:#C2263B; }}
.tl {{ line-height:1.65; white-space:nowrap; }}
.tl i {{ color:#FF8A8A; font-style:normal; font-weight:700; }}
.tok {{ margin-top:auto; align-self:flex-start; font-family:'Plus Jakarta Sans','Noto Sans SC',sans-serif; font-size:13.5px; font-weight:700; color:#fff;
        background:linear-gradient(100deg,#8C0000,#C2263B); border-radius:9px; padding:6px 10px; }}
.iss, .rv, .prc {{ background:#fff; border:1px solid var(--line); }}
.it {{ display:flex; align-items:center; gap:8px; font-size:14px; font-weight:700; color:var(--ink); }}
.dot {{ width:14px; height:14px; border-radius:7px; border:2.5px solid #1a7f37; flex:none; }}
.lbs {{ display:flex; gap:6px; flex-wrap:wrap; margin:9px 0 10px; }}
.lb {{ font-size:12px; font-weight:700; border-radius:999px; padding:3px 9px; background:#EEF2F6; color:#3E5468; border:1px solid #D5DEE7; font-family:'SFMono-Regular',Menlo,monospace; }}
.lb.nr {{ background:#FFF4DB; color:#8A5A00; border-color:#F2DDA8; }}
.lb.ac {{ background:#FBEAEA; color:var(--crimson); border-color:#E7C2C2; }}
.fd {{ display:flex; flex-direction:column; gap:4px; margin-top:8px; }}
.fd span {{ font-size:12px; font-weight:600; color:var(--ink2); }}
.fd i {{ height:8px; border-radius:4px; background:#F1EBEB; }}
.flow {{ display:flex; align-items:center; gap:6px; margin-bottom:14px; }}
.ar {{ color:var(--crimson); font-weight:800; }}
.cr {{ display:flex; gap:9px; font-size:14px; line-height:1.4; color:var(--ink2); margin-top:11px; }}
.ck {{ width:19px; height:19px; flex:none; border-radius:10px; background:#B4182D; display:flex; align-items:center; justify-content:center; margin-top:1px; }}
.pt {{ display:flex; align-items:center; gap:8px; font-size:13.5px; font-weight:700; color:var(--ink); font-family:'SFMono-Regular',Menlo,monospace; }}
.pri {{ width:20px; height:20px; border-radius:6px; background:#EEE7FB; color:#6250d6; font-size:13px; display:flex; align-items:center; justify-content:center; flex:none; font-family:sans-serif; }}
.pf {{ display:flex; justify-content:space-between; align-items:center; margin-top:9px; font-size:12.5px; }}
.pf code {{ color:var(--ink2); font-family:'SFMono-Regular',Menlo,monospace; }}
.pf span {{ color:#1a7f37; font-weight:700; font-family:'SFMono-Regular',Menlo,monospace; }}
.mg {{ margin-top:12px; }}
.mgl {{ font-size:12.5px; font-weight:700; color:#fff; background:#8250df; border-radius:999px; padding:4px 11px; }}
.you {{ display:flex; align-items:center; margin-top:16px; }}
.av {{ width:26px; height:26px; border-radius:13px; border:2.5px solid #fff; margin-right:-8px; }}
.av.me {{ background:linear-gradient(135deg,#8C0000,#C2263B); color:#fff; font-weight:800; font-size:15px; display:flex; align-items:center; justify-content:center; margin-right:10px; }}
.you b {{ font-size:13.5px; color:var(--ink); }}
</style>
<div class="f">{arrow.join(cols)}</div>"""


def shoot(html, lang, width, height, out, transparent=True):
    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False, encoding="utf-8") as f:
        f.write(f'<!doctype html><html lang="{lang}"><head><meta charset="utf-8"></head><body>{html}</body></html>')
        page = f.name
    args = [CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=2",
            f"--window-size={width},{height}", "--virtual-time-budget=6000", "--allow-file-access-from-files",
            f"--screenshot={out}", Path(page).as_uri()]
    if transparent:
        args.insert(1, "--default-background-color=00000000")
    subprocess.run(args, check=True, capture_output=True)
    os.unlink(page)


ONLY = sys.argv[1:]  # e.g. `render.py run` renders only the run figure
want = lambda name: not ONLY or any(name.startswith(o) for o in ONLY)

for key, t in T.items():
    if want("banner"):
        shoot(banner(t), t["lang"], 1360, 400, OUT / f"banner-{key}.png", transparent=False)
        subprocess.run(["sips", "-s", "format", "jpeg", "-s", "formatOptions", "86", str(OUT / f"banner-{key}.png"),
                        "--out", str(OUT / f"banner-{key}.jpg")], check=True, capture_output=True)
        (OUT / f"banner-{key}.png").unlink()
    for name, ic, label, style in t["buttons"]:
        if want(f"button-{name}"):
            shoot(button(t, ic, label, style), t["lang"], 380, 96, OUT / f"button-{name}-{key}.png")
    for kind, ic, eyebrow, title, body, link in t["cards"]:
        if want(f"card-{kind}"):
            shoot(card(t, kind, ic, eyebrow, title, body, link), t["lang"], 1240, 420, OUT / f"card-{kind}-{key}.png")
    if want("run"):
        shoot(run_figure(t), t["lang"], 1240, 660, OUT / f"run-{key}.png")
    if want("contribute"):
        shoot(contribute_figure(t), t["lang"], 1240, 470, OUT / f"contribute-{key}.png")
