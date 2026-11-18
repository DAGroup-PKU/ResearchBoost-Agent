"""Render the README banner, buttons and the trial figure with headless Chrome.

The look follows the ResearchBoost-Agent README (same palette, type and layout). Edit the text below, then run from
the repository root (optionally naming assets, e.g. `trial` or `card-judge`):

    python3 docs/assets/readme/render.py

Needs Google Chrome (set CHROME to its binary if it is not at the macOS default path), network access for Google
Fonts, and macOS `sips` for the banner's JPEG. The banner art is docs/assets/teaser.jpg.
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
    "grid": '<rect x="3" y="3" width="7.5" height="7.5" rx="2"/><rect x="13.5" y="3" width="7.5" height="7.5" rx="2"/>'
            '<rect x="3" y="13.5" width="7.5" height="7.5" rx="2"/><rect x="13.5" y="13.5" width="7.5" height="7.5" rx="2"/>',
    "rocket": '<path d="M5 15c-1.5 1.5-2 4-2 6 2 0 4.5-.5 6-2"/><path d="M9 15l-3-3c1-4 4.5-8 12-9-1 7.5-5 11-9 12z"/><circle cx="14.5" cy="9.5" r="1.5"/>',
    "github": '<path d="M9 19c-4.3 1.4-4.3-2.5-6-3m12 5v-3.5c0-1 .1-1.4-.5-2 2.8-.3 5.5-1.4 5.5-6a4.6 4.6 0 0 0-1.3-3.2 4.2 4.2 0 0 0-.1-3.2s-1.1-.3-3.5 1.3a12.3 12.3 0 0 0-6.2 0C6.5 2.8 5.4 3.1 5.4 3.1a4.2 4.2 0 0 0-.1 3.2A4.6 4.6 0 0 0 4 9.5c0 4.6 2.7 5.7 5.5 6-.6.6-.6 1.2-.5 2V21"/>',
    "book": '<path d="M5 4h9a4 4 0 0 1 4 4v12a3 3 0 0 0-3-3H5z"/><path d="M5 4v13"/>',
    "arrow": '<path d="M5 12h14M13 6l6 6-6 6"/>',
}


def icon(name, size=24, stroke=2.2):
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            f'stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round">{ICONS[name]}</svg>')


# Domain colors as in the ResearchBoost-Agent results figures (LLM blue, MLLM green, 3D ink); RL on LLMs in violet.
TASKS = [
    ("#2a78d6", "LLM", "NanoChat D24", "nanochat · 2 GPUs"),
    ("#2a78d6", "LLM", "YOCO-3B", "yoco · 2 GPUs"),
    ("#1baf7a", "MLLM", "LLaVA-1.5 7B", "llava · 2 GPUs"),
    ("#1baf7a", "MLLM", "BAGEL 7B SFT", "bagel_sft · 4 GPUs"),
    ("#1baf7a", "MLLM RL", "R3 Tree-RL", "r3 · 4 GPUs"),
    ("#6250d6", "LLM RL", "GRPO 9B", "slime_grpo · 4 GPUs"),
    ("#6250d6", "LLM RL", "OPD 9B ← 27B", "slime_opd · 4 GPUs"),
    ("#333333", "3D recon", "3DGS truck", "gaussian_splatting · 1 GPU"),
]

T = {
    "en": {
        "lang": "en",
        "eyebrow": "AI4AI · TRAINING-SPEED BENCHMARK",
        "tagline": "How much faster can an agent make real training?",
        "steps": ["Pinned workload", "Agent optimizes", "Frozen judge scores"],
        "buttons": [("leaderboard", "chart", "Leaderboard", "crimson"), ("blog", "book", "Read the blog", "light"),
                    ("run", "rocket", "How to run", "ink"),
                    ("agent", "github", "ResearchBoost-Agent", "light")],
        "cards": [],  # the README has no highlight cards; card() and card_right() stay for reuse
        "domain_note": "workload · task · GPUs",
        "gates": [("0", "Structure", "entry points, data, exact items"), ("1", "Benefit", "clears the speedup bar"),
                  ("2", "Bitwise", "N0: every tensor matches"), ("3", "Statistical", "N1: within 3σ of seed noise")],
        "frozen": "Frozen at startup",
        "pass": "PASS",
        "rule": "Gates run in order and stop at the first failure; only a PASS earns a reward.",
        "cmd": ["harbor run -p tasks -i slime_opd \\", "  -a &lt;agent&gt; -m &lt;model&gt; -n 1"],
        "phases": ["startup preparation · on the trial's GPUs", "agent · time budget not shown",
                   "verifier · full check, network off"],
        "reward": "reward: accepted · speedup",
        "trial": {
            "stages": [("1", "Startup preparation", "before the agent starts"),
                       ("2", "The agent", "its time budget is not shown"),
                       ("3", "The verifier", "same container, network off")],
            "prep": ["Record the reference twice and compare the two records",
                     "Reference trajectories and their seed-to-seed spread",
                     "Time the reference on these GPUs",
                     "The reference's own quick and full checks",
                     "Controls: trees the judge must reject, and must accept"],
            "handoff": "ready · judge read-only",
            "harbor": ("Harbor", "one container · the trial's GPUs · user agent, no credentials"),
            "box": "/submission/&lt;task&gt; · switches.json · /opt/candidate-env",
            "center": ("agent", "any Harbor agent"),
            "loop": [("Profile", "find the hotspot"), ("Edit", "/submission/&lt;task&gt;"), ("Time", "bench.py time"),
                     ("Check", "bench.py check")],
            "deadline": ("Deadline", "every agent process stops · /submission is taken as it is"),
            "verify": ["Full check of /submission", "Candidate timed against the reference", "Gates 0 → 3, first failure stops",
                       "Judge and startup records root-owned, read-only"],
            "outcome": "PASS → accepted 1, speedup",
            "else": "INCORRECT or no_benefit → 0 · INVALID → judge again",
            "result": "Verified speedup",
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
.fade {{ position:absolute; left:600px; top:0; width:140px; height:400px; background:linear-gradient(90deg,var(--bg),rgba(245,240,238,0)); }}
.txt {{ position:absolute; left:64px; top:0; height:400px; display:flex; flex-direction:column; justify-content:center; padding-top:64px; width:660px; }}
.eyebrow {{ font-size:12.5px; letter-spacing:.26em; color:var(--muted); font-weight:600; }}
h1 {{ font-size:50px; font-weight:800; letter-spacing:-.02em; margin:14px 0 12px; line-height:1.05; white-space:nowrap; }}
h1 b {{ color:var(--crimson); font-weight:800; }}
.tag {{ font-size:21px; font-weight:500; font-style:italic; color:var(--ink2); }}
.steps {{ margin-top:26px; display:flex; align-items:center; gap:14px; font-size:16px; font-weight:600; color:var(--ink2); }}
.steps .dot {{ width:6px; height:6px; border-radius:3px; background:var(--crimson); display:inline-block; }}
</style>
<img class="art" src="{ART}"><div class="fade"></div>
<div class="txt"><div class="eyebrow">{t['eyebrow']}</div><h1><b>RSI-Infra</b> Bench</h1>
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
    if kind == "tasks":
        tiles = "".join(
            f'<div class="tile"><div class="th"><span class="sw" style="background:{c}"></span>{dom}</div>'
            f'<div class="big">{name}</div><div class="pj">{sub.split(" · ")[0]}</div><div class="gp">{sub.split(" · ")[1]}</div></div>'
            for c, dom, name, sub in TASKS)
        return f'<div class="grid">{tiles}</div><div class="note">{t["domain_note"]}</div>'
    if kind == "judge":
        gates = '<div class="gar">›</div>'.join(
            f'<div class="gate"><div class="gn">{n}</div><div class="gt">{name}</div><div class="gd">{d}</div></div>'
            for n, name, d in t["gates"])
        return (f'<div class="gates">{gates}</div><div class="jfoot"><div class="frz">❄ {t["frozen"]}</div>'
                f'<div class="verd"><span class="vx">no_benefit</span><span class="vx">INCORRECT</span>'
                f'<div class="pass">{icon("shield", 18, 2.4)} {t["pass"]}</div></div></div><div class="rule">{t["rule"]}</div>')
    if kind == "harbor":
        lines = "".join(f'<div class="ln">{"<span class=p>$</span> " if i == 0 else "<span class=p>&nbsp;</span> "}{l}</div>'
                        for i, l in enumerate(t["cmd"]))
        phases = "".join(f'<div class="ask"><span class="q">{i + 1}</span>{a}</div>' for i, a in enumerate(t["phases"]))
        return (f'<div class="term"><div class="bar"><i></i><i></i><i></i></div>{lines}<div class="asks">{phases}</div>'
                f'<div class="ok">{icon("shield", 16, 2.4)} {t["reward"]}</div></div>')


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
/* tasks */
.grid {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); grid-template-rows:1fr 1fr; gap:10px; height:250px; }}
.tile {{ border-radius:14px; border:1px solid var(--line); background:#fff; padding:14px 12px 12px; display:flex; flex-direction:column; min-width:0; }}
.th {{ display:flex; align-items:center; gap:6px; font-size:12px; font-weight:600; color:var(--muted); white-space:nowrap; }}
.sw {{ width:12px; height:12px; border-radius:4px; flex:none; }}
.big {{ font-size:16.5px; font-weight:800; letter-spacing:-.02em; margin-top:10px; white-space:nowrap; }}
.pj {{ font-size:10.5px; color:var(--ink2); font-weight:600; white-space:nowrap; margin-top:auto; font-family:'SFMono-Regular',Menlo,monospace; letter-spacing:-.03em; }}
.gp {{ font-size:12px; color:var(--muted); font-weight:600; margin-top:3px; }}
.note {{ position:absolute; right:24px; bottom:12px; font-size:12px; color:var(--muted); letter-spacing:.06em; }}
/* harbor */
.r:has(.term) {{ background:#221B1C; border-color:#221B1C; padding:0; overflow:hidden; }}
.term {{ color:#EDE6E6; font-family:'SFMono-Regular',Menlo,Consolas,monospace; font-size:15px; }}
.bar {{ height:34px; background:#2E2627; display:flex; align-items:center; gap:7px; padding-left:16px; margin-bottom:16px; }}
.bar i {{ width:11px; height:11px; border-radius:6px; background:#5A4E4F; display:block; }}
.bar i:first-child {{ background:#C2263B; }}
.ln {{ padding:0 22px; line-height:1.7; white-space:nowrap; }}
.p {{ color:#FF8A8A; font-weight:700; }}
.asks {{ display:flex; flex-direction:column; gap:7px; margin:16px 22px 0; font-family:'Plus Jakarta Sans','Noto Sans SC',sans-serif; }}
.ask {{ align-self:flex-start; background:#3A3031; border:1px solid #4D4142; border-radius:10px; padding:6px 12px; font-size:14px; display:flex; gap:9px; align-items:center; }}
.q {{ width:18px; height:18px; border-radius:9px; background:#C2263B; color:#fff; font-size:11px; font-weight:800; display:flex; align-items:center; justify-content:center; }}
.ok {{ position:absolute; right:20px; bottom:16px; font-family:'Plus Jakarta Sans','Noto Sans SC',sans-serif; font-size:14px; font-weight:700; color:#fff;
       background:linear-gradient(100deg,#8C0000,#C2263B); border-radius:10px; padding:7px 12px; display:flex; align-items:center; gap:7px; }}
/* judge */
.gates {{ display:flex; align-items:stretch; gap:6px; margin-top:6px; }}
.gate {{ flex:1; border-radius:14px; background:#fff; border:1px solid var(--line); padding:20px 8px 18px; text-align:center; }}
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
</style>
<div class="c"><div class="l"><div class="eh"><div class="ic">{icon(ic, 28)}</div><div class="ey">{eyebrow}</div></div>
<h2>{title}</h2><p>{body}</p><div class="lk">{link} {icon('arrow', 18, 2.6)}</div></div>
<div class="r">{card_right(kind, t)}</div></div>"""


def trial_figure(t):
    r = t["trial"]

    def stage(i):
        n, title, sub = r["stages"][i]
        return f'<div class="sh"><span class="sn">{n}</span><div><b>{title}</b><span>{sub}</span></div></div>'

    check = '<span class="ck"><svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="3.4" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12l5 5 9-10"/></svg></span>'
    prep = "".join(f'<li>{check}<span>{x}</span></li>' for x in r["prep"])
    verify = "".join(f'<li>{check}<span>{x}</span></li>' for x in r["verify"])
    # the agent's loop: five nodes on an ellipse, clockwise from the top, with arcs between them
    W, H, cx, cy, rx, ry = 520, 330, 260, 166, 168, 122
    angles = [-90, 0, 90, 180]
    pt = lambda a: (cx + rx * math.cos(math.radians(a)), cy + ry * math.sin(math.radians(a)))
    arcs = ""
    gap = {-90: 33, 0: 22, 90: 33, 180: 22}  # degrees kept clear around each node (top and bottom nodes are wide)
    for i, a0 in enumerate(angles):
        a1 = angles[(i + 1) % len(angles)] + (360 if i == len(angles) - 1 else 0)
        (x0, y0), (x1, y1) = pt(a0 + gap[a0]), pt(a1 - gap[angles[(i + 1) % len(angles)]])
        arcs += f'<path d="M{x0:.1f} {y0:.1f}A{rx} {ry} 0 0 1 {x1:.1f} {y1:.1f}" marker-end="url(#ah)"/>'
    nodes = "".join(
        f'<div class="nd" style="left:{pt(a)[0]:.0f}px;top:{pt(a)[1]:.0f}px"><b>{name}</b><span>{d}</span></div>'
        for a, (name, d) in zip(angles, r["loop"]))
    loop = (f'<div class="loop" style="width:{W}px;height:{H}px"><svg width="{W}" height="{H}">'
            f'<defs><marker id="ah" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto"><path d="M0 0L8 4L0 8Z" fill="#B4182D"/></marker></defs>'
            f'<g fill="none" stroke="#D9AEB0" stroke-width="2.4">{arcs}</g></svg>{nodes}'
            f'<div class="ctr"><b>{r["center"][0]}</b><span>{r["center"][1]}</span></div></div>')
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
.sbl {{ position:absolute; left:14px; top:-11px; background:#FBF6F5; padding:0 8px; font-size:13px; font-weight:700; color:#9C7D7F; letter-spacing:.01em; font-family:'SFMono-Regular',Menlo,monospace; }}
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
.cf {{ font-size:14.5px; color:var(--ink2); text-align:center; line-height:1.4; font-weight:700; }}
.rp {{ margin-top:auto; margin-bottom:12px; font-size:14px; color:var(--muted); text-align:center; line-height:1.4; }}
.res {{ font-size:17px; font-weight:800; color:#fff; background:linear-gradient(100deg,#8C0000,#C2263B); border-radius:14px; padding:13px 14px;
        display:flex; align-items:center; justify-content:center; gap:9px; box-shadow:0 10px 20px -10px rgba(140,0,0,.7); }}
</style>
<div class="f">
<div class="col">{stage(0)}<div class="pn"><ul>{prep}</ul><div class="ho">{r["handoff"]}</div></div></div>
{nxt}
<div class="col">{stage(1)}
<div class="bar cl"><b>{r["harbor"][0]}</b><span>{r["harbor"][1]}</span></div>
<div class="sb"><div class="sbl">{r["box"]}</div>{loop}</div>
<div class="bar sv"><b>{r["deadline"][0]}</b><span>{r["deadline"][1]}</span></div></div>
{nxt}
<div class="col">{stage(2)}<div class="pn"><ul>{verify}</ul>
<div class="down"><svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"><path d="M12 5v14M6 13l6 6 6-6"/></svg></div>
<div class="cf">{r["outcome"]}</div>
<div class="rp">{r["else"]}</div>
<div class="res">{icon("shield", 20, 2.4)} {r["result"]}</div></div></div>
</div>"""


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


ONLY = sys.argv[1:]  # e.g. `render.py trial` renders only the trial figure
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
    if want("trial"):
        shoot(trial_figure(t), t["lang"], 1240, 660, OUT / f"trial-{key}.png")
