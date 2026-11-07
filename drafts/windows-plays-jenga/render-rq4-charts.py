"""Recreate RQ4 counts from Kim et al., CCS 2025, §6.3 (pp. 3909–3910).

Source: https://insuyun.github.io/pubs/2025/kim%3Ajenga.pdf#page=10
These charts are review illustrations, not figures from the paper.
Run with Python + matplotlib; pass a Korean font with --font on other systems.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--font", default="/System/Library/Fonts/AppleSDGothicNeo.ttc")
args = parser.parse_args()
font_manager.fontManager.addfont(args.font)
plt.rcParams.update({
    "font.family": font_manager.FontProperties(fname=args.font).get_name(),
    "font.size": 16,
    "svg.fonttype": "path",
    "svg.hashsalt": "jenga-rq4",
    "axes.unicode_minus": False,
})

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "content/windows-plays-jenga"
N = 21
SOURCE = "출처: Kim 외, CCS 2025, §6.3 · 본문 수치 재구성"


def chart(filename, title, count, positive, negative, theme="light"):
    palette = {
        "light": ("#202934", "#596575", "#426c9d", "#e4e9ef"),
        "dark": ("#e6e6e6", "#b9c0cb", "#91b8e7", "#46515e"),
    }
    text, muted, accent, remainder = palette[theme]
    if theme == "dark":
        filename += "-dark"
    fig = plt.figure(figsize=(4.6, 5.3), facecolor="none")
    ax = fig.add_axes([0.10, 0.30, 0.80, 0.54])
    fig.text(0.5, 0.91, title, ha="center", fontsize=20,
             weight="bold", color=text)
    # Both slices partition the same 21 submissions; a zero is an empty slice.
    ax.pie([count, N - count], startangle=90, counterclock=False,
           colors=[accent, remainder],
           wedgeprops={"width": 0.22, "edgecolor": "none", "linewidth": 0})
    ax.text(0, 0.10, f"{count / N * 100:.1f}%" if count else "0%",
            ha="center", va="center", fontsize=29, weight="bold", color=text)
    ax.text(0, -0.28, f"{positive} {count} / {N}명", ha="center",
            va="center", fontsize=14, color=muted)
    fig.text(0.5, 0.24, f"{positive}  {count}명", ha="center", fontsize=16, color=accent)
    fig.text(0.5, 0.17, f"{negative}  {N - count}명", ha="center", fontsize=16, color=muted)
    fig.text(0.5, 0.065, "전체 21명 기준 · 원문 §6.3 수치 재구성",
             ha="center", fontsize=11, color=muted)
    svg_path = OUT / f"{filename}.svg"
    fig.savefig(svg_path, transparent=True, metadata={"Date": None,
                "Description": f"{title}. {positive} {count}명, {negative} {N-count}명. {SOURCE}"})
    svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n")
    fig.savefig(Path("/tmp") / f"{filename}.png", dpi=140, transparent=True)
    plt.close(fig)


for theme in ("light", "dark"):
    chart("rq4-implementation", "기능 요구사항 충족", 19, "충족", "미충족", theme)
    chart("rq4-security", "보안 평가 결과", 0, "안전한 구현", "취약점 발견", theme)
