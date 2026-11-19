"""Render every registered H1 method from the independently audited summary."""

import json
from pathlib import Path

root = Path(__file__).resolve().parents[2]
folder = root / "New_Model/reports/h1_pilot_2026-11-15"
summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
lines = [
    "# H1 전체 결과표",
    "",
    "각 셀은 TP/FP (TPR% / FPR%)이다. 정상4,096행, 같은 source 역할과 top3, "
    "교정50창을 사용했다. 최대8모드가 주 설정이고1모드는 민감도 분석이다.",
    "",
    "frozen_dnn20k는20,000행 학습의 별도 참조이다. raw_plus는 행 점수 "
    "정규화 후max 결합이며, 단독 탐지기들의 경보OR가 아니다. 즉시는 현재 판정 "
    "후 정상 확인, 지연10%는5분 뒤 확인을 모사한다. 모든 자료는 개발 자료다.",
    "",
]
for dataset, data in summary["datasets"].items():
    lines += [
        f"## {dataset}",
        "",
        f"공격 {data['attack_windows']}창, 정상 {data['normal_windows']}창.",
        "",
        "| 방법 | 고정 | 지연10% | 즉시 |",
        "|---|---:|---:|---:|",
    ]
    for name, values in data["methods"].items():
        cells = []
        for policy in ("fixed", "delayed_10pct", "immediate"):
            m = values[policy]
            cells.append(f"{m['tp']}/{m['fp']} ({100 * m['tpr']:.2f} / {100 * m['fpr']:.2f})")
        lines.append("| " + " | ".join([name, *cells]) + " |")
    lines.append("")
with (folder / "all_methods.md").open("x", encoding="utf-8") as handle:
    handle.write("\n".join(lines) + "\n")
print(folder / "all_methods.md")
