"""Summarize completed, independently audited H2/H3 experiments without refitting.

Only a completed full six-run audit is accepted. Integration-smoke results are
never promoted to accuracy findings. Existing output directories are preserved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import product
from pathlib import Path

import numpy as np
import torch

POLICIES = ("fixed", "delayed_10pct", "immediate")
DATASETS = ("cic", "xiiotid", "ton")
KINDS = ("quantum", "quantum_noent", "mlp32", "mlp_budget", "full_mlp64")
H3_KINDS = (
    "ar_ridge",
    "delay_support",
    "esn",
    "gru",
    "quantum",
    "quantum_noent",
    "quantum_zero_history",
    "quantum_shuffled",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def comparison_specs(hypothesis):
    """Fixed comparisons; no best method or threshold is selected on target data."""
    result = []
    if hypothesis == "h2":
        for corruption, head in product(("original", "conditional"), ("scalar", "support")):
            candidate = f"{corruption}_quantum_{head}"
            for reference in ("quantum_noent", "mlp32", "mlp_budget"):
                result.append((candidate, f"{corruption}_{reference}_{head}", "same_input", True))
            result.append(
                (candidate, f"{corruption}_full_mlp64_{head}", "full_input_control", False)
            )
            result.append((candidate, "frozen_dnn20k_m8_reference", "unequal_20k_reference", False))
            result.append(("raw_plus_" + candidate, "raw_m8_reference", "raw_bypass", False))
        for corruption, kind in product(("original", "conditional"), KINDS):
            result.append(
                (f"{corruption}_{kind}_scalar", f"{corruption}_{kind}_support", "head", False)
            )
        for kind, head in product(KINDS, ("scalar", "support")):
            result.append(
                (f"conditional_{kind}_{head}", f"original_{kind}_{head}", "corruption", False)
            )
    elif hypothesis == "h3":
        for mode in (8, 1):
            for reference in H3_KINDS:
                if reference != "quantum":
                    result.append(
                        (
                            f"quantum_m{mode}",
                            f"{reference}_m{mode}",
                            "temporal_primary" if mode == 8 else "mode1_sensitivity",
                            mode == 8,
                        )
                    )
    else:
        raise ValueError("Unknown hypothesis")
    return result


def paired_difference(candidate, reference, labels):
    a, b, y = (np.asarray(v) for v in (candidate, reference, labels))
    require(
        a.ndim == b.ndim == y.ndim == 1 and a.shape == b.shape == y.shape,
        "Paired decisions must have identical vector shapes",
    )
    require(np.isin(a, (0, 1)).all() and np.isin(b, (0, 1)).all(), "Invalid paired flags")
    require(np.isin(y, (0, 1)).all() and len(np.unique(y)) == 2, "Both classes required")
    a, b = a.astype(bool), b.astype(bool)
    attack, normal = y == 1, y == 0
    delta_tpr = float(a[attack].mean() - b[attack].mean())
    delta_fpr = float(a[normal].mean() - b[normal].mean())
    return {
        "added_tp": int((a & ~b & attack).sum()),
        "lost_tp": int((~a & b & attack).sum()),
        "added_fp": int((a & ~b & normal).sum()),
        "removed_fp": int((~a & b & normal).sum()),
        "delta_tpr": delta_tpr,
        "delta_fpr": delta_fpr,
        "delta_tpr_percentage_points": 100 * delta_tpr,
        "delta_fpr_percentage_points": 100 * delta_fpr,
        "candidate_tpr": float(a[attack].mean()),
        "candidate_fpr": float(a[normal].mean()),
        "reference_tpr": float(b[attack].mean()),
        "reference_fpr": float(b[normal].mean()),
        "candidate_fpr_within_5pct": bool(a[normal].mean() <= 0.05 + 1e-12),
    }


def practical_screen(entries):
    """A descriptive development gate, not significance or generalization proof."""
    require(set(entries) == set(DATASETS), "A screen requires all three datasets")
    changes = [entries[dataset]["delta_tpr"] for dataset in DATASETS]
    within = all(entries[dataset]["candidate_fpr"] <= 0.05 + 1e-12 for dataset in DATASETS)
    improved = sum(delta >= 0.02 - 1e-12 for delta in changes)
    not_degraded = min(changes) >= -0.01 - 1e-12
    return {
        "passed": bool(within and improved >= 2 and not_degraded),
        "candidate_fpr_within_5pct_all_three": within,
        "datasets_with_tpr_gain_at_least_2pp": improved,
        "minimum_tpr_change_at_least_minus_1pp": bool(not_degraded),
        "per_dataset": entries,
        "interpretation": (
            "practical development screen only; no statistical or quantum-advantage verdict"
        ),
    }


def training_summary(training):
    output = {}
    for name, entry in training.items():
        history = entry["history"]
        selected_epoch = entry["selected_epoch"]
        chosen = [row for row in history if row["epoch"] == selected_epoch]
        require(len(chosen) == 1, f"Missing selected checkpoint: {name}")
        output[name] = {
            "selected_epoch": selected_epoch,
            "epochs_run": history[-1]["epoch"],
            "initial_selection_bce": history[0]["selection_synthetic_bce"],
            "selected_selection_bce": chosen[0]["selection_synthetic_bce"],
            "selection_bce_change_from_initial": (
                chosen[0]["selection_synthetic_bce"] - history[0]["selection_synthetic_bce"]
            ),
            "parameter_count": entry["parameter_count"],
            "parameter_l2_change": entry["parameter_l2_change"],
            "train_changed_row_fraction": entry["train_corruption_audit"]["changed_row_fraction"],
            "validation_changed_row_fraction": entry["validation_corruption_audit"][
                "changed_row_fraction"
            ],
        }
    return output


def load_verified_audit(path):
    audit = json.loads(path.read_text(encoding="utf-8"))
    require(audit.get("status") == "completed", "Full independent audit has not completed")
    require(
        audit.get("smoke") is False and audit.get("scope") == "full_prepared_target_census",
        "Only full accuracy results are accepted; smoke is not an accuracy estimate",
    )
    runs = audit.get("runs", [])
    expected = set(product(("h2", "h3"), DATASETS))
    identities = [(r["hypothesis"], r["dataset"]) for r in runs]
    require(len(identities) == 6 and set(identities) == expected, "All six full runs are required")
    require(audit.get("total_policy_conditions_checked") == 522, "Incomplete full policy coverage")
    for entry in runs:
        require(
            entry.get("all_decisions_thresholds_counts_rates_exact") is True,
            "A run has not passed independent decision replay",
        )
        require(
            entry.get("all_source_observation_roles_disjoint") is True
            and entry.get("full_target_census_matches_h1") is True,
            "Run scope is not verified",
        )
        for filename in (
            "report.json",
            "registration.json",
            "scores.pt",
            "decisions.pt",
            "source_adapter.pt",
        ):
            artifact = entry["artifacts"][filename]
            require(
                digest(artifact["path"]) == artifact["sha256"],
                f"Changed audited artifact: {filename}",
            )
    return audit


def collect_run(entry):
    artifacts = entry["artifacts"]
    report = json.loads(Path(artifacts["report.json"]["path"]).read_text(encoding="utf-8"))
    registration = json.loads(
        Path(artifacts["registration.json"]["path"]).read_text(encoding="utf-8")
    )
    source = torch.load(artifacts["source_adapter.pt"]["path"], weights_only=True)
    scores = torch.load(artifacts["scores.pt"]["path"], weights_only=True)
    decisions = torch.load(artifacts["decisions.pt"]["path"], weights_only=True)
    require(report["methods"] == entry["methods"], "Audited metrics differ from stored report")
    labels = scores["labels"].numpy()
    comparisons = {}
    for candidate, reference, category, gated in comparison_specs(entry["hypothesis"]):
        key = candidate + "_vs_" + reference
        comparisons[key] = {
            "candidate": candidate,
            "reference": reference,
            "category": category,
            "practical_screen_eligible": gated,
            "policies": {},
        }
        for policy in POLICIES:
            change = paired_difference(
                decisions[candidate + "/" + policy]["flags"].numpy(),
                decisions[reference + "/" + policy]["flags"].numpy(),
                labels,
            )
            for method in (candidate, reference):
                metrics = report["methods"][method][policy]
                side = "candidate" if method == candidate else "reference"
                for rate in ("tpr", "fpr"):
                    require(
                        change[side + "_" + rate] == metrics[rate],
                        "Stored paired rate differs from audited report",
                    )
            comparisons[key]["policies"][policy] = change
    variance = source["adapter"]["explained_variance_ratio"].tolist()
    context = {
        "fit_rows": registration.get("fit_rows"),
        "validation_rows": registration.get("validation_rows"),
        "selection_rows": registration.get("selection_rows"),
        "source_counts": report.get("source_counts"),
        "input_audit": registration.get("input_audit"),
        "pca_explained_variance_ratio": variance,
        "pca_normal_variance_retained": sum(variance),
        "target_cold_start_windows": report.get("target_cold_start_windows"),
        "target_missing_current": report.get("target_missing_current"),
        "prediction_fit": {
            name: spec.get("prediction_fit", []) for name, spec in report["audits"].items()
        },
    }
    return {
        "hypothesis": entry["hypothesis"],
        "dataset": entry["dataset"],
        "run": entry["run"],
        "target_windows": entry["target_windows"],
        "attack_windows": entry["attack_windows"],
        "normal_windows": entry["normal_windows"],
        "methods": report["methods"],
        "fpr_exceeds_5pct": {
            name: {p: values[p]["fpr"] > 0.05 + 1e-12 for p in POLICIES}
            for name, values in report["methods"].items()
        },
        "comparisons": comparisons,
        "source_context": context,
        "training_summary": training_summary(report.get("training", {})),
        "full_training_history": report.get("training", {}),
        "representation_audits": report["audits"],
        "audited_quantum_noent_bootstrap": entry.get("paired_quantum_noent", {}),
        "artifacts": artifacts,
        "elapsed_seconds": report["elapsed_seconds"],
    }


def build_screens(runs):
    output = {}
    by_identity = {(r["hypothesis"], r["dataset"]): r for r in runs}
    for hypothesis in ("h2", "h3"):
        reference_run = by_identity[(hypothesis, "cic")]
        for key, comparison in reference_run["comparisons"].items():
            if not comparison["practical_screen_eligible"]:
                continue
            for policy in POLICIES:
                entries = {
                    dataset: by_identity[(hypothesis, dataset)]["comparisons"][key]["policies"][
                        policy
                    ]
                    for dataset in DATASETS
                }
                output[hypothesis + "/" + key + "/" + policy] = practical_screen(entries)
    return output


def metric_cell(metrics):
    mark = " ⚠" if metrics["fpr"] > 0.05 + 1e-12 else ""
    return f"{100 * metrics['tpr']:.2f} / {100 * metrics['fpr']:.2f}{mark}"


def render_markdown(result):
    lines = [
        "# H2·H3 검산 후 결과 정리",
        "",
        "완료된 여섯 전체 실행과 독립 검산으로부터 생성했다. 모든 수치는 반복 개발 자료의 "
        "CPU 시뮬레이션 결과다. 독립된 새 환경의 일반화, 유한 측정 탐지기 또는 "
        "실장치 양자 우위를 입증하지 않는다.",
        "",
        "표는 TPR% / 실제 FPR%다. ⚠는 FPR>5%를 뜻한다. target 라벨로 FPR를 동일하게 맞춘 "
        "비교가 아니므로 탐지율만으로 우열을 판단하지 않는다. "
        "fixed / delayed_10pct / immediate는 각각 "
        "source 고정 / 정상10% 확인5분 지연 / 현재 판정 후 즉시 정상 확인 조건이다.",
        "",
        "H2의 quantum/noent/MLP32/작은 MLP는 같은8차원 입력을 받는다. full_mlp64는 전체 입력 "
        "대조이며 frozen_dnn20k는 학습20,000행의 별도 참조다. "
        "scalar/support는 같은 학습 checkpoint의 두 추론 점수다. "
        "H3는 새로 분리한 시간 역할의 예측 잔차 비교이며 H2와 직접 같은 과제가 아니다.",
        "",
        "유형별 검출, 공격 이후 정상 오경보·회복 진단, 모든 학습 이력과 paired 경보 변화는 "
        "[findings.json](findings.json)에 보존했다. "
        "특정 자료별 최선 방법을 골라 범용 결과로 합치지 않는다.",
        "",
    ]
    for run in result["runs"]:
        variance = 100 * run["source_context"]["pca_normal_variance_retained"]
        lines += [
            f"## {run['hypothesis'].upper()} / {run['dataset']}",
            "",
            f"공격{run['attack_windows']}창, 정상{run['normal_windows']}창. "
            f"정상 PCA8 분산 보존율 {variance:.2f}%. "
            "이는 공격 정보 보존율이 아니다.",
            "",
        ]
        if run["hypothesis"] == "h3":
            lines += [
                f"정상 source 예제 수: `{run['source_context']['source_counts']}`. "
                f"cold-start {run['source_context']['target_cold_start_windows']}창, "
                f"현재 관측 결측 {run['source_context']['target_missing_current']}창. "
                "최대8모드가 주 설정이고1모드는 민감도 조건이다.",
                "",
            ]
        lines += ["| 방법 | fixed | delayed_10pct | immediate |", "|---|---:|---:|---:|"]
        for name, policies in run["methods"].items():
            lines.append(
                "| " + " | ".join([name, *[metric_cell(policies[p]) for p in POLICIES]]) + " |"
            )
        lines += [
            "",
            "### 사전 지정 paired 비교",
            "",
            "차이는 candidate − reference, 단위는%p다. 경보 변화는 같은 창의 판정으로 계산했다.",
            "",
            "| 비교 | 정책 | ΔTPR | ΔFPR | 추가TP / 손실TP | 추가FP / 제거FP |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for name, comparison in run["comparisons"].items():
            for policy, change in comparison["policies"].items():
                lines.append(
                    f"| {name} | {policy} | {change['delta_tpr_percentage_points']:+.2f} | "
                    f"{change['delta_fpr_percentage_points']:+.2f} | "
                    f"{change['added_tp']} / {change['lost_tp']} | "
                    f"{change['added_fp']} / {change['removed_fp']} |"
                )
        if run["training_summary"]:
            lines += [
                "",
                "### 정상/합성 교란 학습",
                "",
                "BCE는 실제 공격 탐지 정확도가 아니다. epoch0 선택도 그대로 표시한다.",
                "",
                "| 모델 / seed | 선택epoch / 실행epoch | 초기BCE | 선택BCE | 파라미터변화L2 |",
                "|---|---:|---:|---:|---:|",
            ]
            for name, training in run["training_summary"].items():
                lines.append(
                    f"| {name} | {training['selected_epoch']} / {training['epochs_run']} | "
                    f"{training['initial_selection_bce']:.6f} | "
                    f"{training['selected_selection_bce']:.6f} | "
                    f"{training['parameter_l2_change']:.6g} |"
                )
        if run["hypothesis"] == "h3":
            lines += [
                "",
                "### 정상 selection 예측 MSE",
                "",
                "| 방법 | 각 seed의 정상 validation MSE |",
                "|---|---:|",
            ]
            for name, fits in run["source_context"]["prediction_fit"].items():
                values = ", ".join(f"{v['validation_mse']:.6g}" for v in fits)
                lines.append(f"| {name} | {values or '직접 support, 예측기 없음'} |")
            lines += [
                "",
                "### 양자 주 설정의 공격 이후 정상 오경보",
                "",
                "| 정책 | 과거8분 공격 관측이 있는 정상창: FP / 창 | 그 외 정상창: FP / 창 |",
                "|---|---:|---:|",
            ]
            for policy in POLICIES:
                post = run["methods"]["quantum_m8"][policy]["post_attack"]
                recent, quiet = post["normal_with_attack_in_previous8min"], post["other_normal"]
                lines.append(
                    f"| {policy} | {recent['false_alarms']} / {recent['windows']} | "
                    f"{quiet['false_alarms']} / {quiet['windows']} |"
                )
        lines.append("")
    lines += [
        "## 세 자료 공통 개발 진입 기준",
        "",
        "같은 정책에서 candidate의 실제FPR≤5%가 세 자료 모두에서 성립하고, reference 대비 "
        "최소두자료TPR+2%p, 나머지하락≤1%p인지 표시한다. 통계적 유의성 판정이나 "
        "양자 우위의 증명이 아니다. 통과한 조합도 유한 측정과 독립 환경 확인이 필요하다. "
        "통과하지 않은 결과는 현재 설정의 확대 근거가 부족하다는 뜻이며 "
        "모든 양자 모델의 불가능성을 뜻하지 않는다.",
        "",
        "| 비교 / 정책 | 기준 통과 | 전 자료FPR≤5% | +2%p 자료 수 | 최대하락≤1%p |",
        "|---|---|---|---:|---|",
    ]
    for name, screen in result["practical_screens"].items():
        lines.append(
            f"| {name} | {screen['passed']} | {screen['candidate_fpr_within_5pct_all_three']} | "
            f"{screen['datasets_with_tpr_gain_at_least_2pp']} | "
            f"{screen['minimum_tpr_change_at_least_minus_1pp']} |"
        )
    lines += [
        "",
        "얽힘 제거 대비 시간 블록 bootstrap은 원래 독립 검산의 값을 JSON에 그대로 보존했다. "
        "관측된 경보의 조건부 재표집이며 새 사이트의 구간·다중비교 보정·"
        "재표집된 피드백 재실행 결과가 아니다.",
        "",
    ]
    return "\n".join(lines)


def run(args):
    audit_path = Path(args.audit)
    require(audit_path.is_absolute(), "--audit must be an absolute path")
    output = Path(args.output).resolve()
    require(not output.exists(), "Refusing to overwrite an existing findings directory")
    audit = load_verified_audit(audit_path)
    torch.set_num_threads(1)
    runs = [collect_run(entry) for entry in audit["runs"]]
    result = {
        "status": "completed",
        "scope": "full_prepared_target_census",
        "claim": (
            "reused development data; exact CPU simulation; "
            "no independent or quantum-advantage claim"
        ),
        "audit": {"path": str(audit_path), "sha256": digest(audit_path)},
        "summary_source_sha256": digest(__file__),
        "policy_conditions": audit["total_policy_conditions_checked"],
        "runs": runs,
        "practical_screens": build_screens(runs),
    }
    markdown = render_markdown(result)
    output.mkdir(parents=True, exist_ok=False)
    with (output / "findings.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    with (output / "findings.md").open("x", encoding="utf-8") as handle:
        handle.write(markdown)
    print("FINDINGS COMPLETE", output / "findings.md", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--output", required=True)
    run(parser.parse_args())
