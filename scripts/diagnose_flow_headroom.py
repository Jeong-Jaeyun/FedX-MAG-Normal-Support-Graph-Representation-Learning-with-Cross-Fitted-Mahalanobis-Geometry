"""EXPLORATORY HEADROOM: how much detection does the v3b feature schema give up?

Builds per-edge (window, src, dst) summaries of flow fields that the v3b schema
discards by design (protocol/service mix, TCP flags, payload, duration, ports;
the formulation classes several as context-only or shortcut-prone). A
train-band normal-window edge reference is scored with the same entity-level
Ledoit-Wolf Mahalanobis and top-3 aggregation on the consumed test band. This
estimates headroom only; it is not a proposed detector. The audit band is not read.
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagnose_raw_records import SPLITS, WINDOW_NS, assign_bands, read_rows  # noqa: E402

from fedxmag.m1.detection import MahalanobisDetector, auroc  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FLAGS = ("is_syn_only", "FIN or RST", "is_with_payload", "is_pure_ack", "Is_SYN_ACK")
FEATURES = ("log_records", "tcp", "udp", "icmp", "svc_dns", "svc_http", "svc_other",
            *FLAGS, "log_duration", "log_dst_ports", "privileged_port", "log_src_bytes",
            "log_dst_bytes")


def edge_table(rows: dict, mask: np.ndarray) -> dict[tuple[str, int], list[np.ndarray]]:
    columns = rows["numeric_columns"]
    position = {name: columns.index(name) for name in
                (*FLAGS, "Duration", "Des_port", "Scr_bytes", "Des_bytes")}
    grouped: dict[tuple[str, int, str, str], list[int]] = defaultdict(list)
    windows = rows["timestamps"] // WINDOW_NS
    for index in np.flatnonzero(mask):
        key = (rows["sessions"][index], int(windows[index]), rows["sources"][index],
               rows["destinations"][index])
        grouped[key].append(int(index))
    per_window: dict[tuple[str, int], list[np.ndarray]] = defaultdict(list)
    values = rows["values"]
    for (session, window, _, _), indices in grouped.items():
        block = values[indices]

        def mean(name: str, block=block) -> float:
            column = block[:, position[name]]
            column = column[np.isfinite(column)]
            return float(column.mean()) if column.size else 0.0

        protocols = [rows["protocols"][i] for i in indices]
        services = [rows["services"][i] for i in indices]
        ports = block[:, position["Des_port"]]
        ports = ports[np.isfinite(ports)]
        vector = [
            math.log1p(len(indices)),
            protocols.count("tcp") / len(indices),
            protocols.count("udp") / len(indices),
            protocols.count("icmp") / len(indices),
            services.count("dns") / len(indices),
            services.count("http") / len(indices),
            services.count("other") / len(indices),
            *(mean(flag) for flag in FLAGS),
            math.log1p(max(mean("Duration"), 0.0)),
            math.log1p(len(set(ports.tolist()))),
            float(np.mean(ports < 1024)) if ports.size else 0.0,
            math.log1p(max(mean("Scr_bytes"), 0.0)),
            math.log1p(max(mean("Des_bytes"), 0.0)),
        ]
        per_window[(session, window)].append(np.asarray(vector))
    return per_window


def main() -> None:
    torch.set_num_threads(4)
    rows = read_rows(ROOT / "data" / "X-IIoTID" / "X-IIoTID dataset.csv")
    bands = assign_bands(rows["sessions"], rows["timestamps"])
    labels = rows["labels"]
    windows = rows["timestamps"] // WINDOW_NS
    window_has_attack: dict[tuple[str, int], bool] = defaultdict(bool)
    train_or_test = np.isin(bands, [SPLITS.index("train"), SPLITS.index("test")])
    for index in np.flatnonzero(train_or_test):
        key = (rows["sessions"][index], int(windows[index]))
        window_has_attack[key] |= bool(labels[index])
    train_rows = bands == SPLITS.index("train")
    test_rows = bands == SPLITS.index("test")
    train_normal_windows = np.asarray([
        not window_has_attack[(rows["sessions"][i], int(windows[i]))] for i in range(len(labels))
    ]) & train_rows
    reference = edge_table(rows, train_normal_windows)
    test = edge_table(rows, test_rows)
    stacked = torch.tensor(np.concatenate([np.stack(v) for v in reference.values()]))
    detector = MahalanobisDetector.fit(stacked, "ledoit_wolf", standardize=True)
    scores = {key: detector.score(torch.tensor(np.stack(v))).numpy() for key, v in test.items()}
    top3 = {key: float(np.sort(value)[-3:].mean()) for key, value in scores.items()}
    window_info = {(w["session"], w["window_start_ns"] // WINDOW_NS): w for w in json.loads(
        (ROOT / "output" / "diagnosis" / "test_band_windows.json").read_text())}
    normal = np.asarray([top3[k] for k in top3 if window_info[k]["attack_records"] == 0])
    attack = np.asarray([top3[k] for k in top3 if window_info[k]["attack_records"] > 0])
    fraction = np.asarray([window_info[k]["attack_fraction"] for k in top3
                           if window_info[k]["attack_records"] > 0])
    normal_dates = np.asarray([k[0] for k in top3 if window_info[k]["attack_records"] == 0])
    attack_dates = np.asarray([k[0] for k in top3 if window_info[k]["attack_records"] > 0])
    weights, values = [], []
    for date in sorted(set(normal_dates) & set(attack_dates)):
        n, a = normal_dates == date, attack_dates == date
        if n.sum() >= 3 and a.sum() >= 3:
            weights.append(n.sum() * a.sum())
            values.append(auroc(normal[n], attack[a]))
    report = {
        "status": "exploratory_headroom",
        "features": FEATURES,
        "reference_edges": int(stacked.shape[0]),
        "test_windows": {"normal": int(normal.size), "attack": int(attack.size)},
        "auroc_top3": auroc(normal, attack),
        "within_date_auroc_top3": float(np.average(values, weights=weights)),
        "auroc_top3_fraction<0.1": auroc(normal, attack[fraction < 0.1]),
        "auroc_top3_fraction>=0.5": auroc(normal, attack[fraction >= 0.5]),
    }
    (ROOT / "output" / "diagnosis" / "flow_headroom.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
