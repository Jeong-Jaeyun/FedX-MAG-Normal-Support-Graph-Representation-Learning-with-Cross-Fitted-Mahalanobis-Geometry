"""Pre-registered M1 experiment: register, train, geometry, and a single test.

``register`` refuses to run unless the independent v3b M0 confirmation passed.
``train`` opens only train-normal and select-normal; ``geometry`` additionally
opens calibration-normal and freezes every detector and threshold; ``test``
opens test-normal and test-attack exactly once.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fedxmag.m1.experiment import geometry, load_context, register, sealed_test, train_arm

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("register", "train", "geometry", "test"))
    parser.add_argument(
        "--protocol", type=Path, default=PROJECT_ROOT / "configs" / "m1_protocol.yaml"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "output" / "m1" / "xiiotid_stage1_m1"
    )
    parser.add_argument("--arm")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    context = load_context(args.protocol, PROJECT_ROOT, args.output_dir)
    if args.phase == "register":
        result = register(context)
        print(json.dumps({key: result[key] for key in ("status", "protocol_sha256")}))
    elif args.phase == "train":
        if args.arm is None or args.seed is None:
            raise ValueError("train requires --arm and --seed")
        print(json.dumps(train_arm(context, args.arm, args.seed, args.threads, args.device)))
    elif args.phase == "geometry":
        geometry(context)
        print(json.dumps({"geometry": "complete"}))
    else:
        report = sealed_test(context)
        print(
            json.dumps(
                {
                    name: entry["aggregate_over_members"]["auroc"]
                    for name, entry in report["methods"].items()
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
