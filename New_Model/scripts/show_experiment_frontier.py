"""Print measured operating points without selecting inherited legacy paths."""

import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("runs", nargs="+")
parser.add_argument("--contains", default="")
args = parser.parse_args()
for run in args.runs:
    for suffix in ("", "capacity50", "capacity59"):
        path = Path(run) / suffix / "report.json"
        if not path.exists():
            print(json.dumps({"path": str(path), "status": "not_written_yet"}))
            continue
        report = json.loads(path.read_text())
        entries = [
            {"method": name, "tpr": value["tpr"], "fpr": value["fpr"]}
            for name, value in report["methods"].items()
            if args.contains in name
            and value.get("tpr") is not None
            and value.get("fpr") is not None
            and value.get("coverage") == 1
        ]
        entries.sort(key=lambda value: (-value["tpr"], value["fpr"]))
        print(
            json.dumps(
                {
                    "path": str(path),
                    "best_under5": [value for value in entries if value["fpr"] <= 0.05][:3],
                    "highest_tpr": entries[:1],
                    "goal_hits": [
                        value for value in entries if value["fpr"] <= 0.05 and value["tpr"] >= 0.95
                    ],
                }
            )
        )
