"""Source-only continuation diagnostic; never scores target observations.

Starts from the stored CIC original-corruption seed17 checkpoint. AdamW moments
are reset, so this is an optimization-capacity probe, not an exact training resume.
The fixed normal adapter, source roles, validation corruption and LR are retained.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from fedxmag_next.contrastive_energy import corrupt
from fedxmag_next.experiments import sha256, write_json
from fedxmag_next.h1_classical_features import SourcePCA
from fedxmag_next.h2_models import ClassicalMLPEnergy, ReuploadingQuantumEnergy
from fedxmag_next.record_experiment import corruption_groups

KINDS = ("quantum", "quantum_noent", "mlp32")


def negative(raw, groups, seed):
    return corrupt(raw, groups, torch.Generator().manual_seed(seed))


def construct(kind, seed=17):
    if kind in ("quantum", "quantum_noent"):
        return ReuploadingQuantumEnergy(seed=seed, entangled=kind == "quantum")
    if kind == "mlp32":
        return ClassicalMLPEnergy(seed=seed)
    raise ValueError("Unsupported source diagnostic model")


@torch.no_grad()
def validation_bce(model, positive, negative_values):
    total, count = 0.0, 0
    for values, label in ((positive, 0.0), (negative_values, 1.0)):
        for part in values.split(512):
            logits = model(part)
            if not torch.isfinite(logits).all():
                raise ValueError("Nonfinite source validation logits")
            total += float(
                nn.functional.binary_cross_entropy_with_logits(
                    logits, torch.full_like(logits, label), reduction="sum"
                )
            )
            count += len(part)
    if not count or not math.isfinite(total):
        raise ValueError("Invalid source validation BCE")
    return total / count


@torch.no_grad()
def feature_summary(model, values):
    features = torch.cat([model.embed(v) for v in values.split(512)])
    if not torch.isfinite(features).all():
        raise ValueError("Nonfinite source representation")
    std = features.std(0, unbiased=False)
    return {
        "dimension": features.shape[1],
        "finite": True,
        "std_min": float(std.min()),
        "std_median": float(std.median()),
        "std_max": float(std.max()),
        "near_constant_below_1e_6": int((std < 1e-6).sum()),
    }


def continue_source(
    model,
    adapter,
    raw_train,
    raw_validation,
    groups,
    *,
    seed=17,
    epoch_offset=30,
    epochs=60,
    on_epoch=None,
):
    if epochs < 1 or epoch_offset < 0:
        raise ValueError("Positive continuation epochs and nonnegative offset required")
    positive = adapter.transform(raw_train)
    validation = adapter.transform(raw_validation)
    fixed_negative = adapter.transform(negative(raw_validation, groups, 7189))
    fresh_negative = adapter.transform(negative(raw_validation, groups, 99173))
    initial_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    before_features = feature_summary(model, positive)
    initial = validation_bce(model, validation, fixed_negative)
    fresh_initial = validation_bce(model, validation, fresh_negative)
    best, best_epoch, best_state = initial, 0, initial_state
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    order_rng = torch.Generator().manual_seed(seed)
    for _ in range(epoch_offset):
        torch.randperm(len(positive), generator=order_rng)
    history = [
        {"extra_epoch": 0, "absolute_epoch": epoch_offset, "selection_synthetic_bce": initial}
    ]
    for extra_epoch in range(1, epochs + 1):
        absolute_epoch = epoch_offset + extra_epoch
        synthetic = adapter.transform(negative(raw_train, groups, seed * 1000 + absolute_epoch))
        losses, gradient_square, gradient_count = [], {}, {}
        model.train()
        for ids in torch.randperm(len(positive), generator=order_rng).split(256):
            values = torch.cat((positive[ids], synthetic[ids]))
            labels = torch.cat((torch.zeros(len(ids)), torch.ones(len(ids)))).double()
            loss = nn.functional.binary_cross_entropy_with_logits(model(values), labels)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite continuation source loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            for name, parameter in model.named_parameters():
                if parameter.grad is None:
                    raise ValueError(f"Source parameter has no gradient: {name}")
                if not torch.isfinite(parameter.grad).all():
                    raise ValueError(f"Nonfinite source parameter gradient: {name}")
                group = (
                    "angles"
                    if name == "angles"
                    else "readout"
                    if name.startswith("readout.")
                    else "encoder"
                )
                gradient_square[group] = gradient_square.get(group, 0.0) + float(
                    parameter.grad.square().sum()
                )
                gradient_count[group] = gradient_count.get(group, 0) + parameter.grad.numel()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        value = validation_bce(model, validation, fixed_negative)
        entry = {
            "extra_epoch": extra_epoch,
            "absolute_epoch": absolute_epoch,
            "train_bce": float(np.mean(losses)),
            "selection_synthetic_bce": value,
            "unclipped_gradient_rms": {
                g: math.sqrt(v / gradient_count[g]) for g, v in gradient_square.items()
            },
        }
        history.append(entry)
        if value < best:
            best, best_epoch = value, extra_epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if on_epoch is not None:
            on_epoch(entry)
    last_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    movement = math.sqrt(
        sum(
            float((best_state[n] - initial_state[n]).square().sum())
            for n, _ in model.named_parameters()
        )
    )
    return {
        "initial_bce": initial,
        "best_bce": best,
        "best_extra_epoch": best_epoch,
        "best_absolute_epoch": epoch_offset + best_epoch,
        "improvement": initial - best,
        "fresh_validation_negative_seed": 99173,
        "fresh_negative_bce_initial": fresh_initial,
        "fresh_negative_bce_at_selected_checkpoint": validation_bce(
            model, validation, fresh_negative
        ),
        "source_feature_summary_initial": before_features,
        "source_feature_summary_selected": feature_summary(model, positive),
        "selected_parameter_l2_movement": movement,
        "history": history,
        "selected_state": best_state,
        "last_state": last_state,
    }


def load_source(root):
    """Only source indices/features/schema are materialized; no labels/census read."""
    run = root / "New_Model/runs/round234_h2_cic_v1"
    registration = json.loads((run / "registration.json").read_text(encoding="utf-8"))
    saved = torch.load(run / "source_adapter.pt", weights_only=True)
    fit, validation = saved["fit_indices"].numpy(), saved["validation_indices"].numpy()
    roles = saved["source_role_indices"]
    if (
        len(fit) != 4096
        or len(validation) != 4096
        or np.intersect1d(fit, validation).size
        or not np.isin(fit, roles["train"].numpy()).all()
        or not np.isin(validation, roles["selection"].numpy()).all()
    ):
        raise ValueError("Source-only role contract changed")
    source_path = root / "New_Model/artifacts/cic_all_legacy_http_v1/records.npz"
    if sha256(source_path) != registration["data_sha256"]:
        raise ValueError("Prepared source data changed")
    with np.load(source_path) as payload:
        # Compressed NPZ must decompress the feature array; only source rows are
        # retained. No label/time/census/target array is requested from the file.
        features = payload["features"]
        train = torch.from_numpy(features[fit].copy())
        selected = torch.from_numpy(features[validation].copy())
        names = payload["feature_names"].tolist()
    return (
        train,
        selected,
        names,
        SourcePCA.from_state_dict(saved["adapter"]),
        {
            "source_data": str(source_path),
            "source_data_sha256": registration["data_sha256"],
            "source_adapter_sha256": sha256(run / "source_adapter.pt"),
            "fit_indices_sha256": __import__("hashlib").sha256(fit.tobytes()).hexdigest(),
            "validation_indices_sha256": __import__("hashlib")
            .sha256(validation.tobytes())
            .hexdigest(),
            "fit_rows": len(fit),
            "validation_rows": len(validation),
            "roles_disjoint": True,
            "target_arrays_read": False,
        },
    )


def run(args):
    torch.set_num_threads(args.workers)
    torch.set_num_interop_threads(1)
    root, out = Path(args.root).resolve(), Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    train, selected, names, adapter, source = load_source(root)
    groups = corruption_groups(names)
    write_json(
        out / "registration.json",
        {
            "status": "registered",
            "args": vars(args),
            "source": source,
            "script_sha256": sha256(Path(__file__)),
            "implementation_source": str(Path(__file__)),
            "kinds": KINDS,
            "seed": 17,
            "corruption": "original",
            "lr": 0.001,
            "weight_decay": 0.001,
            "optimizer_state": "fresh AdamW moments, not exact resume",
            "selection": "same source normal-vs-synthetic BCE, fixed validation seed7189",
            "fresh_negative_seed99173_scope": "diagnostic only, never used for selection",
            "early_stopping": False,
            "extra_epochs": args.epochs,
            "scope": "source-only optimization capacity, no target scoring or hypothesis winner",
        },
    )
    report = {"status": "running", "source_only": True, "target_evaluation": False, "models": {}}
    for kind in KINDS:
        path = root / f"New_Model/runs/round234_h2_cic_v1/original_{kind}_detector.pt"
        saved = torch.load(path, weights_only=True)
        if saved["feature_names"] != names:
            raise ValueError("Source schema differs from saved model")
        checkpoint = next(x for x in saved["seeds"] if x["seed"] == 17)
        model = construct(kind)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        selected_epoch = checkpoint["training"]["selected_epoch"]

        def callback(entry, current_kind=kind):
            write_json(out / f"{current_kind}_epoch_{entry['extra_epoch']:03d}.json", entry)
            if entry["extra_epoch"] == 1 or entry["extra_epoch"] % 10 == 0:
                print(
                    current_kind,
                    entry["absolute_epoch"],
                    entry["selection_synthetic_bce"],
                    flush=True,
                )

        result = continue_source(
            model,
            adapter,
            train,
            selected,
            groups,
            epoch_offset=selected_epoch,
            epochs=args.epochs,
            on_epoch=callback,
        )
        expected = next(
            x["selection_synthetic_bce"]
            for x in checkpoint["training"]["history"]
            if x["epoch"] == selected_epoch
        )
        error = abs(expected - result["initial_bce"])
        if error > 1e-9:
            raise ValueError(f"Reloaded source validation differs: {kind}: {error}")
        for label in ("selected", "last"):
            torch.save(
                {
                    "kind": kind,
                    "seed": 17,
                    "adapter": adapter.state_dict(),
                    "model": result.pop(label + "_state"),
                    "scope": "source-only diagnostic",
                },
                out / f"{kind}_{label}_source_model.pt",
            )
        result.update(
            input_checkpoint=str(path),
            input_checkpoint_sha256=sha256(path),
            original_selected_epoch=selected_epoch,
            reload_validation_abs_error=error,
        )
        report["models"][kind] = result
        write_json(out / f"completed_{kind}.json", result)
    report.update(status="completed", elapsed_seconds=time.perf_counter() - started)
    write_json(out / "report.json", report)
    print("SOURCE MATURITY PROBE COMPLETE", report["elapsed_seconds"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=60)
    run(parser.parse_args())
