# FedX-MAG

This repository is the implementation companion to
`FedX_MAG_formulation_checkpoint_v0_3.tex`. The current code is deliberately
restricted to static Stage 1 (`L=1`). It tests whether a graph encoder trained
on normal traffic can learn a stable normal-support representation without
claiming anomaly discrimination before final held-out evaluation.

## Status as of 2026-11-01

1. **Independent v3b M0 confirmation: PASS.** All four families passed every
   frozen gate on the untouched calibration-normal partition (5 seeds × 25 masks).
   See the [protocol](docs/m0_v3b_confirmation_protocol_2026-10-24.md) and the
   [result](docs/m0_v3b_confirmation_result_2026-10-24.md).
2. **M1 (pre-registered, executed once): mixed.** Normal-only geometry and
   attack-independent thresholds transfer to unseen normal windows. The
   cross-fitted LW objective changes the representation and halves held-out
   energy inflation. The sealed test is **negative**: no learned representation
   detects attack windows (AUROC 0.37-0.47) and none beats handcrafted
   descriptors (0.51-0.55). See the [protocol](docs/m1_protocol_2026-10-24.md)
   and the [result](docs/m1_result_2026-10-25.md), including a post-hoc
   graph-size diagnosis.
3. **Diagnosis and audit-band confirmation (2026-10-25): pooling explains the
   registered graph-score failure.** Attack windows are mostly normal traffic; the pooled graph
   embedding averages the few attack entities away. Scoring the frozen M0
   encoder's node states with a train-normal Mahalanobis model (top-3 node
   mean) reached AUROC 0.81 on the never-scored audit band. It beat
   handcrafted descriptors by +0.24 and was confirmed to benefit from true
   topology and from learning. The L_NS training loss does not help detection.
   Preprocessing and scoring passed bit-exact consistency checks. See the
   [diagnosis](docs/detection_diagnosis_2026-10-25.md) and the
   [confirmation](docs/entity_scoring_confirmation_2026-10-25.md).
4. Every X-IIoTID evaluation partition is now consumed. Further quantitative
   claims on it are exploratory.
5. **Archived confirmed detector:** frozen M0 Graph node states, Ledoit-Wolf covariance
   fitted on train-normal, top-three node energy mean, and thresholds from
   calibration-normal. The reusable detector is implemented in
   `src/fedxmag/scoring.py`; it uses reconstruction-only checkpoints and no
   L_NS optimization. See the [workflow](docs/entity_detector_2026-10-26.md).
6. **Deeper root-cause study (2026-10-26): entity scoring is only a partial repair.**
   Raw-node and flow kNN fusion reached exploratory Test/Audit AUROC 0.930/0.912,
   but capture-disjoint fitting exposed unseen-normal support failure. In a
   controlled commissioning check, adding vetted target-normal references repaired
   ranking where threshold recalibration alone failed. This requires clean target
   data and is not independent confirmation or a deployment guarantee. See the
   [root-cause report and conditional remedy](docs/root_cause_analysis_2026-10-26.md).
7. **Reusable raw-node + flow detector (2026-10-27): implemented, still exploratory.**
   `fedxmag.fusion_detector.RawFlowDetector` packages the root-cause candidate with
   vetted-normal environment commissioning (reference extension plus separate
   recalibration). It reproduces every archived study score with maximum absolute
   difference 0.0, including all five capture folds and the commissioning 2x2. It adds
   no new evidence; the frozen node-LW detector remains the confirmed baseline. See the
   [detector workflow](docs/fusion_detector_2026-10-27.md).
8. **Cross-fitted commissioning calibration (2026-10-27): not supported.** A
   hash-registered check reused the same vetted commissioning windows for calibration
   through time-block cross-fitting. That doubled the calibration windows but lowered
   pooled TPR at the 5% target from 47.5% to 37.1% (capture bootstrap difference
   [-18.2, -2.9] points). Early capture windows inflated the held-out score tail.
   Disjoint, later calibration windows remain the default. See the
   [check](docs/cross_calibration_check_2026-10-27.md).
9. **Calibration-coverage flag and flow candidates (2026-10-27): registered, none selected.**
   Predictions now report `capture_in_calibration`. On the consumed audit band, 4 of 7
   normal windows from captures without calibration windows were false positives,
   versus 5.2% elsewhere. Three flow-channel variants (`activity_flags`, `composition`,
   `window_relative`) were hash-registered with the control before any score was
   computed, then characterized once. None beat the control on every axis; the
   control plus commissioning remained strongest under capture shift. All four carry
   forward unchanged to the next independent protocol. See the
   [flow candidates](docs/flow_variants_2026-10-27.md).
10. **Independent TON_IoT validation (2026-11-01, registered, run once): not validated.**
    A background-contrast design scored each attack-period window with all records
    and with only its label-0 records. Source-only thresholds failed badly (FPR 85.9% at
    the 5% target). Commissioning improved ranking (AUROC 0.944 to 0.975) but missed the
    false-positive target (pooled 10.5%; four captures above 10%): background traffic
    grew within each day after the commissioning period. The detector did not beat a
    record-count baseline (AUROC 0.975 for both; TPR at 5% evaluation FPR 75.1% versus
    96.0%). No flow candidate qualified. TON_IoT is now consumed. See the
    [result](docs/ton_iot_validation_2026-11-01.md) and the
    [protocol](docs/ton_iot_validation_protocol_2026-10-27.md).
11. **TON_IoT false-positive root cause (2026-11-01, post hoc): calibration staleness.**
    Normal background traffic changed within tens of minutes. Thresholds from the 20
    background samples just before each window met the target (TPR 94.9%, FPR 4.8%).
    With 30- and 120-minute delays FPR was 7.1% and 9.9%. Score-size coupling and
    pooled thresholds across captures add to the error. Commissioning slowed staleness
    (9.9% versus 16.9% recalibration-only at 120 minutes). Once FPR was controlled,
    the detector matched the record count (94.7% at 4.8%), and both missed low-volume
    MITM (6.7% versus 4.5%). See the [root cause](docs/ton_iot_fpr_root_cause_2026-11-01.md).
12. **Method development and CIC-IDS2017 protocol (2026-11-01).**
    Exploratory probes on consumed data showed three things. On X-IIoTID, whose attacks are
    low-volume, the detector beats the record count (AUROC 0.930 versus 0.554). Label-free
    alert-gated self-calibration cannot raise its threshold (proposition and TON_IoT FPR
    15.6-39.5%). The detector's thresholds age more slowly than the record count's. The CIC-IDS2017
    pcapng captures were converted label-free to Zeek-style records (`fedxmag.pcap_conn`)
    and labelled from the published schedule (`fedxmag.cic_ids2017`). The protocol tests
    fresh calibration (20 most recent vetted normal windows), detection beyond volume, and
    slower staleness. See the [method development](docs/method_development_2026-11-01.md),
    the [protocol](docs/cic_ids2017_validation_protocol_2026-11-01.md), and the
    [paper outline](docs/paper_outline_2026-11-01.md).
13. **CIC-IDS2017 confirmation (2026-11-01, registered, run once): not validated.**
    - **H1 held:** fresh calibration met the false-positive target (FPR 4.1%; Thursday 4.5%,
      Friday 3.7%).
    - **H2 inconclusive:** the detector led the record count, but the hour-block bootstrap
      lower bound was negative.

      | | Detector | Record count |
      | --- | --- | --- |
      | TPR | 49.8% | 27.4% |
      | AUROC | 0.938 | 0.768 |

      The difference was +22.4 points with a 95% bootstrap interval of [-14.7, +49.7].
    - **Gains and losses by attack type:** the detector won on botnet (72.9% versus 3.4%),
      port scan, and infiltration. It lost on DDoS (4.8% versus 100%). Post hoc, one high
      normal window had set the maximum-of-20 threshold, and the buffer then froze during
      the attack.
    - **H3 not supported**, and label-free alert-gated calibration again failed (FPR 45%).
    - CIC-IDS2017 (Monday, Thursday, Friday) is now consumed. See the
      [result](docs/cic_ids2017_validation_2026-11-01.md).

Publication figures are in `output/figures/{m0,m1,theory,posthoc}` (PDF and
PNG) and tables in `output/tables` (Markdown and LaTeX), produced by
`scripts/make_figures.py`.

## Current scope

- M0: node and edge behavioural reconstruction.
- Pretext controls: mean/constant, local, capacity-matched set-context,
  capacity-matched adjacency-shuffled graph, and edge-aware graph.
- Four candidate families are reported separately: node timing, node
  distribution, edge timing, and edge distribution. The historical v2 audit
  failed timing. The latest v3b confirmation passed all four families.
- Separate primary-target and dependency-closure masks, with graph-local
  corruption budgets.
- Label-independent chronological window construction, graph-balanced
  scaling/loss, and multi-seed, multi-mask admissibility gates.
- Pure-PyTorch edge-aware message passing; PyTorch Geometric is not required.

M1 (cross-fitted Mahalanobis learning) lives in `src/fedxmag/m1/`, outside the
fingerprinted M0 `models/` and `training/` packages, and runs only through the
pre-registered `scripts/run_m1.py` for historical reproduction. The archived
confirmed detection workflow uses frozen M0 node states with score-time Mahalanobis.
Temporal modelling, topology reconstruction, federated learning, and full
explainability are not implemented; top-scoring node localization is available.

The 2026-09-17 audit is superseded. Synthetic reverse-direction observations
incorrectly contributed the connection-row timestamp to timing aggregates, and
the graph model had 27.6x the parameters of the local control. Its checkpoints
and apparent family passes are invalid for scientific inference. See the
[protocol correction](docs/m0_protocol_correction_2026-09-19.md) and the
[superseded audit record](docs/m0_audit_2026-09-17.md).

The corrected 2026-09-19 audit is complete. Edge distribution and node
distribution passed every family-wise gate. Edge timing failed the standardized
MSE ceiling, and node timing failed both equal-capacity topology controls. No
threshold was changed and M1 remained blocked at that point. See the
[corrected M0 audit](docs/m0_audit_2026-09-19.md).

A resolution-aware v3b development repair has also completed on train-normal
and select-normal only. It removes same-second quantization from spacing,
retains collision rate once, and uses a both-endpoint shuffled topology
control. In the seed-17 pilot, Graph beat Local, equal-capacity Set-context,
and equal-capacity Shuffled graph for all four families on each of seven fixed
report masks. This is evidence that the repair direction is viable, not an
admissibility result: select-normal was used for early stopping, the prior
audit was no longer untouched, and M1 remained blocked at that point. See the
[v3 development protocol and results](docs/m0_v3_development_protocol_2026-09-20.md).

## Archived confirmed detection workflow

Build reusable detectors from the five confirmed M0 checkpoints, replay the
archived results, and score standardized graph artifacts:

```powershell
$py = 'C:\Users\user\anaconda3\envs\quantum\python.exe'
$env:PYTHONPATH = 'src'
& $py scripts\run_entity_detector.py build
& $py scripts\run_entity_detector.py replay
& $py scripts\run_entity_detector.py score --seed 17 `
  --input data\processed\xiiotid_stage1_v3b_development\test_attack.pt `
  --predictions output\entity_detector\test_attack_seed17_predictions.json
```

Ready-to-use artifacts are in `output/entity_detector/`. Each seed has its own
detector and calibrated thresholds; reported averages are means of seed metrics,
not ensemble predictions. Score output contains per-window energy, alerts at
1/5/10% target FPR, and the three highest-energy nodes. Inputs must use the
original frozen v3b feature standardizer. Build/replay refuse to overwrite
artifacts; use `--output-dir` for a new output directory.

## Exploratory raw-node + flow detector

The candidate from the root-cause study scores availability-aware raw node features
and per-edge flow context (protocol/service mix, TCP flags, duration, ports, bytes)
with train-normal kNN-20, fuses the select-normal-q95-normalized top-three window
scores by maximum, and calibrates on calibration-normal. A new environment is
commissioned with vetted normal windows only:

```powershell
$py = 'C:\Users\user\anaconda3\envs\quantum\python.exe'
$env:PYTHONPATH = 'src'
& $py scripts\run_fusion_detector.py build
& $py scripts\run_fusion_detector.py replay
& $py scripts\run_fusion_detector.py commission-replay
& $py scripts\run_fusion_detector.py commission --reference ref.pt --calibration cal.pt `
  --raw-csv site.csv --environment site-a --artifact output\fusion_detector_v2\detector_site-a.pt
& $py scripts\run_fusion_detector.py score --detector output\fusion_detector_v2\detector_site-a.pt `
  --input windows.pt --raw-csv site.csv --predictions predictions.json
```

The current artifacts are in `output/fusion_detector_v2/` (format 2). The original
format-1 build in `output/fusion_detector/` still loads, with unknown calibration
coverage. Predictions flag windows whose capture is absent from the normal reference
(`capture_in_reference: false`), the condition under which capture-disjoint stress
tests produced false-positive surges, and windows whose capture did not shape the
thresholds (`capture_in_calibration: false`). Targets lacking calibration support
report `null` instead of a silently capped threshold. `build --flow-variant` selects
one of the registered flow candidates; the default `all` is unchanged.

## Environment

The working interpreter is:

```powershell
C:\Users\user\anaconda3\envs\quantum\python.exe
```

Install the package in editable mode and run the checks:

```powershell
& 'C:\Users\user\anaconda3\envs\quantum\python.exe' -m pip install -e .
& 'C:\Users\user\anaconda3\envs\quantum\python.exe' -m pytest
& 'C:\Users\user\anaconda3\envs\quantum\python.exe' -m ruff check .
```

Run the deterministic synthetic M0 smoke experiment:

```powershell
& 'C:\Users\user\anaconda3\envs\quantum\python.exe' scripts\smoke_m0.py
```

The synthetic experiment validates the software path only. It is not evidence
for the scientific hypothesis.

Reproduce the archived corrected-v2 preparation and normal-only pretext audit:

```powershell
& 'C:\Users\user\anaconda3\envs\quantum\python.exe' scripts\prepare_xiiotid.py `
  --input C:\path\to\XIIoTID.csv `
  --output-dir data\processed\xiiotid_stage1_v2 `
  --config configs\stage1_m0.yaml
& 'C:\Users\user\anaconda3\envs\quantum\python.exe' scripts\run_pretext_audit.py `
  --data-dir data\processed\xiiotid_stage1_v2 `
  --output-dir output\m0\xiiotid_stage1_v2 `
  --config configs\stage1_m0.yaml
```

Run the final v3b development path without opening audit, calibration, or test
artifacts:

```powershell
& 'C:\Users\user\anaconda3\envs\quantum\python.exe' scripts\prepare_xiiotid.py `
  --input C:\path\to\XIIoTID.csv `
  --output-dir data\processed\xiiotid_stage1_v3b_development `
  --config configs\stage1_m0_v3_development.yaml
& 'C:\Users\user\anaconda3\envs\quantum\python.exe' scripts\run_m0_development.py `
  --data-dir data\processed\xiiotid_stage1_v3b_development `
  --output-dir output\m0\xiiotid_stage1_v3b_development `
  --config configs\stage1_m0_v3_development.yaml `
  --seed 17 --device cpu
```

The v3b configuration is explicitly development-only. Do not use it to claim
admissibility through `run_pretext_audit.py`. The official v3b admissibility
decision came from the separately registered confirmation protocol:

```powershell
$py = 'C:\Users\user\anaconda3\envs\quantum\python.exe'
& $py scripts\run_m0_confirmation.py register
& $py scripts\run_m0_confirmation.py train --seed 17   # and 29, 41, 53, 67
& $py scripts\run_m0_confirmation.py evaluate
& $py scripts\run_m1.py register                       # refuses unless M0 passed
& $py scripts\run_m1_queue.py --workers 5              # all arm x seed runs
& $py scripts\run_m1.py geometry                       # normal-only; freezes thresholds
& $py scripts\run_m1.py test                           # single sealed test
& $py scripts\simulate_ns_incentive.py
& $py scripts\posthoc_size_diagnostics.py              # exploratory
```

Every phase refuses to overwrite its outputs or to run after a bound hash has
changed. The project interpreter's numpy/scipy/matplotlib BLAS crashes on matrix
products (Windows `0xc06d007f`), so all linear algebra uses torch. Figures are
rendered with any interpreter that has working numpy and matplotlib, for example:

```powershell
& 'C:\Users\user\anaconda3\python.exe' scripts\make_figures.py m0   # also: m1, theory, posthoc
```

The preparation step emits `preflight.json` with ingestion counts, split drift,
feature availability, shortcut exposure, masking rates, and input/config
fingerprints. It refuses to overwrite artifacts unless `--overwrite` is passed.
Protocol/session/attack-subtype and endpoint metadata are persisted for audit
stratification, but model input tensors never include them. The training audit
uses `select_normal` only for early stopping and the separate `audit_normal`
split for final M0 admissibility decisions.

The current corrected preprocessing artifact is:

- prepared data: `data/processed/xiiotid_stage1_v2`
- corrected M0 output: `output/m0/xiiotid_stage1_v2`

X-IIoTID rows are connection records. A row timestamp is observed only for its
recorded source-to-destination orientation. Destination counters still create a
reverse size/distribution observation, but never a reverse timing observation.
Consequently, timing features are connection-record temporal spacing, not raw
packet IAT or complete flow-stream IAT.

This dataset exposes 44 usable capture dates rather than enough independent
normal-only sessions for a session-held-out five-way split. The frozen primary
protocol therefore uses non-overlapping chronological windows within each
capture date. Window overlap is zero, but all 44 sessions occur in every split;
claims are limited to future-window generalization within observed captures.
A session-held-out experiment remains a required later stress test.

## Research contracts

- [Professor feedback decisions](docs/feedback_decisions.md)
- [Frozen Stage-1 specification](docs/stage1_spec.md)
- [2026-09-19 protocol correction](docs/m0_protocol_correction_2026-09-19.md)
- [Corrected 2026-09-19 M0 audit](docs/m0_audit_2026-09-19.md)
- [2026-09-20 v3 development protocol and results](docs/m0_v3_development_protocol_2026-09-20.md)
- [2026-10-24 v3b independent confirmation protocol](docs/m0_v3b_confirmation_protocol_2026-10-24.md)
  and [result (PASS)](docs/m0_v3b_confirmation_result_2026-10-24.md)
- [2026-10-24 M1 protocol](docs/m1_protocol_2026-10-24.md) and
  [result](docs/m1_result_2026-10-25.md)
- [2026-09-19 storage-integrity incident](docs/storage_integrity_incident_2026-09-19.md)
- [Superseded 2026-09-17 M0 audit](docs/m0_audit_2026-09-17.md)
- [Preferred entity detector and replay](docs/entity_detector_2026-10-26.md)
- [Exploratory raw-node + flow detector and commissioning](docs/fusion_detector_2026-10-27.md)
- [Cross-fitted commissioning calibration check (not supported)](docs/cross_calibration_check_2026-10-27.md)
  and its [protocol](configs/cross_calibration_protocol.yaml)
- [Flow-channel candidates and calibration coverage](docs/flow_variants_2026-10-27.md)
  and the [candidate registration](configs/flow_variant_candidates.yaml)
- [TON_IoT independent validation protocol](docs/ton_iot_validation_protocol_2026-10-27.md)
  ([registered YAML](configs/ton_iot_validation_protocol.yaml)) and
  [result (not validated)](docs/ton_iot_validation_2026-11-01.md)
- [TON_IoT false-positive root cause](docs/ton_iot_fpr_root_cause_2026-11-01.md) and
  [method development](docs/method_development_2026-11-01.md)
- [CIC-IDS2017 confirmation protocol](docs/cic_ids2017_validation_protocol_2026-11-01.md)
  ([YAML](configs/cic_ids2017_validation_protocol.yaml)) and
  [result (not validated)](docs/cic_ids2017_validation_2026-11-01.md)
- [Paper outline](docs/paper_outline_2026-11-01.md)
- [Archived v2 M0 configuration](configs/stage1_m0.yaml)
- [v3b development configuration](configs/stage1_m0_v3_development.yaml)
