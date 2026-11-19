# FedX-MAG Next: ongoing normal-only deep-energy experiments

`FedX-MAG Next` is an isolated replacement experiment.  It does not modify the
archived implementation or reuse an already-consumed evaluation split as a new
claim.

## Measured status (2026-11-05)

For the current architecture and measured whole-window results, start with
[`docs/STRUCTURE_REVISION.md`](docs/STRUCTURE_REVISION.md). Continuation state is
[`docs/LATEST_STATE.md`](docs/LATEST_STATE.md). Improvements on X-IIoTID and CIC
are measured. Overall TPR/FPR is the user's priority; subtype weaknesses remain
reported. TON has an overall95%/5% deep candidate, while CIC remains near72% TPR.
The research goal is active.

Audited normal-prototype candidates and complete initial model states are in
[`candidates/prototype_structure_v1/registry.json`](candidates/prototype_structure_v1/registry.json).
Recorded points include TON97.37%/4.42%, full-CIC80.00%/4.14%, and
X-IIoTID92.92%/4.71% (TPR/FPR). These use verified-normal feedback after prediction;
the TON point uses raw geometry. Configurations differ and the data are reused
development data. See the November8 experiment log for matched controls.

The initial masked-reconstruction GNN proposal below was actually trained and
did **not** beat the raw-feature baseline. It is retained as a failed control.
The current candidates train a DNN against group-permuted normal observations,
then fit shrinkage-Mahalanobis support to its deep features. Mean and median
ensembles, raw-feature controls, capture-disjoint refitting, commissioning, and
causal calibration replays are measured separately.

All supplied datasets are available for iterative development. Reuse is recorded
as development evidence; it is not an independent confirmation claim. The
ongoing results and remaining failures are in
[`docs/EXPERIMENT_LOG_2026-11-05.md`](docs/EXPERIMENT_LOG_2026-11-05.md).
The latest observation, personalized-support, and temporal-state work is recorded
in [`docs/EXPERIMENT_LOG_2026-11-08.md`](docs/EXPERIMENT_LOG_2026-11-08.md).

Runnable experiment modules (use `--help` for arguments):

* `fedxmag_next.experiments`: initial GNN and raw baselines.
* `fedxmag_next.flow_experiment`: invertible DNN energy / exact likelihood.
* `fedxmag_next.contrastive_energy`: normal-only synthetic dependency learning.
* `fedxmag_next.capture_experiment`: five capture-disjoint refits.
* `fedxmag_next.commission_experiment`: paired vetted-normal adaptation.
* `fedxmag_next.feature_energy_experiment`: Mahalanobis on contrastive deep features.
* `fedxmag_next.second_dataset`: actual full-window CIC / TON evaluation.
* `fedxmag_next.record_data`: six-process, label-independent actual-row preparation.
* `fedxmag_next.record_experiment`: completed-connection energy scoring.
* `fedxmag_next.record_calibration`: strictly causal verified-normal feedback.
* `fedxmag_next.temporal_context` / `native_context`: label-free, strictly past behavior.
* `fedxmag_next.source_only`: disjoint normal source roles with temporal purges.
* `fedxmag_next.window_bridge`: actual whole-window labels and frozen source controls.
* `fedxmag_next.node_experiment`: directed node pooling with deep/raw normal support.
* `fedxmag_next.joint_rank`: symmetric flow/node rank combination with all controls.
* `fedxmag_next.normal_memory`: conservative verified-normal prototype filter.
* `fedxmag_next.quota_sample`: label-independent coverage supplement for sparse minutes.
* `fedxmag_next.feature_subset`: matched current/pair/outgoing history ablations.
* `fedxmag_next.conditional_experiment`: source-normal current density given history.
* `fedxmag_next.conditional_detector`: read-only conditional model restoration and scoring.
* `fedxmag_next.neural_transforms`: normal-only learned-mask contrastive control.
* `fedxmag_next.cache_node_keys`: grouping metadata for personalized normal memory.
* `fedxmag_next.trusted_adjacency`: explicitly identity-sensitive graph baseline.
* `fedxmag_next.role_sample`: label-free coverage of every active directed node role.
* `fedxmag_next.protocol_fields`: shape-only protocol metadata with unchanged rows.
* `fedxmag_next.protocol_context`: typed outgoing histories over all original rows.
* `fedxmag_next.context_association`: matched history-swap normal discrimination.
* `fedxmag_next.geometry_training`: raw-feature bypass and differentiable normal geometry.
* `fedxmag_next.cohort_experiment`: fixed-encoder service geometry and two-sided typicality.
* `fedxmag_next.frozen_residual`: raw bypass ablation with exact frozen DNN weights.
* `fedxmag_next.control_audit`: exact raw-control replay with matched feedback capacity.
* `fedxmag_next.neighbor_experiment`: frozen deep nearest-neighbor controls.
* `fedxmag_next.early_adaptation`: chronological verified-normal reference/refit controls.
* `fedxmag_next.merge_cic_observations`: full connection queries preserving legacy features.
* `fedxmag_next.expand_frozen_source`: fixed-weight source expansion with exact prototypes.
* `fedxmag_next.entity_support_experiment`: same-network endpoint-conditioned normal support.
* `fedxmag_next.hierarchical_experiment`: measured nested endpoint-rank aggregation control.
* `fedxmag_next.risk_experiment`: causal base-alarm evidence with bounded endpoint memory.
* `fedxmag_next.prototype_guard`: verified-normal keyed prototypes and restoration.
* `fedxmag_next.prototype_risk_experiment`: matched normal-filter/temporal-evidence factorial.
* `fedxmag_next.prototype_continuous`: continuous local normal energy with unchanged cache.
* `fedxmag_next.prototype_calibration`: final normal-feedback-reference ablation only.

Runs live under `runs/`; inputs, weights, histories, and scores are retained.
New registrations archive code hashes and source snapshots. Original datasets
and archived research outputs are preserved. Parser workers and training threads
default to six. The temporal GRU scaffold is excluded from the measured models;
it has not been trained for sequence prediction.
From round120, runners also freeze the execution package before launching jobs;
workspace edits cannot alter a running experiment. The direct-geometry candidate
is a new measured hypothesis, separate from the failed reconstruction proposal.

## Initial GNN hypothesis (failed development control)

The archived experiments established four constraints:

1. a graph mean can erase a sparse attack signal;
2. one Gaussian is not an adequate model of all normal capture/site modes;
3. lowering a normal Mahalanobis energy is not an anomaly-detection objective;
4. an alert-gated reference or threshold buffer cannot safely track upward
   normal drift.

This package consequently uses the following score-time pipeline:

```text
causal graph encoder
    -> node and edge embeddings (never graph-mean pooled)
    -> conditional mixture of shrinkage-Mahalanobis normal supports
    -> fixed top-k entity tails
    -> selection-normal scale fusion
    -> independently vetted, recent normal conformal calibration
```

The encoder is trained only with masked reconstruction.  It is **not** trained
to minimize an in-sample or cross-fitted Mahalanobis energy: the archived result
showed that objective is not a valid proxy for detecting attacks.  The detector
uses the representation after that pretext training and estimates its support
only from a separate normal-reference role.

Each entity is scored against a mixture of normal modes, not a single global
covariance.  Each component has an isotropically-shrunk covariance and its own
mean.  The mixture score is the lowest prior-adjusted component energy; it is a
Mahalanobis normal-support score but permits multiple normal operating modes.

## Safe deployment contract

* Reference, selection-scale, calibration, and evaluation windows must be
  disjoint at the capture/site level for a generalization test.
* New normal modes may be added only through an external, vetted-normal
  commissioning path.  A queried or merely non-alerting window is never
  admitted.
* A 5% conformal threshold requires at least 19 vetted calibration windows.
  Calibration is deliberately an explicit operation so a stale threshold cannot
  silently become a deployment guarantee.
* The causal state store is optional.  It can contain only past, vetted-normal
  entity states; scoring is read-only and therefore cannot let an attack poison
  future context.

## Quantum note

No quantum implementation or reproducible quantum advantage was present in the
workspace.  This code deliberately does not label a classical or simulated
kernel as a quantum advantage.  If a separately validated quantum feature map
is supplied, it can be inserted before `ConditionalMahalanobis.fit`; it must
obey exactly the same normal-reference, capture split, and score-time contracts
and be compared with the classical encoder at matched resources.

## Layout

* `src/fedxmag_next/`: package implementation.
* `tests/`: unit and synthetic end-to-end tests.
* `scripts/smoke.py`: deterministic synthetic path check (not scientific
  evidence).
* `docs/design.md`: architecture, falsifiable hypotheses, and evaluation gates.

## Quick start

Use the existing Python 3.12 environment without installing into the archived
package:

```powershell
$py = 'C:\Users\user\anaconda3\envs\quantum\python.exe'
$env:PYTHONPATH = 'New_Model\src'
& $py -m pytest New_Model\tests -q
& $py New_Model\scripts\smoke.py
```

To train from the existing serialized graph artifact format (use only a newly
registered, unconsumed protocol for a scientific run):

```powershell
& $py -m fedxmag_next.cli train `
  --train path\reference_train.pt --selection path\selection_normal.pt `
  --calibration path\calibration_normal.pt --artifact New_Model\artifacts\detector.pt `
  --workers 6
& $py -m fedxmag_next.cli score --artifact New_Model\artifacts\detector.pt `
  --input path\evaluation.pt --output New_Model\artifacts\scores.json
```

The loader reads the repository's tensor-only graph artifact format but does not
import the archived detector code.
