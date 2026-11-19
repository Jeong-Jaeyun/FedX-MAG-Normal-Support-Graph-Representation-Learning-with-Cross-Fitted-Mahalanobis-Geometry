"""Learn normal current/history compatibility with intact current observations.

Negatives replace ONLY strictly-past context with a DIFFERENT context from the
same protocol/service cohort. No real attack participates. Constant-context
cohorts supply no changed-context pairs and are reported explicitly.
"""

from __future__ import annotations

import torch
from torch import nn

from .record_data import FEATURES, PROTOCOLS, SERVICES


def cohort_ids(features):
    proto = [FEATURES.index(f"protocol_{name}") for name in (*PROTOCOLS, "other")]
    service = [FEATURES.index(f"service_{name}") for name in (*SERVICES, "other")]
    if not torch.all(features[:, proto].sum(1) == 1) or not torch.all(
        features[:, service].sum(1) == 1
    ):
        raise ValueError("Association cohorts require observed protocol/service one-hot values")
    return features[:, proto].argmax(1) * (len(SERVICES) + 1) + features[:, service].argmax(1)


class ContextPairSampler:
    def __init__(self, values, cohorts, context_columns):
        if values.ndim != 2 or len(values) != len(cohorts) or not context_columns:
            raise ValueError("Aligned rows, cohorts and nonempty history columns required")
        self.values, self.columns = values, list(context_columns)
        self.groups = []
        self.eligible = 0
        for cohort in torch.unique(cohorts):
            indices = torch.nonzero(cohorts == cohort).flatten()
            _, inverse, counts = torch.unique(
                values[indices][:, self.columns], dim=0, return_inverse=True, return_counts=True
            )
            if len(counts) < 2:
                continue
            ordered = indices[torch.argsort(inverse, stable=True)]
            sizes = counts[inverse]
            offsets = (counts.cumsum(0) - counts)[inverse]
            self.groups.append((indices, ordered, sizes, offsets))
            self.eligible += len(indices)

    def sample(self, generator):
        positive, donors = [], []
        for indices, ordered, sizes, offsets in self.groups:
            # Uniform row from the complement of this row's equal-context block.
            draw = (torch.rand(len(indices), generator=generator) * (len(indices) - sizes)).long()
            draw = draw + (draw >= offsets) * sizes
            partner = ordered[draw]
            if not torch.any(
                self.values[indices][:, self.columns] != self.values[partner][:, self.columns],
                dim=1,
            ).all():
                raise AssertionError("A context negative equals the real normal observation")
            positive.append(indices)
            donors.append(partner)
        if not positive:
            raise ValueError("No changed-context normal association pairs")
        return torch.cat(positive), torch.cat(donors)

    def negative(self, indices, donors):
        result = self.values[indices].clone()
        result[:, self.columns] = self.values[donors][:, self.columns]
        return result


def train_association(
    model,
    reference,
    selection,
    reference_cohorts,
    selection_cohorts,
    context_columns,
    *,
    seed,
    epochs,
):
    train_pairs = ContextPairSampler(reference, reference_cohorts, context_columns)
    validation_pairs = ContextPairSampler(selection, selection_cohorts, context_columns)
    if min(train_pairs.eligible, validation_pairs.eligible) < 32 or epochs < 1:
        raise ValueError("Insufficient changed-context normal pairs or training budget")
    generator = torch.Generator().manual_seed(seed)
    fixed = torch.Generator().manual_seed(7189)
    validation_ids, validation_donors = validation_pairs.sample(fixed)
    positive = selection[validation_ids]
    negative = validation_pairs.negative(validation_ids, validation_donors)
    validation = torch.cat((positive, negative))
    validation_labels = torch.cat((torch.zeros(len(positive)), torch.ones(len(negative))))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    history, best_state, best_value = [], None, float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        ids, donors = train_pairs.sample(generator)
        order = torch.randperm(len(ids), generator=generator)
        total, count = 0.0, 0
        for positions in order.split(256):
            positive = reference[ids[positions]]
            negative = train_pairs.negative(ids[positions], donors[positions])
            predictions = model(torch.cat((positive, negative)))
            labels = torch.cat((torch.zeros(len(positive)), torch.ones(len(negative))))
            loss = criterion(predictions, labels)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite normal association objective")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach()) * len(positions)
            count += len(positions)
        model.eval()
        with torch.no_grad():
            validation_loss = float(criterion(model(validation), validation_labels))
        entry = {
            "epoch": epoch,
            "train_bce": total / count,
            "selection_matched_context_bce": validation_loss,
            "train_changed_context_pairs": train_pairs.eligible,
            "selection_changed_context_pairs": validation_pairs.eligible,
        }
        history.append(entry)
        if validation_loss < best_value:
            best_value = validation_loss
            best_state = {
                name: value.detach().clone() for name, value in model.state_dict().items()
            }
        if epoch == 1 or epoch % 10 == 0:
            print(entry, flush=True)
    model.load_state_dict(best_state)
    model.eval()
    return history
