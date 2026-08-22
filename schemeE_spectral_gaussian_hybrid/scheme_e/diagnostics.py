from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch
import torch.nn.functional as functional

from .angle_delay import ChannelShape
from .metrics import official_score, pas_spectrum, pdp_spectrum


@dataclass(frozen=True)
class SampleMetricBatch:
    pas: np.ndarray
    pas_sum: np.ndarray
    pas_count: np.ndarray
    pdp: np.ndarray
    pdp_sum: np.ndarray
    pdp_count: np.ndarray
    error_energy: np.ndarray
    target_energy: np.ndarray
    prediction_energy: np.ndarray
    cross_real: np.ndarray
    cross_imag: np.ndarray
    target_log_power: np.ndarray
    prediction_log_power: np.ndarray
    sample_nmse: np.ndarray
    sample_score: np.ndarray

    def as_dict(self, prefix: str = "") -> dict[str, np.ndarray]:
        return {
            f"{prefix}{name}": np.asarray(value)
            for name, value in self.__dict__.items()
        }


def _row_cosine_components(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = prediction.shape[0]
    prediction = prediction.float().reshape(batch, -1, prediction.shape[-1])
    target = target.float().reshape(batch, -1, target.shape[-1])
    prediction_scale = prediction.abs().amax(dim=-1, keepdim=True)
    target_scale = target.abs().amax(dim=-1, keepdim=True)
    valid = target_scale[..., 0] > 1e-30
    prediction = torch.where(
        prediction_scale > 1e-30,
        prediction / prediction_scale.clamp_min(1e-30),
        torch.zeros_like(prediction),
    )
    target = target / target_scale.clamp_min(1e-30)
    cosine = functional.cosine_similarity(
        prediction, target, dim=-1, eps=1e-8
    ).clamp(0.0, 1.0)
    cosine = cosine.masked_fill(~valid, 0.0)
    return cosine.sum(dim=1), valid.sum(dim=1)


@torch.no_grad()
def sample_metric_batch(
    prediction: torch.Tensor,
    target: torch.Tensor,
    shape: ChannelShape,
    true_outage: torch.Tensor | np.ndarray | None = None,
) -> SampleMetricBatch:
    if prediction.shape != target.shape:
        raise ValueError(
            f"Prediction shape {tuple(prediction.shape)} != target shape {tuple(target.shape)}"
        )
    pas_sum, pas_count = _row_cosine_components(
        pas_spectrum(prediction, shape), pas_spectrum(target, shape)
    )
    pdp_sum, pdp_count = _row_cosine_components(
        pdp_spectrum(prediction), pdp_spectrum(target)
    )
    if true_outage is not None:
        outage = torch.as_tensor(true_outage, device=prediction.device).bool()
        pas_sum = pas_sum.masked_fill(outage, 0.0)
        pdp_sum = pdp_sum.masked_fill(outage, 0.0)
        pas_count = pas_count.masked_fill(outage, 0)
        pdp_count = pdp_count.masked_fill(outage, 0)

    dimensions = tuple(range(1, prediction.ndim))
    difference = prediction - target
    error_energy = difference.abs().square().sum(dim=dimensions).double()
    target_energy = target.abs().square().sum(dim=dimensions).double()
    prediction_energy = prediction.abs().square().sum(dim=dimensions).double()
    cross = (prediction.conj() * target).sum(dim=dimensions)
    target_power = target.abs().square().mean(dim=dimensions).double()
    prediction_power = prediction.abs().square().mean(dim=dimensions).double()

    pas = pas_sum / pas_count.clamp_min(1)
    pdp = pdp_sum / pdp_count.clamp_min(1)
    sample_nmse = error_energy / target_energy.clamp_min(1e-30)
    sample_nmse = torch.where(
        target_energy > 1e-30,
        sample_nmse,
        prediction_energy / 1e-30,
    )
    sample_score = (
        0.4 * pas.double()
        + 0.4 * pdp.double()
        + 0.2 / (1.0 + sample_nmse)
    )
    target_log_power = torch.where(
        target_power > 1e-30,
        torch.log10(target_power.clamp_min(1e-30)),
        torch.zeros_like(target_power),
    )
    prediction_log_power = torch.where(
        prediction_power > 1e-30,
        torch.log10(prediction_power.clamp_min(1e-30)),
        torch.zeros_like(prediction_power),
    )

    def array(value: torch.Tensor, dtype: np.dtype | None = None) -> np.ndarray:
        result = value.detach().cpu().numpy()
        return result.astype(dtype, copy=False) if dtype is not None else result

    return SampleMetricBatch(
        pas=array(pas, np.float64),
        pas_sum=array(pas_sum, np.float64),
        pas_count=array(pas_count, np.int64),
        pdp=array(pdp, np.float64),
        pdp_sum=array(pdp_sum, np.float64),
        pdp_count=array(pdp_count, np.int64),
        error_energy=array(error_energy, np.float64),
        target_energy=array(target_energy, np.float64),
        prediction_energy=array(prediction_energy, np.float64),
        cross_real=array(cross.real.double(), np.float64),
        cross_imag=array(cross.imag.double(), np.float64),
        target_log_power=array(target_log_power, np.float64),
        prediction_log_power=array(prediction_log_power, np.float64),
        sample_nmse=array(sample_nmse, np.float64),
        sample_score=array(sample_score, np.float64),
    )


def concatenate_metric_batches(
    batches: list[SampleMetricBatch],
) -> dict[str, np.ndarray]:
    if not batches:
        raise ValueError("At least one metric batch is required")
    return {
        name: np.concatenate([np.asarray(getattr(batch, name)) for batch in batches])
        for name in SampleMetricBatch.__dataclass_fields__
    }


def aggregate_sample_metrics(values: Mapping[str, np.ndarray]) -> dict[str, float | int]:
    pas_count = int(np.asarray(values["pas_count"], dtype=np.int64).sum())
    pdp_count = int(np.asarray(values["pdp_count"], dtype=np.int64).sum())
    pas = float(np.asarray(values["pas_sum"], dtype=np.float64).sum() / max(pas_count, 1))
    pdp = float(np.asarray(values["pdp_sum"], dtype=np.float64).sum() / max(pdp_count, 1))
    numerator = float(np.asarray(values["error_energy"], dtype=np.float64).sum())
    denominator = float(np.asarray(values["target_energy"], dtype=np.float64).sum())
    channel_nmse = numerator / max(denominator, 1e-30)
    return {
        "pas": pas,
        "pdp": pdp,
        "nmse": channel_nmse,
        "score": official_score(pas, pdp, channel_nmse),
        "samples": int(len(np.asarray(values["target_energy"]))),
        "pas_rows": pas_count,
        "pdp_rows": pdp_count,
        "error_energy": numerator,
        "target_energy": denominator,
    }


@torch.no_grad()
def scale_oracle_predictions(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, torch.Tensor]:
    dimensions = tuple(range(1, prediction.ndim))
    prediction_energy = prediction.abs().square().sum(dim=dimensions).double()
    target_energy = target.abs().square().sum(dim=dimensions).double()
    cross = (prediction.conj() * target).sum(dim=dimensions)
    safe_prediction_energy = prediction_energy.clamp_min(1e-30)
    real_scale = (cross.real.double() / safe_prediction_energy).clamp_min(0.0)
    complex_scale = cross / safe_prediction_energy.to(cross.dtype)
    power_scale = torch.sqrt(
        target_energy.clamp_min(0.0) / safe_prediction_energy
    )
    view = (-1,) + (1,) * (prediction.ndim - 1)
    return {
        "oracle_real_scale": prediction * real_scale.reshape(view).to(prediction.dtype),
        "oracle_complex_scale": prediction
        * complex_scale.reshape(view).to(prediction.dtype),
        "oracle_power_scale": prediction
        * power_scale.reshape(view).to(prediction.dtype),
    }


def hard_zero_metric_arrays(
    values: Mapping[str, np.ndarray], mask: np.ndarray
) -> dict[str, np.ndarray]:
    selected = np.asarray(mask, dtype=bool)
    output = {name: np.array(value, copy=True) for name, value in values.items()}
    if len(selected) != len(output["target_energy"]):
        raise ValueError("Hard-zero mask and metric sample counts differ")
    output["pas_sum"][selected] = 0.0
    output["pdp_sum"][selected] = 0.0
    output["error_energy"][selected] = output["target_energy"][selected]
    output["prediction_energy"][selected] = 0.0
    output["cross_real"][selected] = 0.0
    output["cross_imag"][selected] = 0.0
    output["prediction_log_power"][selected] = 0.0
    output["pas"][selected] = 0.0
    output["pdp"][selected] = 0.0
    output["sample_nmse"][selected] = np.where(
        output["target_energy"][selected] > 1e-30, 1.0, 0.0
    )
    output["sample_score"][selected] = np.where(
        output["target_energy"][selected] > 1e-30, 0.1, 0.2
    )
    return output


def outage_threshold_oracle(
    values: Mapping[str, np.ndarray],
    outage_probability: np.ndarray,
    true_outage: np.ndarray,
    cell_ids: np.ndarray,
    *,
    threshold_steps: int = 50,
) -> dict[str, object]:
    probability = np.asarray(outage_probability, dtype=np.float64)
    target = np.asarray(true_outage, dtype=bool)
    cells = np.asarray(cell_ids, dtype=np.int64)
    if not (len(probability) == len(target) == len(cells)):
        raise ValueError("Outage diagnostic arrays have different lengths")
    thresholds = np.concatenate(
        [np.linspace(0.0, 1.0, int(threshold_steps) + 1), np.asarray([1.000001])]
    )

    def evaluate(mask: np.ndarray) -> dict[str, float | int]:
        return aggregate_sample_metrics(hard_zero_metric_arrays(values, mask))

    global_candidates = [
        (float(threshold), evaluate(probability >= threshold))
        for threshold in thresholds
    ]
    global_threshold, global_metrics = max(
        global_candidates, key=lambda item: float(item[1]["score"])
    )

    unique_cells = np.unique(cells)
    best_cell_thresholds = {str(int(cell)): 1.000001 for cell in unique_cells}
    best_cell_metrics = evaluate(np.zeros(len(target), dtype=bool))
    if len(unique_cells) == 2:
        first, second = (int(value) for value in unique_cells)
        for first_threshold in thresholds:
            first_mask = (cells == first) & (probability >= first_threshold)
            for second_threshold in thresholds:
                mask = first_mask | (
                    (cells == second) & (probability >= second_threshold)
                )
                metrics = evaluate(mask)
                if float(metrics["score"]) > float(best_cell_metrics["score"]):
                    best_cell_metrics = metrics
                    best_cell_thresholds = {
                        str(first): float(first_threshold),
                        str(second): float(second_threshold),
                    }

    positives = int(target.sum())
    negatives = int((~target).sum())
    auc = float("nan")
    if positives and negatives:
        order = np.argsort(probability, kind="mergesort")
        ranks = np.empty(len(order), dtype=np.float64)
        ranks[order] = np.arange(1, len(order) + 1, dtype=np.float64)
        _, inverse, counts = np.unique(
            probability, return_inverse=True, return_counts=True
        )
        rank_sums = np.bincount(inverse, weights=ranks)
        ranks = (rank_sums / counts)[inverse]
        auc = float(
            (ranks[target].sum() - positives * (positives + 1) / 2)
            / (positives * negatives)
        )

    return {
        "diagnostic_only": True,
        "true_outage_count": positives,
        "nonoutage_count": negatives,
        "probability_auc": auc,
        "perfect_label_metrics": evaluate(target),
        "best_global_hard_threshold": global_threshold,
        "best_global_hard_metrics": global_metrics,
        "best_per_cell_hard_thresholds": best_cell_thresholds,
        "best_per_cell_hard_metrics": best_cell_metrics,
    }


def target_informed_expert_oracle(
    experts: Mapping[str, Mapping[str, np.ndarray]],
    *,
    maximum_iterations: int = 20,
) -> dict[str, object]:
    if len(experts) < 2:
        raise ValueError("At least two experts are required for an oracle")
    names = list(experts)
    count = len(np.asarray(experts[names[0]]["target_energy"]))
    for name in names[1:]:
        if len(np.asarray(experts[name]["target_energy"])) != count:
            raise ValueError("Expert sample counts differ")

    pas_total = max(
        int(np.asarray(experts[names[0]]["pas_count"], dtype=np.int64).sum()), 1
    )
    pdp_total = max(
        int(np.asarray(experts[names[0]]["pdp_count"], dtype=np.int64).sum()), 1
    )
    target_total = max(
        float(np.asarray(experts[names[0]]["target_energy"], dtype=np.float64).sum()),
        1e-30,
    )
    score_matrix = np.stack(
        [np.asarray(experts[name]["sample_score"], dtype=np.float64) for name in names],
        axis=1,
    )
    selection = np.argmax(score_matrix, axis=1)
    sample_score_selection = selection.copy()

    def selected_arrays(selected: np.ndarray) -> dict[str, np.ndarray]:
        output: dict[str, np.ndarray] = {}
        for field in SampleMetricBatch.__dataclass_fields__:
            matrix = np.stack(
                [np.asarray(experts[name][field]) for name in names], axis=1
            )
            output[field] = matrix[np.arange(count), selected]
        return output

    for iteration in range(maximum_iterations):
        current = selected_arrays(selection)
        nmse = float(np.asarray(current["error_energy"]).sum() / target_total)
        error_weight = 0.2 / (target_total * (1.0 + nmse) ** 2)
        utility = []
        for name in names:
            values = experts[name]
            utility.append(
                0.4 * np.asarray(values["pas_sum"], dtype=np.float64) / pas_total
                + 0.4 * np.asarray(values["pdp_sum"], dtype=np.float64) / pdp_total
                - error_weight
                * np.asarray(values["error_energy"], dtype=np.float64)
            )
        updated = np.argmax(np.stack(utility, axis=1), axis=1)
        if np.array_equal(updated, selection):
            break
        selection = updated

    candidates = {
        "sample_score": (sample_score_selection, selected_arrays(sample_score_selection)),
        "global_linearized": (selection, selected_arrays(selection)),
    }
    selected_name, (best_selection, best_arrays) = max(
        candidates.items(), key=lambda item: float(aggregate_sample_metrics(item[1][1])["score"])
    )
    metrics = aggregate_sample_metrics(best_arrays)
    return {
        "diagnostic_only": True,
        "method": selected_name,
        "experts": names,
        "metrics": metrics,
        "selection": best_selection.astype(np.int16),
        "selection_counts": {
            name: int(np.sum(best_selection == index))
            for index, name in enumerate(names)
        },
        "iterations": int(iteration + 1),
    }
