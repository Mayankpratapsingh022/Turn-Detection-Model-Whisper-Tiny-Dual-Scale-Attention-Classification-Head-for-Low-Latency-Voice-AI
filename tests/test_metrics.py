import math

from turn_detector.evaluation.metrics import (
    PausePrediction,
    binary_classification_metrics,
    bootstrap_policy,
    evaluate_policy,
    fit_temperature,
    mcnemar_test,
    operating_point,
    paired_group_bootstrap_delta,
    pareto_frontier,
    policy_sweep,
)


def test_binary_metrics() -> None:
    metrics = binary_classification_metrics([0, 0, 1, 1], [0.1, 0.8, 0.7, 0.9])
    assert metrics["accuracy"] == 0.75
    assert metrics["false_cutoff_rate"] == 0.5
    assert metrics["recall"] == 1.0
    assert 0 <= metrics["ece"] <= 1


def causal_predictions() -> list[PausePrediction]:
    return [
        PausePrediction("a-hold", "a", "hold", 0.9, 700),
        PausePrediction("a-eot", "a", "eot", 0.9, 10_000),
        PausePrediction("b-hold", "b", "hold", 0.9, 100),
        PausePrediction("b-eot", "b", "eot", 0.9, 10_000),
    ]


def test_policy_evaluation_is_turn_level() -> None:
    result = evaluate_policy(
        causal_predictions(), threshold=0.5, action_delay_ms=200, timeout_ms=1_000
    )
    assert result.false_cutoff_rate == 0.5
    assert result.mean_endpoint_latency_ms == 200
    assert result.turns == 2


def test_policy_sweep_and_operating_point() -> None:
    results = policy_sweep(
        causal_predictions(),
        thresholds=[0.5, 0.95],
        action_delays_ms=[200],
        timeouts_ms=[1_000],
    )
    safe = operating_point(results, max_false_cutoff_rate=0.0)
    assert safe is not None
    assert safe.mean_endpoint_latency_ms == 1_000
    assert pareto_frontier(results)


def test_calibration_and_mcnemar() -> None:
    temperature = fit_temperature([0, 0, 1, 1], [-4.0, -2.0, 2.0, 4.0])
    assert math.isfinite(temperature) and temperature > 0
    result = mcnemar_test([0, 0, 1, 1], [0.1, 0.9, 0.9, 0.9], [0.1, 0.1, 0.1, 0.9])
    assert 0 <= result["p_value"] <= 1


def test_group_bootstraps_are_paired_and_turn_level() -> None:
    policy_interval = bootstrap_policy(
        causal_predictions(),
        threshold=0.5,
        action_delay_ms=200,
        timeout_ms=1_000,
        samples=50,
        seed=7,
    )
    assert policy_interval["false_cutoff_rate"][0] == 0.5
    delta = paired_group_bootstrap_delta(
        [0, 0, 1, 1],
        [0.1, 0.2, 0.8, 0.9],
        [0.9, 0.8, 0.2, 0.1],
        ["a", "b", "c", "d"],
        samples=50,
    )
    assert delta["delta"] > 0
    assert 0 <= delta["probability_candidate_better"] <= 1
