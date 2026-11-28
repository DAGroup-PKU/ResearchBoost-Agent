"""Validate complete GPU records and evaluate exact, timing and N1 gates."""

import math
import re
import statistics

from runtime import require


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def step_kind(completed, contract):
    gate = contract["gate0"]
    schedule = gate["update_boundary"]["schedule"]
    period = gate["exact_items"]["config"][schedule["generator_period_config_key"]]
    prior = completed - 1
    both = (prior > 0 or not schedule["generator_starts_after_step_zero"]) and prior % period == schedule["generator_step_modulo"]
    kind = "generator_dmd_and_critic" if both else "critic_only"
    return kind, schedule[kind]


def validate_ranks(ranks, spec, contract):
    groups = contract["correctness"]["trajectory"]["endpoints_by_step_type"]
    require(len(ranks) == contract["world_size"], "Missing rank reports")
    for rank, report in enumerate(ranks):
        require(report["rank"] == rank and report["seed"] == spec["seed"], "Rank or seed mismatch")
        require(len(report["steps"]) == spec["steps"], "Incomplete rank training updates")
        require(digest(report["config"]), "Missing resolved runtime config")
        require(set(report["initial"]) == {"optimizer", "starting_weights", "tensor_metadata", "grad_clip"}
                and all(digest(report["initial"][key]) for key in ("optimizer", "starting_weights", "tensor_metadata"))
                and len(report["initial"]["grad_clip"]) == 2
                and all(number(value) for value in report["initial"]["grad_clip"]), "Missing starting state evidence")
        require(not spec["reference"] or type(report["flash_calls"]) is int and report["flash_calls"] > 0,
                "Reference did not execute FlashAttention-2")
        require(set(report["timestamps"]) == {str(step) for window in spec["windows"] for step in window}
                and all(number(stamp) for stamp in report["timestamps"].values()), "Incomplete window boundaries")
        for completed, row in enumerate(report["steps"], start=1):
            kind, updates = step_kind(completed, contract)
            require(row["step"] == completed and [event["kind"] for event in row["exact"]] == updates,
                    "Rank did not complete the required optimizer schedule")
            for event in row["exact"]:
                require(len(event["sample_ids"]) == contract["gate0"]["exact_items"]["config"]["batch_size"]
                        and all(type(index) is int and index >= 0 for index in event["sample_ids"])
                        and all(digest(event[key]) for key in (
                            "prompts", "sampling", "rollout_trace", "score_trace", "gradients", "optimizer_state")),
                        "Incomplete exact update evidence")
            require(set(row["endpoints"]) == (set(groups[kind]) if spec["endpoints"] else set())
                    and all(number(value) and (not key.endswith(("_norm", "_delta")) or value >= 0)
                            for key, value in row["endpoints"].items()), "Missing or nonfinite trajectory endpoints")


def gate0(reference, candidate):
    for field in ("seed", "assets", "contract", "steps", "windows", "recipe", "endpoints"):
        require(reference[field] == candidate[field], f"Gate 0: mismatched {field}", "INCORRECT")
    for rank, (expected, observed) in enumerate(zip(reference["ranks"], candidate["ranks"], strict=True)):
        for field in ("config", "initial"):
            require(expected[field] == observed[field], f"Gate 0: rank {rank} {field} differs", "INCORRECT")
        for left, right in zip(expected["steps"], observed["steps"], strict=True):
            require(left["exact"] == right["exact"],
                    f"Gate 0: rank {rank}, completed step {left['step']}: sample, rollout, dtype or update differs",
                    "INCORRECT")


def endpoints(run):
    values = {}
    for rows in zip(*(rank["steps"] for rank in run["ranks"]), strict=True):
        for key in rows[0]["endpoints"]:
            local = [row["endpoints"][key] for row in rows]
            if key.endswith("_parameter_delta"):
                value = math.sqrt(math.fsum(local))
                require(value > 0, "Optimizer did not change any parameter on any rank")
            else:
                value = statistics.fmean(local)
            values[(rows[0]["step"], key)] = value
    return values


def n1(calibration, comparisons, contract):
    policy = contract["correctness"]["trajectory"]
    require([run["seed"] for run in calibration] == policy["reference_seeds"]
            and len(calibration) == policy["reference_seed_count"]
            and all(run["reference"] for run in calibration) and comparisons, "Missing N1 reference calibration")
    all_values = [endpoints(run) for run in calibration]
    expected_steps = {(step, key) for step in range(1, policy["completed_steps"] + 1)
                      for key in policy["endpoints_by_step_type"][step_kind(step, contract)[0]]}
    require({step_kind(step, contract)[0] for step, _ in expected_steps} == set(policy["endpoints_by_step_type"])
            and all(values.keys() == expected_steps for values in all_values), "Incomplete N1 endpoint groups")
    sigma = {endpoint: statistics.stdev(values[endpoint] for values in all_values) for endpoint in expected_steps}
    checked = 0
    for reference, candidate in comparisons:
        gate0(reference, candidate)
        left, right = endpoints(reference), endpoints(candidate)
        require(left.keys() == right.keys() == expected_steps, "Incomplete paired trajectory")
        for step, key in sorted(expected_steps):
            endpoint = step, key
            bias = abs(left[endpoint] - right[endpoint])
            bound = policy["sigma_multiplier"] * sigma[endpoint]
            require(math.isfinite(bias) and math.isfinite(bound) and bias <= bound,
                    f"Gate 3: seed {reference['seed']}, completed step {step}, {key} outside N1 band", "INCORRECT")
            checked += 1
    return {"tier": "N1", "paired_seeds": len(comparisons), "endpoints_checked": checked}


def window_times(run):
    values = []
    for low, high in run["windows"]:
        elapsed = [rank["timestamps"][str(high)] - rank["timestamps"][str(low)] for rank in run["ranks"]]
        require(all(number(value) and value > 0 for value in elapsed), "Nonpositive or invalid timing window")
        values.append(max(elapsed) / (high - low))
    require(values, "No timing windows")
    return values


def timing(reference_a, candidate, reference_b, contract):
    require(reference_a["reference"] and reference_b["reference"], "Missing live reference timing brackets")
    gate0(reference_a, reference_b)
    gate0(reference_a, candidate)
    first, submitted, last = map(window_times, (reference_a, candidate, reference_b))
    spread = max(abs(left - right) / min(left, right) for left, right in zip(first, last, strict=True))
    policy = contract["timing"]
    require(spread <= policy["spread_band"], "Live reference timings exceed the machine consistency band",
            policy["over_band_status"])
    speedup = statistics.fmean(min(left, right) / cand for left, cand, right in zip(first, submitted, last, strict=True))
    benefit = contract["benefit"]
    threshold = max(benefit["min_speedup"], 1 + benefit["reference_spread_multiplier"] * spread)
    require(number(speedup) and number(threshold), "Invalid speedup statistic")
    return {"windows": candidate["windows"], "s_per_update": submitted, "reference_a": first, "reference_b": last,
            "reference_spread": spread, "speedup": speedup, "threshold": threshold}
