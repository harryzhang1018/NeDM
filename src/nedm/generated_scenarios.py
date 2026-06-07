from __future__ import annotations

import random
from typing import Any


def validate_generator_config(generator_cfg: dict[str, Any]) -> None:
    families = generator_cfg.get("families", [])
    if not families:
        raise ValueError("scenario_generator.families must not be empty")
    for family in families:
        if family.get("count", 0) <= 0:
            raise ValueError(f"Generated family {family.get('name', '<unknown>')} must have a positive count")


def expand_scenarios(config: dict[str, Any]) -> list[dict[str, Any]]:
    scenarios = [with_family(scenario) for scenario in config.get("scenarios", [])]
    generator_cfg = config.get("scenario_generator")
    if not generator_cfg:
        return scenarios
    return scenarios + generate_scenarios(generator_cfg)


def with_family(scenario: dict[str, Any]) -> dict[str, Any]:
    scenario_copy = dict(scenario)
    scenario_copy.setdefault("family", scenario_copy["name"])
    return scenario_copy


def generate_scenarios(generator_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    seed = int(generator_cfg.get("seed", 0))
    rng = random.Random(seed)
    scenarios: list[dict[str, Any]] = []

    builders = {
        "launch_brake": build_launch_brake_scenario,
        "step_steer": build_step_steer_scenario,
        "doublet_steer": build_doublet_steer_scenario,
        "sine_steer": build_sine_steer_scenario,
        "chirp_steer": build_chirp_steer_scenario,
        "steer_brake": build_steer_brake_scenario,
        "multi_steer": build_multi_steer_scenario,
    }

    for family_cfg in generator_cfg["families"]:
        family_name = family_cfg["name"]
        builder_name = family_cfg.get("template", family_name)
        if builder_name not in builders:
            raise ValueError(
                f"Unsupported generated maneuver family: {family_name} "
                f"(template={builder_name})"
            )
        builder = builders[builder_name]
        for family_index in range(int(family_cfg["count"])):
            scenario = builder(rng, generator_cfg, family_cfg, family_index)
            if builder_name != family_name or "scenario_prefix" in family_cfg or "family_label" in family_cfg:
                family_label = str(family_cfg.get("family_label", family_name))
                scenario_prefix = str(family_cfg.get("scenario_prefix", family_label))
                scenario["family"] = family_label
                scenario["name"] = f"{scenario_prefix}_{family_index:05d}"
            scenarios.append(scenario)

    shuffle_seed = int(generator_cfg.get("shuffle_seed", seed + 1))
    shuffle_rng = random.Random(shuffle_seed)
    shuffle_rng.shuffle(scenarios)
    return scenarios


def sample_range(rng: random.Random, family_cfg: dict[str, Any], key: str, default: tuple[float, float]) -> float:
    low, high = family_cfg.get(key, default)
    return rng.uniform(float(low), float(high))


def sample_int_range(rng: random.Random, family_cfg: dict[str, Any], key: str, default: tuple[int, int]) -> int:
    low, high = family_cfg.get(key, default)
    return rng.randint(int(low), int(high))


def constant_profile(value: float) -> dict[str, Any]:
    return {"kind": "constant", "value": float(value)}


def piecewise_profile(points: list[tuple[float, float]]) -> dict[str, Any]:
    return {"kind": "piecewise_linear", "points": [[float(t), float(v)] for t, v in points]}


def sine_profile(amplitude: float, frequency_hz: float, start_s: float, end_s: float, offset: float = 0.0) -> dict[str, Any]:
    return {
        "kind": "sine",
        "amplitude": float(amplitude),
        "offset": float(offset),
        "frequency_hz": float(frequency_hz),
        "phase_rad": 0.0,
        "start_s": float(start_s),
        "end_s": float(end_s),
    }


def chirp_profile(
    amplitude: float,
    start_frequency_hz: float,
    end_frequency_hz: float,
    start_s: float,
    end_s: float,
    offset: float = 0.0,
) -> dict[str, Any]:
    return {
        "kind": "chirp",
        "amplitude": float(amplitude),
        "offset": float(offset),
        "start_frequency_hz": float(start_frequency_hz),
        "end_frequency_hz": float(end_frequency_hz),
        "phase_rad": 0.0,
        "start_s": float(start_s),
        "end_s": float(end_s),
    }


def bounded_next(current: float, delta: float, upper: float) -> float:
    return min(current + delta, upper)


def random_sign(rng: random.Random) -> float:
    return -1.0 if rng.random() < 0.5 else 1.0


def default_warmup(generator_cfg: dict[str, Any], family_cfg: dict[str, Any]) -> float:
    return float(family_cfg.get("warmup_s", generator_cfg.get("warmup_s", 2.5)))


def build_hold_throttle_profile(
    rng: random.Random,
    family_cfg: dict[str, Any],
    warmup_s: float,
    duration_s: float,
    peak_default: tuple[float, float],
) -> dict[str, Any]:
    peak = sample_range(rng, family_cfg, "throttle_peak_range", peak_default)
    start_s = warmup_s + sample_range(rng, family_cfg, "throttle_delay_s_range", (0.2, 0.9))
    rise_s = sample_range(rng, family_cfg, "throttle_rise_s_range", (0.8, 2.0))
    end_s = max(duration_s, start_s + rise_s)
    return piecewise_profile(
        [
            (0.0, 0.0),
            (start_s, 0.0),
            (bounded_next(start_s, rise_s, duration_s), peak),
            (duration_s, peak),
        ]
    )


def build_brake_pulse_profile(
    brake_start_s: float,
    duration_s: float,
    brake_peak: float,
    brake_rise_s: float,
    brake_hold_s: float,
    brake_release_s: float,
) -> dict[str, Any]:
    brake_peak_s = bounded_next(brake_start_s, brake_rise_s, duration_s)
    brake_hold_end_s = bounded_next(brake_peak_s, brake_hold_s, duration_s)
    brake_release_end_s = bounded_next(brake_hold_end_s, brake_release_s, duration_s)
    return piecewise_profile(
        [
            (0.0, 0.0),
            (brake_start_s, 0.0),
            (brake_peak_s, brake_peak),
            (brake_hold_end_s, brake_peak),
            (brake_release_end_s, 0.0),
            (duration_s, 0.0),
        ]
    )


def append_point(points: list[tuple[float, float]], time_s: float, value: float) -> None:
    time_s = float(time_s)
    value = float(value)
    if points and time_s < points[-1][0]:
        return
    if points and abs(time_s - points[-1][0]) < 1e-9:
        points[-1] = (time_s, value)
        return
    points.append((time_s, value))


def build_random_throttle_profile(
    rng: random.Random,
    family_cfg: dict[str, Any],
    warmup_s: float,
    duration_s: float,
) -> dict[str, Any]:
    points: list[tuple[float, float]] = [(0.0, 0.0), (warmup_s, 0.0)]
    current_s = warmup_s + sample_range(rng, family_cfg, "throttle_delay_s_range", (0.2, 1.0))
    append_point(points, current_s, 0.0)

    while current_s < duration_s - 0.5:
        target = sample_range(rng, family_cfg, "throttle_peak_range", (0.15, 0.5))
        ramp_s = sample_range(rng, family_cfg, "throttle_rise_s_range", (0.5, 2.0))
        hold_s = sample_range(rng, family_cfg, "throttle_hold_s_range", (2.0, 7.0))
        current_s = min(current_s + ramp_s, duration_s)
        append_point(points, current_s, target)
        current_s = min(current_s + hold_s, duration_s)
        append_point(points, current_s, target)
        if rng.random() < float(family_cfg.get("coast_probability", 0.25)):
            coast_s = sample_range(rng, family_cfg, "coast_s_range", (0.8, 3.0))
            current_s = min(current_s + 0.3, duration_s)
            append_point(points, current_s, 0.0)
            current_s = min(current_s + coast_s, duration_s)
            append_point(points, current_s, 0.0)

    append_point(points, duration_s, points[-1][1])
    return piecewise_profile(points)


def build_random_braking_profile(
    rng: random.Random,
    family_cfg: dict[str, Any],
    warmup_s: float,
    duration_s: float,
) -> dict[str, Any]:
    pulse_count = sample_int_range(rng, family_cfg, "brake_pulse_count_range", (0, 3))
    if pulse_count <= 0:
        return constant_profile(0.0)

    points: list[tuple[float, float]] = [(0.0, 0.0), (warmup_s, 0.0)]
    candidate_times = sorted(
        rng.uniform(warmup_s + 2.0, max(warmup_s + 2.0, duration_s - 1.5))
        for _ in range(pulse_count)
    )
    current_s = warmup_s
    for start_s in candidate_times:
        if start_s <= current_s + 0.5:
            continue
        peak = sample_range(rng, family_cfg, "brake_peak_range", (0.1, 0.5))
        rise_s = sample_range(rng, family_cfg, "brake_rise_s_range", (0.15, 0.8))
        hold_s = sample_range(rng, family_cfg, "brake_hold_s_range", (0.3, 1.5))
        release_s = sample_range(rng, family_cfg, "brake_release_s_range", (0.2, 1.0))
        append_point(points, start_s, 0.0)
        append_point(points, min(start_s + rise_s, duration_s), peak)
        append_point(points, min(start_s + rise_s + hold_s, duration_s), peak)
        current_s = min(start_s + rise_s + hold_s + release_s, duration_s)
        append_point(points, current_s, 0.0)
    append_point(points, duration_s, 0.0)
    return piecewise_profile(points)


def build_multi_steer_profile(
    rng: random.Random,
    family_cfg: dict[str, Any],
    warmup_s: float,
    duration_s: float,
) -> dict[str, Any]:
    event_count = sample_int_range(rng, family_cfg, "event_count_range", (5, 12))
    points: list[tuple[float, float]] = [(0.0, 0.0), (warmup_s, 0.0)]
    current_s = warmup_s + sample_range(rng, family_cfg, "steer_start_offset_s_range", (0.5, 2.0))
    append_point(points, current_s, 0.0)

    for _ in range(event_count):
        if current_s >= duration_s - 1.0:
            break
        gap_s = sample_range(rng, family_cfg, "event_gap_s_range", (0.4, 3.0))
        current_s = min(current_s + gap_s, duration_s)
        append_point(points, current_s, 0.0)

        amplitude = random_sign(rng) * sample_range(rng, family_cfg, "steering_amplitude_range", (0.08, 0.8))
        rise_s = sample_range(rng, family_cfg, "steer_rise_s_range", (0.15, 0.8))
        hold_s = sample_range(rng, family_cfg, "steer_hold_s_range", (0.4, 3.0))
        return_s = sample_range(rng, family_cfg, "steer_return_s_range", (0.15, 0.8))

        current_s = min(current_s + rise_s, duration_s)
        append_point(points, current_s, amplitude)
        current_s = min(current_s + hold_s, duration_s)
        append_point(points, current_s, amplitude)

        if rng.random() < float(family_cfg.get("reverse_pulse_probability", 0.35)):
            ratio = sample_range(rng, family_cfg, "reverse_pulse_ratio_range", (0.6, 1.2))
            cross_s = sample_range(rng, family_cfg, "reverse_cross_s_range", (0.2, 0.8))
            reverse_hold_s = sample_range(rng, family_cfg, "reverse_hold_s_range", (0.3, 1.8))
            current_s = min(current_s + cross_s, duration_s)
            append_point(points, current_s, -amplitude * ratio)
            current_s = min(current_s + reverse_hold_s, duration_s)
            append_point(points, current_s, -amplitude * ratio)

        current_s = min(current_s + return_s, duration_s)
        append_point(points, current_s, 0.0)

    append_point(points, duration_s, 0.0)
    return piecewise_profile(points)


def build_multi_steer_scenario(
    rng: random.Random,
    generator_cfg: dict[str, Any],
    family_cfg: dict[str, Any],
    family_index: int,
) -> dict[str, Any]:
    warmup_s = default_warmup(generator_cfg, family_cfg)
    duration_s = sample_range(rng, family_cfg, "duration_s_range", (35.0, 70.0))

    return {
        "name": f"multi_steer_{family_index:05d}",
        "family": "multi_steer",
        "duration_s": duration_s,
        "warmup_s": warmup_s,
        "driver": {
            "steering": build_multi_steer_profile(rng, family_cfg, warmup_s, duration_s),
            "throttle": build_random_throttle_profile(rng, family_cfg, warmup_s, duration_s),
            "braking": build_random_braking_profile(rng, family_cfg, warmup_s, duration_s),
        },
    }


def build_launch_brake_scenario(
    rng: random.Random,
    generator_cfg: dict[str, Any],
    family_cfg: dict[str, Any],
    family_index: int,
) -> dict[str, Any]:
    warmup_s = default_warmup(generator_cfg, family_cfg)
    duration_s = sample_range(rng, family_cfg, "duration_s_range", (10.0, 14.0))

    throttle_peak = sample_range(rng, family_cfg, "throttle_peak_range", (0.18, 0.65))
    throttle_start_s = warmup_s + sample_range(rng, family_cfg, "throttle_delay_s_range", (0.2, 0.9))
    throttle_rise_s = sample_range(rng, family_cfg, "throttle_rise_s_range", (0.8, 2.0))
    throttle_hold_s = sample_range(rng, family_cfg, "throttle_hold_s_range", (1.5, 4.0))
    throttle_release_s = sample_range(rng, family_cfg, "throttle_release_s_range", (0.3, 1.2))

    throttle_peak_s = bounded_next(throttle_start_s, throttle_rise_s, duration_s)
    throttle_hold_end_s = bounded_next(throttle_peak_s, throttle_hold_s, duration_s)
    throttle_release_end_s = bounded_next(throttle_hold_end_s, throttle_release_s, duration_s)

    brake_peak = sample_range(rng, family_cfg, "brake_peak_range", (0.25, 0.7))
    brake_start_s = bounded_next(
        throttle_release_end_s,
        sample_range(rng, family_cfg, "brake_delay_s_range", (0.2, 0.8)),
        duration_s,
    )
    brake_profile = build_brake_pulse_profile(
        brake_start_s=brake_start_s,
        duration_s=duration_s,
        brake_peak=brake_peak,
        brake_rise_s=sample_range(rng, family_cfg, "brake_rise_s_range", (0.2, 0.8)),
        brake_hold_s=sample_range(rng, family_cfg, "brake_hold_s_range", (0.5, 1.8)),
        brake_release_s=sample_range(rng, family_cfg, "brake_release_s_range", (0.3, 1.0)),
    )

    return {
        "name": f"launch_brake_{family_index:05d}",
        "family": "launch_brake",
        "duration_s": duration_s,
        "warmup_s": warmup_s,
        "driver": {
            "steering": constant_profile(0.0),
            "throttle": piecewise_profile(
                [
                    (0.0, 0.0),
                    (throttle_start_s, 0.0),
                    (throttle_peak_s, throttle_peak),
                    (throttle_hold_end_s, throttle_peak),
                    (throttle_release_end_s, 0.0),
                    (duration_s, 0.0),
                ]
            ),
            "braking": brake_profile,
        },
    }


def build_step_steer_scenario(
    rng: random.Random,
    generator_cfg: dict[str, Any],
    family_cfg: dict[str, Any],
    family_index: int,
) -> dict[str, Any]:
    warmup_s = default_warmup(generator_cfg, family_cfg)
    duration_s = sample_range(rng, family_cfg, "duration_s_range", (10.0, 14.0))
    sign = random_sign(rng)
    amplitude = sign * sample_range(rng, family_cfg, "steering_amplitude_range", (0.05, 0.32))
    start_s = warmup_s + sample_range(rng, family_cfg, "steer_start_offset_s_range", (0.7, 2.5))
    rise_s = sample_range(rng, family_cfg, "steer_rise_s_range", (0.2, 0.7))
    hold_s = sample_range(rng, family_cfg, "steer_hold_s_range", (0.4, 2.0))
    return_s = sample_range(rng, family_cfg, "steer_return_s_range", (0.2, 0.7))
    peak_s = bounded_next(start_s, rise_s, duration_s)
    hold_end_s = bounded_next(peak_s, hold_s, duration_s)
    return_end_s = bounded_next(hold_end_s, return_s, duration_s)

    return {
        "name": f"step_steer_{family_index:05d}",
        "family": "step_steer",
        "duration_s": duration_s,
        "warmup_s": warmup_s,
        "driver": {
            "steering": piecewise_profile(
                [
                    (0.0, 0.0),
                    (start_s, 0.0),
                    (peak_s, amplitude),
                    (hold_end_s, amplitude),
                    (return_end_s, 0.0),
                    (duration_s, 0.0),
                ]
            ),
            "throttle": build_hold_throttle_profile(rng, family_cfg, warmup_s, duration_s, (0.15, 0.45)),
            "braking": constant_profile(0.0),
        },
    }


def build_doublet_steer_scenario(
    rng: random.Random,
    generator_cfg: dict[str, Any],
    family_cfg: dict[str, Any],
    family_index: int,
) -> dict[str, Any]:
    warmup_s = default_warmup(generator_cfg, family_cfg)
    duration_s = sample_range(rng, family_cfg, "duration_s_range", (10.0, 14.0))
    sign = random_sign(rng)
    amplitude = sign * sample_range(rng, family_cfg, "steering_amplitude_range", (0.06, 0.28))
    amplitude_ratio = sample_range(rng, family_cfg, "second_pulse_ratio_range", (0.7, 1.2))
    start_s = warmup_s + sample_range(rng, family_cfg, "steer_start_offset_s_range", (0.7, 2.2))
    pulse_s = sample_range(rng, family_cfg, "pulse_width_s_range", (0.3, 1.0))
    gap_s = sample_range(rng, family_cfg, "pulse_gap_s_range", (0.15, 0.6))
    release_s = sample_range(rng, family_cfg, "steer_return_s_range", (0.2, 0.6))

    pulse_1_end_s = bounded_next(start_s, pulse_s, duration_s)
    middle_s = bounded_next(pulse_1_end_s, gap_s, duration_s)
    pulse_2_end_s = bounded_next(middle_s, pulse_s, duration_s)
    final_end_s = bounded_next(pulse_2_end_s, release_s, duration_s)

    return {
        "name": f"doublet_steer_{family_index:05d}",
        "family": "doublet_steer",
        "duration_s": duration_s,
        "warmup_s": warmup_s,
        "driver": {
            "steering": piecewise_profile(
                [
                    (0.0, 0.0),
                    (start_s, 0.0),
                    (pulse_1_end_s, amplitude),
                    (middle_s, 0.0),
                    (pulse_2_end_s, -amplitude * amplitude_ratio),
                    (final_end_s, 0.0),
                    (duration_s, 0.0),
                ]
            ),
            "throttle": build_hold_throttle_profile(rng, family_cfg, warmup_s, duration_s, (0.15, 0.4)),
            "braking": constant_profile(0.0),
        },
    }


def build_sine_steer_scenario(
    rng: random.Random,
    generator_cfg: dict[str, Any],
    family_cfg: dict[str, Any],
    family_index: int,
) -> dict[str, Any]:
    warmup_s = default_warmup(generator_cfg, family_cfg)
    duration_s = sample_range(rng, family_cfg, "duration_s_range", (10.0, 14.0))
    start_s = warmup_s + sample_range(rng, family_cfg, "steer_start_offset_s_range", (0.5, 1.5))
    end_margin_s = sample_range(rng, family_cfg, "end_margin_s_range", (0.5, 1.5))
    end_s = max(start_s + 1.0, duration_s - end_margin_s)
    amplitude = sample_range(rng, family_cfg, "steering_amplitude_range", (0.05, 0.24))
    frequency_hz = sample_range(rng, family_cfg, "frequency_hz_range", (0.2, 1.0))

    return {
        "name": f"sine_steer_{family_index:05d}",
        "family": "sine_steer",
        "duration_s": duration_s,
        "warmup_s": warmup_s,
        "driver": {
            "steering": sine_profile(amplitude, frequency_hz, start_s, end_s),
            "throttle": build_hold_throttle_profile(rng, family_cfg, warmup_s, duration_s, (0.15, 0.45)),
            "braking": constant_profile(0.0),
        },
    }


def build_chirp_steer_scenario(
    rng: random.Random,
    generator_cfg: dict[str, Any],
    family_cfg: dict[str, Any],
    family_index: int,
) -> dict[str, Any]:
    warmup_s = default_warmup(generator_cfg, family_cfg)
    duration_s = sample_range(rng, family_cfg, "duration_s_range", (10.0, 14.0))
    start_s = warmup_s + sample_range(rng, family_cfg, "steer_start_offset_s_range", (0.5, 1.5))
    end_margin_s = sample_range(rng, family_cfg, "end_margin_s_range", (0.5, 1.5))
    end_s = max(start_s + 1.5, duration_s - end_margin_s)
    amplitude = sample_range(rng, family_cfg, "steering_amplitude_range", (0.05, 0.22))
    f0 = sample_range(rng, family_cfg, "start_frequency_hz_range", (0.1, 0.4))
    f1 = sample_range(rng, family_cfg, "end_frequency_hz_range", (0.8, 1.8))

    return {
        "name": f"chirp_steer_{family_index:05d}",
        "family": "chirp_steer",
        "duration_s": duration_s,
        "warmup_s": warmup_s,
        "driver": {
            "steering": chirp_profile(amplitude, f0, f1, start_s, end_s),
            "throttle": build_hold_throttle_profile(rng, family_cfg, warmup_s, duration_s, (0.15, 0.4)),
            "braking": constant_profile(0.0),
        },
    }


def build_steer_brake_scenario(
    rng: random.Random,
    generator_cfg: dict[str, Any],
    family_cfg: dict[str, Any],
    family_index: int,
) -> dict[str, Any]:
    warmup_s = default_warmup(generator_cfg, family_cfg)
    duration_s = sample_range(rng, family_cfg, "duration_s_range", (10.0, 14.0))
    sign = random_sign(rng)
    amplitude = sign * sample_range(rng, family_cfg, "steering_amplitude_range", (0.08, 0.25))
    steer_start_s = warmup_s + sample_range(rng, family_cfg, "steer_start_offset_s_range", (0.7, 2.2))
    steer_rise_s = sample_range(rng, family_cfg, "steer_rise_s_range", (0.2, 0.6))
    steer_hold_s = sample_range(rng, family_cfg, "steer_hold_s_range", (1.0, 2.5))
    steer_return_s = sample_range(rng, family_cfg, "steer_return_s_range", (0.2, 0.7))
    steer_peak_s = bounded_next(steer_start_s, steer_rise_s, duration_s)
    steer_hold_end_s = bounded_next(steer_peak_s, steer_hold_s, duration_s)
    steer_return_end_s = bounded_next(steer_hold_end_s, steer_return_s, duration_s)

    brake_peak = sample_range(rng, family_cfg, "brake_peak_range", (0.15, 0.55))
    brake_start_s = bounded_next(
        steer_peak_s,
        sample_range(rng, family_cfg, "brake_start_offset_s_range", (0.2, 0.8)),
        duration_s,
    )
    brake_profile = build_brake_pulse_profile(
        brake_start_s=brake_start_s,
        duration_s=duration_s,
        brake_peak=brake_peak,
        brake_rise_s=sample_range(rng, family_cfg, "brake_rise_s_range", (0.2, 0.7)),
        brake_hold_s=sample_range(rng, family_cfg, "brake_hold_s_range", (0.6, 1.8)),
        brake_release_s=sample_range(rng, family_cfg, "brake_release_s_range", (0.2, 0.8)),
    )

    return {
        "name": f"steer_brake_{family_index:05d}",
        "family": "steer_brake",
        "duration_s": duration_s,
        "warmup_s": warmup_s,
        "driver": {
            "steering": piecewise_profile(
                [
                    (0.0, 0.0),
                    (steer_start_s, 0.0),
                    (steer_peak_s, amplitude),
                    (steer_hold_end_s, amplitude),
                    (steer_return_end_s, 0.0),
                    (duration_s, 0.0),
                ]
            ),
            "throttle": build_hold_throttle_profile(rng, family_cfg, warmup_s, duration_s, (0.2, 0.55)),
            "braking": brake_profile,
        },
    }
