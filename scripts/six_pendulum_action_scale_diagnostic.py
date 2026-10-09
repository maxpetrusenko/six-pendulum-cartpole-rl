#!/usr/bin/env python3
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from six_pendulum_mjwarp_gpu_kernels import DEFAULT_ACTION_SCALE, WarpScoreKernel


DEFAULT_OUTPUT = Path(
    "local-artifacts/outputs/training-checkpoints/puffer-mjwarp-action-scale-diagnostic.json"
)


def action_value(pattern: str, step_index: int, total_steps: int, control_dt: float) -> float:
    if pattern == "positive":
        return 1.0
    if pattern == "negative":
        return -1.0
    if pattern == "flip-quarter":
        period_steps = max(1, int(round(0.25 / control_dt)))
        return 1.0 if (step_index // period_steps) % 2 == 0 else -1.0
    if pattern == "flip-eighth":
        period_steps = max(1, int(round(0.125 / control_dt)))
        return 1.0 if (step_index // period_steps) % 2 == 0 else -1.0
    if pattern == "sine":
        phase = step_index / max(1, total_steps - 1)
        return math.sin(phase * 2.0 * math.pi)
    raise ValueError(f"Unsupported pattern: {pattern}")


def run_pattern(
    mjcf_xml: str,
    links: int,
    nworld: int,
    steps: int,
    force_scale: float,
    pattern: str,
    seed: int,
) -> dict:
    import mujoco
    import mujoco_warp as mjw

    started = time.time()
    mjm = mujoco.MjModel.from_xml_string(mjcf_xml)
    model = mjw.put_model(mjm)
    data = mjw.make_data(mjm, nworld=int(nworld))
    device = str(getattr(data.qpos, "device", "cpu"))
    runner = WarpScoreKernel(
        nworld=nworld,
        links=links,
        action_scale=force_scale,
        device=device,
        terminal_boundary=2.35,
    )
    runner.reset_worlds(data.qpos, data.qvel, data.ctrl, "down", seed, reset_all=True)
    mjw.forward(model, data)
    runner.score_device(data.qpos, data.qvel, synchronize=True)
    runner.initialize_prev_potential_from_current(synchronize=True)

    control_dt = float(mjm.opt.timestep)
    min_abs_tip_angle = np.full(int(nworld), np.inf, dtype=np.float32)
    max_tip_height = np.zeros(int(nworld), dtype=np.float32)
    max_abs_cart = np.zeros(int(nworld), dtype=np.float32)
    max_abs_cart_velocity = np.zeros(int(nworld), dtype=np.float32)
    max_abs_tip_velocity = np.zeros(int(nworld), dtype=np.float32)
    terminal_seen = np.zeros(int(nworld), dtype=bool)
    first_near_vertical_step = np.full(int(nworld), -1, dtype=np.int32)

    for step_index in range(int(steps)):
        normalized = action_value(pattern, step_index, int(steps), control_dt)
        actions = np.full((int(nworld), 1), normalized, dtype=np.float32)
        runner.apply_actions(actions, data.ctrl)
        mjw.step(model, data)
        runner.score_device(data.qpos, data.qvel, synchronize=True)
        runner.post_step_device(False, int(steps) + 1, synchronize=True)

        qpos = data.qpos.numpy()
        qvel = data.qvel.numpy()
        running_angle = np.zeros(int(nworld), dtype=np.float32)
        running_velocity = np.zeros(int(nworld), dtype=np.float32)
        for link_index in range(int(links)):
            relative = ((qpos[:, 1 + link_index] + np.pi) % (2.0 * np.pi)) - np.pi
            running_angle = ((running_angle + relative + np.pi) % (2.0 * np.pi)) - np.pi
            running_velocity += qvel[:, 1 + link_index]
        abs_tip_angle = np.abs(running_angle)
        tip_height = (np.cos(running_angle) + 1.0) * 0.5
        min_abs_tip_angle = np.minimum(min_abs_tip_angle, abs_tip_angle)
        max_tip_height = np.maximum(max_tip_height, tip_height)
        max_abs_cart = np.maximum(max_abs_cart, np.abs(qpos[:, 0]))
        max_abs_cart_velocity = np.maximum(max_abs_cart_velocity, np.abs(qvel[:, 0]))
        max_abs_tip_velocity = np.maximum(max_abs_tip_velocity, np.abs(running_velocity))
        terminal_seen |= runner.terminal_wp.numpy() > 0.5
        near_vertical = (abs_tip_angle < 0.25) & (first_near_vertical_step < 0)
        first_near_vertical_step[near_vertical] = step_index + 1

    strict_score = runner.rollout_max_strict_score_wp.numpy()
    max_held_steps = runner.rollout_max_held_steps_wp.numpy()
    reached_near_vertical = first_near_vertical_step >= 0
    return {
        "pattern": pattern,
        "links": int(links),
        "nworld": int(nworld),
        "steps": int(steps),
        "seconds": float(steps) * control_dt,
        "forceScale": float(force_scale),
        "seed": int(seed),
        "elapsedSeconds": time.time() - started,
        "device": device,
        "minAbsTipAngleRad": float(np.min(min_abs_tip_angle)),
        "minAbsTipAngleDeg": float(np.min(min_abs_tip_angle) * 180.0 / np.pi),
        "maxTipHeight": float(np.max(max_tip_height)),
        "reachedNearVerticalWorlds": int(np.sum(reached_near_vertical)),
        "firstNearVerticalSecondsMin": float(np.min(first_near_vertical_step[reached_near_vertical]) * control_dt)
        if np.any(reached_near_vertical)
        else None,
        "maxStrictScore": float(np.max(strict_score)) if strict_score.size else 0.0,
        "maxHeldSeconds": float(np.max(max_held_steps) * control_dt) if max_held_steps.size else 0.0,
        "terminalWorlds": int(np.sum(terminal_seen)),
        "maxAbsCart": float(np.max(max_abs_cart)),
        "maxAbsCartVelocity": float(np.max(max_abs_cart_velocity)),
        "maxAbsTipVelocity": float(np.max(max_abs_tip_velocity)),
        "reachableForSwingup": bool(np.any(reached_near_vertical)),
        "notes": [
            "This diagnostic uses open-loop max-force actions only; it is not a policy and does not count toward solve.",
            "Near vertical is abs tip angle < 0.25 rad from upright.",
            "Cart terminal is still measured at the normal training boundary; no resets are applied during the diagnostic.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="Check one-link action-scale reachability with open-loop max-force patterns.")
    parser.add_argument("--links", type=int, default=1)
    parser.add_argument("--nworld", type=int, default=16)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--force-scales", type=float, nargs="+", default=[DEFAULT_ACTION_SCALE, 64.0, 120.0, 240.0])
    parser.add_argument("--patterns", nargs="+", default=["positive", "negative", "flip-quarter", "flip-eighth", "sine"])
    parser.add_argument("--seed", type=int, default=426210)
    parser.add_argument("--write-result", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    mjcf_path = Path(f"app/ailab/six-pendulum-cartpole/mjcf/cartpole_{args.links}_link.xml")
    if not mjcf_path.exists():
        raise FileNotFoundError(f"Missing MJCF file: {mjcf_path}")
    started = time.time()
    results = []
    for force_scale in args.force_scales:
        for pattern in args.patterns:
            results.append(
                run_pattern(
                    mjcf_path.read_text(),
                    args.links,
                    args.nworld,
                    args.steps,
                    force_scale,
                    pattern,
                    args.seed,
                )
            )

    reachable = [row for row in results if row["reachableForSwingup"]]
    result = {
        "schema": "six-pendulum-action-scale-diagnostic-v1",
        "status": "action-scale-diagnostic-finished",
        "links": int(args.links),
        "nworld": int(args.nworld),
        "steps": int(args.steps),
        "forceScales": [float(value) for value in args.force_scales],
        "patterns": args.patterns,
        "elapsedSeconds": time.time() - started,
        "results": results,
        "bestReachability": sorted(
            reachable,
            key=lambda row: (
                row["firstNearVerticalSecondsMin"] if row["firstNearVerticalSecondsMin"] is not None else 999.0,
                -row["maxTipHeight"],
            ),
        )[0]
        if reachable
        else None,
        "gates": {
            "learnedPolicyOnly": False,
            "countsTowardSolve": False,
            "purpose": "verify action authority before PPO/down-start training",
        },
    }
    args.write_result.parent.mkdir(parents=True, exist_ok=True)
    args.write_result.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
