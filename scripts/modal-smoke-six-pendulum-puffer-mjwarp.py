#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path

import modal


app = modal.App("six-pendulum-puffer-mjwarp-smoke")
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "mujoco-warp==3.9.0.1",
    "numpy==2.2.6",
    "pufferlib",
    "torch==2.7.1",
)


def score_batch(qpos, qvel, last_action, links: int) -> dict:
    import numpy as np

    safe_links = max(1, min(6, int(links)))
    relative = ((qpos[:, 1 : 1 + safe_links] + np.pi) % (2 * np.pi)) - np.pi
    absolute = np.cumsum(relative, axis=1)
    absolute = ((absolute + np.pi) % (2 * np.pi)) - np.pi
    rel_vel = qvel[:, 1 : 1 + safe_links]
    abs_vel = np.cumsum(rel_vel, axis=1)

    max_upright_error = np.max(np.abs(absolute), axis=1)
    max_bend_error = np.max(np.abs(relative[:, 1:]), axis=1) if safe_links > 1 else np.zeros(qpos.shape[0])
    mean_upright_error = np.mean(np.abs(absolute), axis=1)
    mean_speed = np.mean(np.abs(abs_vel), axis=1)
    mean_tip_height = np.mean((np.cos(absolute) + 1.0) * 0.5, axis=1)
    near_top = (max_upright_error < 0.16) & (max_bend_error < 0.14)
    strict_score = np.where(
        near_top,
        np.clip(
            100.0
            - np.sum(np.abs(absolute) * np.arange(1, safe_links + 1), axis=1) * 12.0
            - max_bend_error * 30.0
            - np.sum(np.abs(abs_vel), axis=1) * 2.5
            - np.abs(qpos[:, 0]) * 8.0,
            0.0,
            100.0,
        ),
        0.0,
    )
    dense_alignment = np.exp(-mean_upright_error * 1.15 - max_bend_error * 2.0 - mean_speed * 0.08)
    whip = (np.abs(absolute[:, -1]) < 0.65) & (mean_speed > 1.0)
    reward = mean_tip_height * 0.08 + dense_alignment * 0.1 + (strict_score / 100.0) ** 2 * 1.8
    reward += np.where(whip, 0.1, 0.0)
    reward -= np.abs(qpos[:, 0]) * 0.015
    reward -= (last_action / 32.0) ** 2 * 0.002

    obs = np.zeros((qpos.shape[0], 3 + 6 * 5), dtype=np.float32)
    obs[:, 0] = qpos[:, 0]
    obs[:, 1] = qvel[:, 0] / 5.0
    obs[:, 2] = last_action / 32.0
    cursor = 3
    for index in range(6):
        if index < safe_links:
            obs[:, cursor] = np.sin(absolute[:, index])
            obs[:, cursor + 1] = np.cos(absolute[:, index])
            obs[:, cursor + 2] = np.sin(relative[:, index])
            obs[:, cursor + 3] = np.cos(relative[:, index])
            obs[:, cursor + 4] = abs_vel[:, index] / 8.0
        cursor += 5

    return {
        "observation": obs,
        "reward": reward,
        "strictScore": strict_score,
        "meanTipHeight": mean_tip_height,
        "maxUprightError": max_upright_error,
        "maxChainBend": max_bend_error,
        "whip": whip.astype(np.float32),
    }


def run_env_contract_smoke(mjcf_xml: str, links: int = 1, nworld: int = 4, steps: int = 64) -> dict:
    import mujoco
    import mujoco_warp as mjw
    import numpy as np
    import pufferlib
    import warp as wp

    started = time.time()
    safe_links = max(1, min(6, int(links)))
    safe_nworld = max(2, int(nworld))
    safe_steps = max(1, int(steps))
    control_dt = 0.0025

    mjm = mujoco.MjModel.from_xml_string(mjcf_xml)
    m = mjw.put_model(mjm)
    d = mjw.make_data(mjm, nworld=safe_nworld)

    def reset_pose(pose: str):
        qpos = d.qpos.numpy()
        qvel = d.qvel.numpy()
        qpos.fill(0.0)
        qvel.fill(0.0)
        qpos[:, 0] = np.linspace(-0.04, 0.04, safe_nworld)
        qvel[:, 0] = np.linspace(-0.03, 0.03, safe_nworld)
        if pose == "hold":
            base = np.linspace(-0.04, 0.04, safe_nworld)
            for index in range(safe_links):
                qpos[:, 1 + index] = base * (index + 1)
        else:
            base = np.linspace(-0.06, 0.06, safe_nworld)
            for index in range(safe_links):
                qpos[:, 1 + index] = np.pi - index * 0.05 + base
        d.qpos.assign(qpos)
        d.qvel.assign(qvel)
        mjw.forward(m, d)
        wp.synchronize()

    def rollout(pose: str):
        reset_pose(pose)
        max_strict_score = np.zeros(safe_nworld, dtype=np.float32)
        held_steps = np.zeros(safe_nworld, dtype=np.int32)
        max_held_steps = np.zeros(safe_nworld, dtype=np.int32)
        last_action = np.zeros(safe_nworld, dtype=np.float32)
        first_metrics = None
        final_metrics = None
        for step_index in range(safe_steps):
            qpos = d.qpos.numpy()
            qvel = d.qvel.numpy()
            if pose == "hold":
                action = np.clip(-18.0 * qpos[:, 0] - 5.0 * qvel[:, 0], -32.0, 32.0)
            else:
                phase = step_index / max(1, safe_steps - 1)
                action = np.sin(phase * np.pi * 4.0) * 18.0 - qpos[:, 0] * 3.0
            d.ctrl.assign(action.reshape(safe_nworld, 1).astype(np.float32))
            last_action = action.astype(np.float32)
            mjw.step(m, d)
            wp.synchronize()
            metrics = score_batch(d.qpos.numpy(), d.qvel.numpy(), last_action, safe_links)
            if first_metrics is None:
                first_metrics = metrics
            final_metrics = metrics
            max_strict_score = np.maximum(max_strict_score, metrics["strictScore"])
            is_held = metrics["strictScore"] > 82.0
            held_steps = np.where(is_held, held_steps + 1, 0)
            max_held_steps = np.maximum(max_held_steps, held_steps)

        return {
            "pose": pose,
            "observationShape": list(first_metrics["observation"].shape),
            "initialMeanTipHeight": float(np.mean(first_metrics["meanTipHeight"])),
            "finalMeanTipHeight": float(np.mean(final_metrics["meanTipHeight"])),
            "finalMeanReward": float(np.mean(final_metrics["reward"])),
            "maxStrictScore": float(np.max(max_strict_score)),
            "meanStrictScore": float(np.mean(final_metrics["strictScore"])),
            "maxHeldSeconds": float(np.max(max_held_steps) * control_dt),
            "solvedOneSecond": bool(np.max(max_held_steps) * control_dt >= 1.0),
            "whipFraction": float(np.mean(final_metrics["whip"])),
        }

    down = rollout("down")
    hold = rollout("hold")
    elapsed = time.time() - started
    return {
        "schema": "six-pendulum-mjwarp-env-contract-smoke-v1",
        "status": "env-contract-smoke-passed",
        "links": safe_links,
        "nworld": safe_nworld,
        "steps": safe_steps,
        "elapsedSeconds": elapsed,
        "model": {
            "nq": int(mjm.nq),
            "nv": int(mjm.nv),
            "nu": int(mjm.nu),
            "timestep": float(mjm.opt.timestep),
            "gravity": [float(value) for value in mjm.opt.gravity],
            "dofDamping": [float(value) for value in mjm.dof_damping],
            "dofFrictionLoss": [float(value) for value in mjm.dof_frictionloss],
        },
        "versions": {
            "mujoco": getattr(mujoco, "__version__", "unknown"),
            "mujocoWarp": getattr(mjw, "__version__", "unknown"),
            "pufferlib": getattr(pufferlib, "__version__", "unknown"),
            "warp": getattr(wp, "__version__", "unknown"),
        },
        "devices": [str(device) for device in wp.get_devices()],
        "gates": {
            "strictScoreRequiresUpright": True,
            "oneSecondMinimum": True,
            "downStartSolved": down["solvedOneSecond"],
            "holdStartSolved": hold["solvedOneSecond"],
        },
        "rollouts": [down, hold],
        "nextRequiredWork": [
            "Move reset, action, observation, reward, and strict-score math into GPU-side kernels.",
            "Wrap the contract as a PufferLib environment.",
            "Train one-link recurrent PPO from down start and promote only after held-out one-second validation.",
        ],
    }


def run_mjwarp_smoke(mjcf_xml: str, links: int = 1, nworld: int = 1024, steps: int = 256) -> dict:
    import mujoco
    import mujoco_warp as mjw
    import pufferlib
    import warp as wp

    started = time.time()
    safe_links = max(1, min(6, int(links)))
    safe_nworld = max(1, int(nworld))
    safe_steps = max(1, int(steps))

    mjm = mujoco.MjModel.from_xml_string(mjcf_xml)
    m = mjw.put_model(mjm)
    d = mjw.make_data(mjm, nworld=safe_nworld)

    wp.synchronize()
    direct_started = time.time()
    for _ in range(min(8, safe_steps)):
        mjw.step(m, d)
    wp.synchronize()
    direct_steps = min(8, safe_steps)
    direct_elapsed = time.time() - direct_started

    graph_steps = max(0, safe_steps - direct_steps)
    graph_elapsed = 0.0
    graph_captured = False
    graph_error = None
    if graph_steps:
        try:
            with wp.ScopedCapture() as capture:
                mjw.step(m, d)
            graph_captured = True
            graph_started = time.time()
            for _ in range(graph_steps):
                wp.capture_launch(capture.graph)
            wp.synchronize()
            graph_elapsed = time.time() - graph_started
        except Exception as exc:  # CPU Warp builds can step, but cannot CUDA-capture.
            graph_error = f"{type(exc).__name__}: {exc}"
            fallback_started = time.time()
            for _ in range(graph_steps):
                mjw.step(m, d)
            wp.synchronize()
            graph_elapsed = time.time() - fallback_started

    elapsed = time.time() - started
    total_steps = safe_nworld * safe_steps
    sps = total_steps / elapsed if elapsed > 0 else 0.0

    return {
        "schema": "six-pendulum-puffer-mjwarp-smoke-v1",
        "status": "substrate-smoke-passed",
        "links": safe_links,
        "nworld": safe_nworld,
        "steps": safe_steps,
        "simulatedSteps": total_steps,
        "elapsedSeconds": elapsed,
        "stepsPerSecond": sps,
        "directElapsedSeconds": direct_elapsed,
        "graphElapsedSeconds": graph_elapsed,
        "graphCaptured": graph_captured,
        "graphError": graph_error,
        "model": {
            "nq": int(mjm.nq),
            "nv": int(mjm.nv),
            "nu": int(mjm.nu),
            "timestep": float(mjm.opt.timestep),
            "gravity": [float(value) for value in mjm.opt.gravity],
            "dofDamping": [float(value) for value in mjm.dof_damping],
            "dofFrictionLoss": [float(value) for value in mjm.dof_frictionloss],
            "geomContype": [int(value) for value in mjm.geom_contype],
            "geomConaffinity": [int(value) for value in mjm.geom_conaffinity],
        },
        "versions": {
            "mujoco": getattr(mujoco, "__version__", "unknown"),
            "mujocoWarp": getattr(mjw, "__version__", "unknown"),
            "pufferlib": getattr(pufferlib, "__version__", "unknown"),
            "warp": getattr(wp, "__version__", "unknown"),
        },
        "devices": [str(device) for device in wp.get_devices()],
        "nextRequiredWork": [
            "Port observations, rewards, resets, and controls into a true batched MJWarp environment.",
            "Attach PufferPPO with a recurrent MinGRU/PufferNet policy.",
            "Run 1-link down-start sweeps first; unlock 2 links only after held-out validation holds for at least 1 second.",
            "Enable randomized episode horizons only after whip behavior appears.",
        ],
        "sourceConstraints": [
            "Yacine thread: PufferPPO/PufferLib, puffer MinGRU, about 1m parameter policy.",
            "Yacine thread: MuJoCo Warp plus CUDA graph capture for wallclock speed.",
            "Yacine thread: gravity 9.8, no hinge friction, cart tracks 0.",
            "Yacine thread: about 3.6k experiments; report wallclock versus strict score.",
        ],
    }


@app.function(image=image, gpu="L4", timeout=1800)
def smoke_puffer_mjwarp(mjcf_xml: str, links: int = 1, nworld: int = 1024, steps: int = 256) -> str:
    return json.dumps(run_mjwarp_smoke(mjcf_xml, links, nworld, steps), indent=2)


@app.local_entrypoint()
def main(links: int = 1, nworld: int = 1024, steps: int = 256):
    path = Path(f"app/ailab/six-pendulum-cartpole/mjcf/cartpole_{links}_link.xml")
    if not path.exists():
        raise FileNotFoundError(f"Missing MJCF file: {path}")
    return smoke_puffer_mjwarp.remote(path.read_text(), links, nworld, steps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the six-pendulum MJWarp substrate smoke locally.")
    parser.add_argument("--links", type=int, default=1)
    parser.add_argument("--nworld", type=int, default=4)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--contract", action="store_true")
    parser.add_argument(
        "--write-result",
        type=Path,
        default=Path("local-artifacts/outputs/training-checkpoints/puffer-mjwarp-local-substrate-smoke.json"),
    )
    args = parser.parse_args()

    mjcf_path = Path(f"app/ailab/six-pendulum-cartpole/mjcf/cartpole_{args.links}_link.xml")
    if not mjcf_path.exists():
        raise FileNotFoundError(f"Missing MJCF file: {mjcf_path}")

    if args.contract:
        result = run_env_contract_smoke(mjcf_path.read_text(), args.links, args.nworld, args.steps)
    else:
        result = run_mjwarp_smoke(mjcf_path.read_text(), args.links, args.nworld, args.steps)
    args.write_result.parent.mkdir(parents=True, exist_ok=True)
    args.write_result.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
