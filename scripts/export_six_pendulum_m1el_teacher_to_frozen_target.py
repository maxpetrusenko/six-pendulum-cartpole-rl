#!/usr/bin/env python3
"""Port m1el direct-collocation/TVLQR controls into the frozen MuJoCo target."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

import validate_six_pendulum_frozen_target as frozen_validator


OBS_DIM = 33
MAX_LINKS = 6


def wrap_pi(values: np.ndarray) -> np.ndarray:
    return (values + np.pi) % (2.0 * np.pi) - np.pi


def add_m1el_path(path: Path) -> None:
    resolved = path.resolve()
    if str(resolved) not in sys.path:
        sys.path.insert(0, str(resolved))


def interp_columns(times: np.ndarray, values: np.ndarray, t: float) -> np.ndarray:
    flat = values.reshape(values.shape[0], -1)
    out = np.asarray([np.interp(t, times, flat[:, index]) for index in range(flat.shape[1])])
    return out.reshape(values.shape[1:])


def simulate_m1el_source(
    repo: Path,
    controls_path: Path,
    target_dt: float,
    hold_seconds: float,
) -> dict[str, Any]:
    add_m1el_path(repo)
    from pendulum.balance import upright_lqr
    from pendulum.dynamics import Chain
    from pendulum.sim import rk4_step

    bundle = dict(np.load(controls_path))
    n = int(bundle["n"])
    source_dt = float(bundle["dt"])
    chain = Chain(n, float(bundle["g"]))
    times = np.asarray(bundle["t"], dtype=np.float64)
    theta_nom = np.asarray(bundle["theta_nom"], dtype=np.float64)
    thetad_nom = np.asarray(bundle["thetad_nom"], dtype=np.float64)
    a_ff = np.asarray(bundle["a_ff"], dtype=np.float64)
    v_nom = np.asarray(bundle["v_nom"], dtype=np.float64)
    gains = np.asarray(bundle["K"], dtype=np.float64)
    swing_seconds = float(times[-1])

    y = np.concatenate([theta_nom[0], thetad_nom[0]])
    x = 0.0
    v = 0.0

    theta_rows: list[np.ndarray] = []
    thetad_rows: list[np.ndarray] = []
    x_rows: list[float] = []
    v_rows: list[float] = []
    accel_rows: list[float] = []

    swing_steps = int(round(swing_seconds / target_dt))
    gain_times = times[:-1]
    for step in range(swing_steps):
        t = min(step * target_dt, gain_times[-1])
        theta_ref = interp_columns(times, theta_nom, t)
        thetad_ref = interp_columns(times, thetad_nom, t)
        v_ref = float(np.interp(t, times, v_nom))
        k_ref = interp_columns(gain_times, gains, t)
        a_ref = float(np.interp(t, gain_times, a_ff[: len(gain_times)]))
        z = np.concatenate([y[:n], y[n:], [v]])
        z_ref = np.concatenate([theta_ref, thetad_ref, [v_ref]])
        accel = a_ref - float((k_ref @ (z - z_ref))[0])
        theta_rows.append(y[:n].copy())
        thetad_rows.append(y[n:].copy())
        x_rows.append(float(x))
        v_rows.append(float(v))
        accel_rows.append(float(accel))
        y_new = rk4_step(chain, y, accel, target_dt)
        v_new = v + accel * target_dt
        x += 0.5 * (v + v_new) * target_dt
        y, v = y_new, v_new

    balance_gain, _, _, _ = upright_lqr(chain, target_dt, r=0.01, q_theta=100)
    hold_steps = int(round(hold_seconds / target_dt))
    for _ in range(hold_steps):
        z_balance = np.concatenate([wrap_pi(y[:n]), y[n:], [x, v]])
        accel = -float((balance_gain @ z_balance)[0])
        theta_rows.append(y[:n].copy())
        thetad_rows.append(y[n:].copy())
        x_rows.append(float(x))
        v_rows.append(float(v))
        accel_rows.append(float(accel))
        y_new = rk4_step(chain, y, accel, target_dt)
        v_new = v + accel * target_dt
        x += 0.5 * (v + v_new) * target_dt
        y, v = y_new, v_new

    return {
        "sourceBundle": bundle,
        "sourceDt": source_dt,
        "targetDt": target_dt,
        "links": n,
        "thetaAbs": np.asarray(theta_rows, dtype=np.float64),
        "thetadAbs": np.asarray(thetad_rows, dtype=np.float64),
        "cartX": np.asarray(x_rows, dtype=np.float64),
        "cartV": np.asarray(v_rows, dtype=np.float64),
        "cartAcceleration": np.asarray(accel_rows, dtype=np.float64),
        "sourceSwingSeconds": swing_seconds,
        "sourceHoldSeconds": hold_seconds,
    }


def absolute_to_mujoco_relative(
    theta_abs: np.ndarray,
    thetad_abs: np.ndarray,
    cart_x: np.ndarray,
    cart_v: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rel = np.zeros_like(theta_abs)
    rel[:, 0] = theta_abs[:, 0]
    rel[:, 1:] = wrap_pi(theta_abs[:, 1:] - theta_abs[:, :-1])
    rel_vel = np.zeros_like(thetad_abs)
    rel_vel[:, 0] = thetad_abs[:, 0]
    rel_vel[:, 1:] = thetad_abs[:, 1:] - thetad_abs[:, :-1]
    qpos = np.column_stack([cart_x, rel])
    qvel = np.column_stack([cart_v, rel_vel])
    qpos[:, 1:] = wrap_pi(qpos[:, 1:])
    return qpos, qvel


def force_for_cart_acceleration(model, data, qpos: np.ndarray, qvel: np.ndarray, cart_acceleration: float) -> float:
    import mujoco

    data.qpos[:] = qpos
    data.qvel[:] = qvel
    mujoco.mj_forward(model, data)
    mass = np.zeros((model.nv, model.nv), dtype=np.float64)
    mujoco.mj_fullM(model, mass, data.qM)
    passive = np.asarray(data.qfrc_passive).copy()
    bias = np.asarray(data.qfrc_bias).copy()
    hinge_acceleration = np.linalg.solve(
        mass[1:, 1:],
        -(mass[1:, 0] * cart_acceleration + bias[1:] - passive[1:]),
    )
    return float(mass[0, 0] * cart_acceleration + mass[0, 1:] @ hinge_acceleration + bias[0] - passive[0])


def convert_acceleration_to_forces(
    qpos: np.ndarray,
    qvel: np.ndarray,
    accelerations: np.ndarray,
    links: int,
    profile: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    import mujoco

    model = frozen_validator.make_mujoco_model(links, profile)
    data = mujoco.MjData(model)
    force_cap = float(model.actuator_ctrlrange[0, 1])
    raw_forces = np.asarray(
        [force_for_cart_acceleration(model, data, qpos[index], qvel[index], accelerations[index]) for index in range(len(accelerations))],
        dtype=np.float64,
    )
    clipped = np.clip(raw_forces, -force_cap, force_cap)
    stats = {
        "forceCap": force_cap,
        "rawForceAbsMax": float(np.max(np.abs(raw_forces))) if raw_forces.size else 0.0,
        "clippedForceAbsMax": float(np.max(np.abs(clipped))) if clipped.size else 0.0,
        "forceClipFraction": float(np.mean(np.abs(raw_forces) > force_cap)) if raw_forces.size else 0.0,
    }
    return clipped, stats


def encode_observations(qpos: np.ndarray, qvel: np.ndarray, controls: np.ndarray, links: int, force_cap: float) -> np.ndarray:
    observations = np.zeros((len(controls), OBS_DIM), dtype=np.float32)
    relative = wrap_pi(qpos[:, 1 : links + 1])
    absolute = wrap_pi(np.cumsum(relative, axis=1))
    abs_vel = np.cumsum(qvel[:, 1 : links + 1], axis=1)
    observations[:, 0] = qpos[:, 0]
    observations[:, 1] = qvel[:, 0] / 5.0
    observations[:, 2] = controls / force_cap
    cursor = 3
    for index in range(MAX_LINKS):
        if index < links:
            observations[:, cursor] = np.sin(absolute[:, index])
            observations[:, cursor + 1] = np.cos(absolute[:, index])
            observations[:, cursor + 2] = np.sin(relative[:, index])
            observations[:, cursor + 3] = np.cos(relative[:, index])
            observations[:, cursor + 4] = abs_vel[:, index] / 8.0
        cursor += 5
    return observations


def export_m1el_teacher(
    m1el_repo: Path,
    controls_path: Path,
    profile: str,
    hold_seconds: float,
    write_trajectories: Path,
    write_summary: Path,
) -> dict[str, Any]:
    started = time.time()
    target_dt = float(frozen_validator.TARGET["dt"])
    source = simulate_m1el_source(m1el_repo, controls_path, target_dt, hold_seconds)
    links = int(source["links"])
    if links > MAX_LINKS:
        raise ValueError(f"export supports at most {MAX_LINKS} links, got {links}")
    source_qpos, source_qvel = absolute_to_mujoco_relative(
        source["thetaAbs"],
        source["thetadAbs"],
        source["cartX"],
        source["cartV"],
    )
    controls, force_stats = convert_acceleration_to_forces(
        source_qpos,
        source_qvel,
        source["cartAcceleration"],
        links,
        profile,
    )
    replay_qpos, replay_qvel, replay_dt, replay_force_cap = frozen_validator.replay_controls(
        links,
        profile,
        source_qpos[0],
        source_qvel[0],
        controls,
    )
    metrics = frozen_validator.strict_metrics(replay_qpos, replay_qvel, links, replay_dt)
    observations = encode_observations(replay_qpos, replay_qvel, controls, links, replay_force_cap)
    actions = (controls / replay_force_cap).reshape(-1, 1).astype(np.float32)

    metadata = {
        "schema": "six-pendulum-m1el-frozen-target-teacher-bank-v1",
        "status": "m1el-controls-exported-to-frozen-target",
        "source": {
            "repo": str(m1el_repo),
            "controlsPath": str(controls_path),
            "algorithm": "m1el-direct-collocation-plus-full-state-tvlqr",
            "dt": source["sourceDt"],
            "resampledDt": target_dt,
            "swingSeconds": source["sourceSwingSeconds"],
            "holdSeconds": source["sourceHoldSeconds"],
            "controlUnits": "pivot-acceleration",
        },
        "target": {
            "profile": profile,
            "dt": replay_dt,
            "links": links,
            "controlUnits": "cart-force",
            **force_stats,
        },
        "links": links,
        "profile": profile,
        "dt": replay_dt,
        "steps": int(len(controls)),
        "elapsedSeconds": time.time() - started,
        "trajectoryPath": str(write_trajectories),
        "observationShape": list(observations.shape),
        "actionShape": list(actions.shape),
        "qposShape": list(replay_qpos.shape),
        "qvelShape": list(replay_qvel.shape),
        "metrics": metrics,
        "gates": {
            "strictOneSecondRequired": True,
            "teacherSolvesDownStart": bool(metrics["solvedOneSecond"]),
            "teacherOnlyDoesNotPromote": True,
            "runPodMaxWallclockMinutes": 60,
        },
    }
    write_trajectories.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        write_trajectories,
        observations=observations,
        actions=actions,
        qpos=replay_qpos.astype(np.float64),
        qvel=replay_qvel.astype(np.float64),
        controls=controls.astype(np.float64),
        source_qpos=source_qpos.astype(np.float32),
        source_qvel=source_qvel.astype(np.float32),
        source_cart_acceleration=source["cartAcceleration"].astype(np.float32),
        metadataJson=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    write_summary.parent.mkdir(parents=True, exist_ok=True)
    write_summary.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m1el-repo", type=Path, default=Path("oss/inverted-pendulum"))
    parser.add_argument("--controls", type=Path, default=Path("oss/inverted-pendulum/repro/n6_controls.npz"))
    parser.add_argument("--profile", choices=["thread-matched"], default="thread-matched")
    parser.add_argument("--hold-seconds", type=float, default=6.0)
    parser.add_argument("--write-trajectories", type=Path, required=True)
    parser.add_argument("--write-summary", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = args.write_summary or args.write_trajectories.with_suffix(".json")
    result = export_m1el_teacher(
        args.m1el_repo,
        args.controls,
        args.profile,
        args.hold_seconds,
        args.write_trajectories,
        summary,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
