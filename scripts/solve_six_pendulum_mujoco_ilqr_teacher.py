#!/usr/bin/env python3
import argparse
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import time

import numpy as np


OBS_DIM = 33
MAX_LINKS = 6


def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def load_generator():
    path = Path(__file__).with_name("generate-six-pendulum-mjcf.py")
    spec = importlib.util.spec_from_file_location("six_pendulum_mjcf_generator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load MJCF generator from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_model_xml(links: int, profile: str) -> str:
    generator = load_generator()
    with tempfile.TemporaryDirectory() as tmp:
        path = generator.write_model(links, Path(tmp), profile)
        return path.read_text()


def make_mujoco_model(xml: str):
    import mujoco

    return mujoco.MjModel.from_xml_string(xml)


def set_state(model, data, state: np.ndarray, force: float):
    data.qpos[:] = state[: model.nq]
    data.qvel[:] = state[model.nq : model.nq + model.nv]
    data.ctrl[0] = force


def step_state(model, state: np.ndarray, force: float) -> np.ndarray:
    import mujoco

    data = mujoco.MjData(model)
    set_state(model, data, state, force)
    mujoco.mj_step(model, data)
    out = np.concatenate([np.asarray(data.qpos).copy(), np.asarray(data.qvel).copy()])
    out[1 : model.nq] = wrap_angle(out[1 : model.nq])
    return out


def rollout(model, initial_state: np.ndarray, controls: np.ndarray) -> np.ndarray:
    states = np.zeros((len(controls) + 1, initial_state.size), dtype=np.float64)
    states[0] = initial_state
    current = initial_state
    for index, force in enumerate(controls):
        current = step_state(model, current, float(force))
        states[index + 1] = current
    return states


def encode_observations(states: np.ndarray, controls: np.ndarray, links: int, force_cap: float) -> np.ndarray:
    observations = np.zeros((len(controls), OBS_DIM), dtype=np.float32)
    qpos = states[:-1, : links + 1]
    qvel = states[:-1, links + 1 : 2 * (links + 1)]
    relative = wrap_angle(qpos[:, 1 : links + 1])
    absolute = wrap_angle(np.cumsum(relative, axis=1))
    abs_vel = np.cumsum(qvel[:, 1 : links + 1], axis=1)
    observations[:, 0] = qpos[:, 0]
    observations[:, 1] = qvel[:, 0] / 5.0
    observations[:, 2] = np.asarray(controls, dtype=np.float32) / force_cap
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


def trajectory_metrics(model, states: np.ndarray, force_cap: float) -> dict:
    links = model.nq - 1
    qpos = states[:, : model.nq]
    qvel = states[:, model.nq : model.nq + model.nv]
    relative = wrap_angle(qpos[:, 1 : links + 1])
    absolute = wrap_angle(np.cumsum(relative, axis=1))
    abs_vel = np.cumsum(qvel[:, 1 : links + 1], axis=1)
    max_upright_error = np.max(np.abs(absolute), axis=1)
    max_bend_error = np.max(np.abs(relative[:, 1:]), axis=1) if links > 1 else np.zeros(states.shape[0])
    mean_speed = np.mean(np.abs(abs_vel), axis=1)
    near_top = (max_upright_error < 0.16) & (max_bend_error < 0.14) & (mean_speed < 2.0)
    held_steps = 0
    max_held_steps = 0
    for value in near_top:
        held_steps = held_steps + 1 if value else 0
        max_held_steps = max(max_held_steps, held_steps)
    return {
        "maxHeldSeconds": float(max_held_steps * model.opt.timestep),
        "solvedOneSecond": bool(max_held_steps * model.opt.timestep >= 1.0),
        "finalUprightError": float(max_upright_error[-1]),
        "finalMeanSpeed": float(mean_speed[-1]),
        "maxAbsCart": float(np.max(np.abs(qpos[:, 0]))),
        "forceCap": float(force_cap),
    }


def running_cost(state: np.ndarray, force: float, model, force_cap: float, final: bool = False) -> float:
    qpos = state[: model.nq]
    qvel = state[model.nq : model.nq + model.nv]
    theta = wrap_angle(qpos[1])
    omega = qvel[1]
    x = qpos[0]
    v = qvel[0]
    weights = (60.0, 1.8, 0.45, 0.08, 0.0005)
    if final:
        weights = (4200.0, 120.0, 30.0, 12.0, 0.0)
    cart_over = max(abs(float(x)) - 1.85, 0.0)
    return (
        weights[0] * theta * theta
        + weights[1] * omega * omega
        + weights[2] * x * x
        + weights[3] * v * v
        + weights[4] * (force / force_cap) ** 2
        + 2500.0 * cart_over * cart_over
    )


def total_cost(model, states: np.ndarray, controls: np.ndarray, force_cap: float) -> float:
    cost = 0.0
    for state, force in zip(states[:-1], controls):
        cost += running_cost(state, float(force), model, force_cap)
    cost += running_cost(states[-1], 0.0, model, force_cap, final=True)
    terminal_cart = abs(float(states[-1, 0]))
    cost += 5000.0 * max(terminal_cart - 2.0, 0.0) ** 2
    return float(cost)


def energy_seed_force(state: np.ndarray, force_cap: float) -> float:
    x, theta, v, omega = float(state[0]), float(wrap_angle(state[1])), float(state[2]), float(state[3])
    energy = (1.0 / 6.0) * omega * omega + (9.8 / 2.0) * math.cos(theta)
    target_energy = 9.8 / 2.0
    pump_acc = 12.0 * (energy - target_energy) * omega * math.cos(theta)
    if abs(omega) < 0.2 and abs(theta) > math.pi - 0.5:
        pump_acc = 30.0
    acceleration = max(-30.0, min(30.0, pump_acc - 0.8 * v - 0.35 * x))
    catch = abs(theta) < 0.45 and abs(omega) < 3.8
    stabilizer = -(10.0 * x + 5.0 * v - 75.0 * theta - 18.0 * omega)
    swing = acceleration * 6.75
    return float(np.clip(stabilizer if catch else swing, -force_cap, force_cap))


def finite_difference_linearization(model, states: np.ndarray, controls: np.ndarray, eps: float) -> tuple[np.ndarray, np.ndarray]:
    horizon = len(controls)
    state_dim = states.shape[1]
    a_mats = np.zeros((horizon, state_dim, state_dim), dtype=np.float64)
    b_mats = np.zeros((horizon, state_dim, 1), dtype=np.float64)
    for t in range(horizon):
        state = states[t]
        force = float(controls[t])
        for i in range(state_dim):
            delta = np.zeros(state_dim)
            delta[i] = eps
            plus = step_state(model, state + delta, force)
            minus = step_state(model, state - delta, force)
            diff = plus - minus
            diff[1 : model.nq] = wrap_angle(diff[1 : model.nq])
            a_mats[t, :, i] = diff / (2.0 * eps)
        plus = step_state(model, state, force + eps)
        minus = step_state(model, state, force - eps)
        diff = plus - minus
        diff[1 : model.nq] = wrap_angle(diff[1 : model.nq])
        b_mats[t, :, 0] = diff / (2.0 * eps)
    return a_mats, b_mats


def quadratize(model, states: np.ndarray, controls: np.ndarray, force_cap: float):
    horizon = len(controls)
    state_dim = states.shape[1]
    q = np.zeros((horizon, state_dim), dtype=np.float64)
    q_terminal = np.zeros(state_dim, dtype=np.float64)
    q_mat = np.diag([0.45, 60.0, 0.08, 1.8])
    q_terminal_mat = np.diag([30.0, 4200.0, 12.0, 120.0])
    r_mat = np.asarray([[0.0005 / (force_cap * force_cap)]], dtype=np.float64)
    for t in range(horizon):
        q[t] = 2.0 * q_mat @ states[t]
        q[t, 1] = 2.0 * 60.0 * wrap_angle(states[t, 1])
    q_terminal[:] = 2.0 * q_terminal_mat @ states[-1]
    q_terminal[1] = 2.0 * 4200.0 * wrap_angle(states[-1, 1])
    return q, q_terminal, q_mat, q_terminal_mat, r_mat


def initial_controls(model, horizon: int, force_cap: float) -> np.ndarray:
    initial = np.zeros(model.nq + model.nv, dtype=np.float64)
    initial[1 : model.nq] = np.pi
    state = initial
    controls = np.zeros(horizon, dtype=np.float64)
    for t in range(horizon):
        controls[t] = energy_seed_force(state, force_cap)
        state = step_state(model, state, controls[t])
    return np.clip(controls, -force_cap, force_cap)


def solve_ilqr(model, horizon: int, force_cap: float, iterations: int, seed: int) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    np.random.seed(seed)
    initial = np.zeros(model.nq + model.nv, dtype=np.float64)
    initial[1 : model.nq] = np.pi
    controls = initial_controls(model, horizon, force_cap)
    states = rollout(model, initial, controls)
    cost = total_cost(model, states, controls, force_cap)
    history = [{"iteration": 0, "cost": cost, **trajectory_metrics(model, states, force_cap)}]
    reg = 5.0

    for iteration in range(1, iterations + 1):
        a_mats, b_mats = finite_difference_linearization(model, states, controls, eps=1e-4)
        q, q_terminal, q_mat, q_terminal_mat, r_mat = quadratize(model, states, controls, force_cap)
        value_x = q_terminal
        value_xx = q_terminal_mat.copy()
        gains = []
        feedforward = []
        failed = False
        for t in reversed(range(horizon)):
            a_t, b_t = a_mats[t], b_mats[t]
            q_x = q[t] + a_t.T @ value_x
            q_u = np.asarray([2.0 * r_mat[0, 0] * controls[t]]) + b_t.T @ value_x
            q_xx = q_mat + a_t.T @ value_xx @ a_t
            q_ux = b_t.T @ value_xx @ a_t
            q_uu = r_mat + b_t.T @ value_xx @ b_t + reg * np.eye(1)
            try:
                inv_q_uu = np.linalg.inv(q_uu)
            except np.linalg.LinAlgError:
                failed = True
                break
            k = -inv_q_uu @ q_u
            k_mat = -inv_q_uu @ q_ux
            feedforward.append(k)
            gains.append(k_mat)
            value_x = q_x + k_mat.T @ q_uu @ k + k_mat.T @ q_u + q_ux.T @ k
            value_xx = q_xx + k_mat.T @ q_uu @ k_mat + k_mat.T @ q_ux + q_ux.T @ k_mat
            value_xx = 0.5 * (value_xx + value_xx.T)
        if failed:
            reg *= 10.0
            continue
        feedforward.reverse()
        gains.reverse()

        accepted = False
        for alpha in (1.0, 0.5, 0.25, 0.1, 0.04, 0.015):
            candidate_controls = np.zeros_like(controls)
            candidate_states = np.zeros_like(states)
            candidate_states[0] = states[0]
            for t in range(horizon):
                dx = candidate_states[t] - states[t]
                dx[1 : model.nq] = wrap_angle(dx[1 : model.nq])
                du = alpha * feedforward[t][0] + (gains[t] @ dx)[0]
                candidate_controls[t] = np.clip(controls[t] + du, -force_cap, force_cap)
                candidate_states[t + 1] = step_state(model, candidate_states[t], candidate_controls[t])
            candidate_cost = total_cost(model, candidate_states, candidate_controls, force_cap)
            if candidate_cost < cost:
                controls, states, cost = candidate_controls, candidate_states, candidate_cost
                reg = max(reg * 0.5, 1e-4)
                accepted = True
                break
        if not accepted:
            reg *= 4.0
        history.append({"iteration": iteration, "cost": cost, "regularization": reg, **trajectory_metrics(model, states, force_cap)})
    return states, controls, history


def export_teacher(
    links: int,
    profile: str,
    horizon: int,
    iterations: int,
    seed: int,
    write_trajectories: Path,
    write_summary: Path,
) -> dict:
    if links != 1:
        raise ValueError("The iLQR teacher slice is intentionally locked to L1")
    started = time.time()
    xml = make_model_xml(links, profile)
    model = make_mujoco_model(xml)
    force_cap = float(model.actuator_ctrlrange[0, 1])
    states, controls, history = solve_ilqr(model, horizon, force_cap, iterations, seed)
    metrics = trajectory_metrics(model, states, force_cap)
    observations = encode_observations(states, controls, links, force_cap)
    actions = (controls / force_cap).reshape(-1, 1).astype(np.float32)
    qpos = states[:-1, : model.nq].astype(np.float32)
    qvel = states[:-1, model.nq : model.nq + model.nv].astype(np.float32)
    metadata = {
        "schema": "six-pendulum-mujoco-ilqr-teacher-bank-v1",
        "status": "ilqr-teacher-exported",
        "algorithm": "finite-difference-ilqr-on-exact-mujoco-mjcf",
        "links": links,
        "profile": profile,
        "horizon": horizon,
        "dt": float(model.opt.timestep),
        "seconds": float(horizon * model.opt.timestep),
        "iterations": iterations,
        "seed": seed,
        "forceScale": force_cap,
        "elapsedSeconds": time.time() - started,
        "trajectoryPath": str(write_trajectories),
        "observationShape": list(observations.shape),
        "actionShape": list(actions.shape),
        "qposShape": list(qpos.shape),
        "qvelShape": list(qvel.shape),
        "history": history,
        "metrics": metrics,
        "gates": {
            "strictOneSecondRequired": True,
            "teacherSolvesDownStart": bool(metrics["solvedOneSecond"]),
            "teacherOnlyDoesNotPromote": True,
            "ppoBlockedUntilTeacherExportExists": True,
            "runPodMaxWallclockMinutes": 60,
        },
    }
    write_trajectories.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        write_trajectories,
        observations=observations,
        actions=actions,
        qpos=qpos,
        qvel=qvel,
        controls=controls.astype(np.float32),
        metadataJson=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    write_summary.parent.mkdir(parents=True, exist_ok=True)
    write_summary.write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Export an L1 MuJoCo iLQR teacher trajectory bank.")
    parser.add_argument("--links", type=int, default=1)
    parser.add_argument("--profile", choices=["thread-matched", "current"], default="thread-matched")
    parser.add_argument("--horizon", type=int, default=800)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--seed", type=int, default=426210)
    parser.add_argument("--write-trajectories", type=Path, required=True)
    parser.add_argument("--write-summary", type=Path)
    args = parser.parse_args()

    write_summary = args.write_summary or args.write_trajectories.with_suffix(".json")
    result = export_teacher(
        args.links,
        args.profile,
        args.horizon,
        args.iterations,
        args.seed,
        args.write_trajectories,
        write_summary,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
