#!/usr/bin/env python3
"""Validate six-pendulum artifacts against the frozen promotion target."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import tempfile
from typing import Any

import numpy as np


SCHEMA = "six-pendulum-frozen-target-validator-v1"

TARGET = {
    "profile": "thread-matched",
    "exactDownStart": True,
    "dt": 0.002,
    "strictHoldSeconds": 1.0,
    "maxUprightAngleRad": 0.16,
    "maxChainBendRad": 0.14,
    "cartTravelLimit": None,
    "maxAbsAngularVelocity": 4.5,
    "downStartToleranceRad": 1e-6,
}

PROMOTION_KINDS = {"learned-policy-rollout"}
TEACHER_KINDS = {"teacher-trajectory", "optimizer-trajectory"}


def normalize_candidate_kind(kind: str | None) -> str | None:
    if kind == "learned-policy":
        return "learned-policy-rollout"
    return kind


def wrap_pi(values: np.ndarray) -> np.ndarray:
    return (values + math.pi) % (2.0 * math.pi) - math.pi


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def save_json(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


def as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def find_numeric_key(value: Any, wanted_keys: set[str]) -> float | None:
    if isinstance(value, dict):
        for key, nested in value.items():
            if key in wanted_keys:
                found = as_float(nested)
                if found is not None:
                    return found
        for nested in value.values():
            found = find_numeric_key(nested, wanted_keys)
            if found is not None:
                return found
    elif isinstance(value, list):
        for nested in value:
            found = find_numeric_key(nested, wanted_keys)
            if found is not None:
                return found
    return None


def infer_candidate_kind(path: Path, artifact_type: str, payload: dict[str, Any] | None) -> str:
    if artifact_type == "trajectory-bank":
        if payload and "teacher" in str(payload.get("schema", "")).lower():
            return "teacher-trajectory"
        return "unknown-trajectory"
    if payload:
        model_type = str(payload.get("modelType", "")).lower()
        schema = str(payload.get("schema", "")).lower()
        if "pezzza" in model_type:
            return "browser-policy"
        if "teacher" in schema or "ilqr" in schema:
            return "teacher-summary"
        if "ppo" in schema or "policy" in schema:
            return "training-summary"
    if path.suffix == ".json":
        return "summary-json"
    return "unknown"


def longest_true_run_seconds(mask: np.ndarray, dt: float) -> float:
    best = 0
    current = 0
    for item in mask.astype(bool):
        if item:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return float(best * dt)


def metadata_from_npz(data: np.lib.npyio.NpzFile) -> dict[str, Any]:
    if "metadataJson" not in data.files:
        return {}
    raw = data["metadataJson"]
    try:
        return json.loads(str(raw.item() if hasattr(raw, "item") else raw))
    except json.JSONDecodeError:
        return {}


def load_mjcf_generator():
    path = Path(__file__).with_name("generate-six-pendulum-mjcf.py")
    spec = importlib.util.spec_from_file_location("six_pendulum_mjcf_generator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load MJCF generator from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_mujoco_model(links: int, profile: str):
    import mujoco

    generator = load_mjcf_generator()
    with tempfile.TemporaryDirectory() as tmp:
        path = generator.write_model(links, Path(tmp), profile)
        return mujoco.MjModel.from_xml_path(str(path))


def replay_controls(links: int, profile: str, initial_qpos: np.ndarray, initial_qvel: np.ndarray, controls: np.ndarray):
    import mujoco

    model = make_mujoco_model(links, profile)
    if model.nq != links + 1 or model.nv != links + 1:
        raise ValueError(f"compiled model dimensions do not match links={links}")
    data = mujoco.MjData(model)
    data.qpos[:] = initial_qpos
    data.qvel[:] = initial_qvel
    qpos = np.zeros((len(controls) + 1, model.nq), dtype=np.float64)
    qvel = np.zeros((len(controls) + 1, model.nv), dtype=np.float64)
    qpos[0] = np.asarray(data.qpos)
    qvel[0] = np.asarray(data.qvel)
    for index, force in enumerate(controls):
        data.ctrl[0] = float(force)
        mujoco.mj_step(model, data)
        qpos[index + 1] = np.asarray(data.qpos)
        qvel[index + 1] = np.asarray(data.qvel)
    qpos[:, 1:] = wrap_pi(qpos[:, 1:])
    return qpos[:-1], qvel[:-1], float(model.opt.timestep), float(model.actuator_ctrlrange[0, 1])


def strict_metrics(qpos: np.ndarray, qvel: np.ndarray, links: int, dt: float) -> dict[str, Any]:
    rel_angles = wrap_pi(qpos[:, 1:])
    abs_angles = wrap_pi(np.cumsum(rel_angles, axis=1))
    rel_vel = qvel[:, 1:]
    abs_vel = np.cumsum(rel_vel, axis=1)

    down_start_error = float(np.max(np.abs(wrap_pi(abs_angles[0] - math.pi)))) if links else None
    exact_down_start = bool(down_start_error is not None and down_start_error <= TARGET["downStartToleranceRad"])

    max_abs_upright_angle = np.max(np.abs(abs_angles), axis=1) if links else np.zeros(qpos.shape[0])
    if links > 1:
        max_abs_chain_bend = np.max(np.abs(rel_angles[:, 1:]), axis=1)
    else:
        max_abs_chain_bend = np.zeros(qpos.shape[0])
    max_abs_angular_velocity = np.max(np.abs(abs_vel), axis=1) if links else np.zeros(qpos.shape[0])
    cart_abs = np.abs(qpos[:, 0])

    strict_mask = (
        (max_abs_upright_angle <= TARGET["maxUprightAngleRad"])
        & (max_abs_chain_bend <= TARGET["maxChainBendRad"])
        & (max_abs_angular_velocity <= TARGET["maxAbsAngularVelocity"])
    )
    max_held_seconds = longest_true_run_seconds(strict_mask, dt)
    return {
        "dt": dt,
        "steps": int(qpos.shape[0]),
        "durationSeconds": float(qpos.shape[0] * dt),
        "maxHeldSeconds": max_held_seconds,
        "solvedOneSecond": bool(max_held_seconds >= TARGET["strictHoldSeconds"]),
        "downStartErrorRad": down_start_error,
        "exactDownStart": exact_down_start,
        "maxCartAbs": float(np.max(cart_abs)),
        "maxAbsUprightAngleRad": float(np.max(max_abs_upright_angle)),
        "maxAbsChainBendRad": float(np.max(max_abs_chain_bend)),
        "maxAbsAngularVelocity": float(np.max(max_abs_angular_velocity)),
        "strictSamples": int(np.sum(strict_mask)),
    }


def add_gate_failures(metrics: dict[str, Any], failures: list[str]) -> None:
    if not metrics.get("exactDownStart"):
        failures.append("not-exact-down-start")
    if not metrics.get("solvedOneSecond"):
        failures.append("strict-hold-under-one-second")


def validate_trajectory_bank(
    path: Path,
    candidate_kind: str | None,
) -> dict[str, Any]:
    failures: list[str] = []
    warnings: list[str] = []

    with np.load(path, allow_pickle=False) as data:
        files = set(data.files)
        missing = {"qpos", "qvel"} - files
        metadata = metadata_from_npz(data)
        if missing:
            failures.append(f"trajectory-bank-missing-{','.join(sorted(missing))}")
            links = int(metadata.get("links", 0) or 0)
            metrics: dict[str, Any] = {}
        else:
            qpos = np.asarray(data["qpos"], dtype=np.float64)
            qvel = np.asarray(data["qvel"], dtype=np.float64)
            if qpos.ndim != 2 or qvel.ndim != 2 or qpos.shape[0] != qvel.shape[0]:
                failures.append("invalid-qpos-qvel-shapes")
                links = 0
                metrics = {}
            else:
                links = int(qpos.shape[1] - 1)
                dt = float(metadata.get("dt", TARGET["dt"]) or TARGET["dt"])
                profile = str(metadata.get("profile", "unknown"))
                if profile != TARGET["profile"]:
                    failures.append(f"profile-not-thread-matched:{profile}")
                if abs(dt - TARGET["dt"]) > 1e-9:
                    failures.append(f"dt-not-frozen:{dt}")
                if links < 1:
                    failures.append("no-pendulum-links")
                if "controls" in files:
                    try:
                        controls = np.asarray(data["controls"], dtype=np.float64).reshape(-1)
                        replay_qpos, replay_qvel, replay_dt, force_cap = replay_controls(
                            links, profile, qpos[0], qvel[0], controls
                        )
                        replay_metrics = strict_metrics(replay_qpos, replay_qvel, links, replay_dt)
                        logged_metrics = strict_metrics(qpos, qvel, links, dt)
                        comparable = min(len(replay_qpos), len(qpos))
                        qpos_delta = float(np.max(np.abs(wrap_pi(replay_qpos[:comparable] - qpos[:comparable]))))
                        qvel_delta = float(np.max(np.abs(replay_qvel[:comparable] - qvel[:comparable])))
                        if qpos_delta > 1e-4 or qvel_delta > 1e-4:
                            failures.append("logged-states-differ-from-mujoco-replay")
                        metrics = {
                            **replay_metrics,
                            "scoredSource": "mujoco-control-replay",
                            "forceCap": force_cap,
                            "loggedMaxHeldSeconds": logged_metrics["maxHeldSeconds"],
                            "replayMaxQposDelta": qpos_delta,
                            "replayMaxQvelDelta": qvel_delta,
                        }
                    except Exception as exc:  # noqa: BLE001
                        failures.append(f"mujoco-replay-failed:{type(exc).__name__}")
                        metrics = strict_metrics(qpos, qvel, links, dt)
                        metrics["scoredSource"] = "logged-states-replay-failed"
                        metrics["replayError"] = str(exc)
                else:
                    metrics = strict_metrics(qpos, qvel, links, dt)
                    metrics["scoredSource"] = "logged-states-no-controls"
                    warnings.append("trajectory bank has no controls, so validator could not re-simulate physics")
                add_gate_failures(metrics, failures)

    kind = normalize_candidate_kind(candidate_kind) or infer_candidate_kind(
        path, "trajectory-bank", metadata if "metadata" in locals() else None
    )
    promotion_eligible = (
        not failures
        and kind in PROMOTION_KINDS
        and bool(metrics.get("solvedOneSecond"))
        and bool(metrics.get("exactDownStart"))
    )
    if not promotion_eligible:
        if kind in TEACHER_KINDS:
            promotion_reason = "teacher-trajectory-does-not-promote-learned-policy"
        elif kind not in PROMOTION_KINDS:
            promotion_reason = f"candidate-kind-not-promotional:{kind}"
        elif failures:
            promotion_reason = "frozen-target-failed"
        else:
            promotion_reason = "not-eligible"
    else:
        promotion_reason = "learned-policy-rollout-solved-frozen-target"

    if kind in TEACHER_KINDS:
        warnings.append("teacher evidence can feed BC/DAgger but cannot promote the learned policy")

    return {
        "schema": SCHEMA,
        "status": "pass" if not failures else "fail",
        "artifact": str(path),
        "artifactType": "trajectory-bank",
        "candidateKind": kind,
        "links": links,
        "target": TARGET,
        "metrics": metrics,
        "promotion": {"eligible": promotion_eligible, "reason": promotion_reason},
        "failures": failures,
        "warnings": warnings,
    }


def validate_summary_json(path: Path, candidate_kind: str | None) -> dict[str, Any]:
    payload = load_json(path)
    kind = normalize_candidate_kind(candidate_kind) or infer_candidate_kind(path, "summary-json", payload)
    max_held = find_numeric_key(payload, {"maxHeldSeconds", "maxHeld", "max_hold_seconds", "maxHoldSeconds"})
    solved = bool(max_held is not None and max_held >= TARGET["strictHoldSeconds"])
    failures = ["summary-only-not-replayable"]
    if kind in {"browser-policy", "summary-json"} and "modelType" in payload:
        failures.append("browser-policy-needs-target-replay")

    return {
        "schema": SCHEMA,
        "status": "fail",
        "artifact": str(path),
        "artifactType": "summary-json",
        "candidateKind": kind,
        "links": payload.get("links"),
        "target": TARGET,
        "metrics": {
            "reportedMaxHeldSeconds": max_held,
            "reportedSolvedOneSecond": solved,
        },
        "promotion": {"eligible": False, "reason": "summary-json-cannot-promote-without-target-replay"},
        "failures": failures,
        "warnings": ["use replayable qpos/qvel or deterministic target rollout logs for promotion"],
    }


def validate_artifact(path: Path, candidate_kind: str | None = None) -> dict[str, Any]:
    candidate_kind = normalize_candidate_kind(candidate_kind)
    if not path.exists():
        return {
            "schema": SCHEMA,
            "status": "fail",
            "artifact": str(path),
            "artifactType": "missing",
            "candidateKind": candidate_kind or "unknown",
            "links": None,
            "target": TARGET,
            "metrics": {},
            "promotion": {"eligible": False, "reason": "artifact-missing"},
            "failures": ["artifact-missing"],
            "warnings": [],
        }
    if path.suffix == ".npz":
        return validate_trajectory_bank(path, candidate_kind)
    if path.suffix == ".json":
        return validate_summary_json(path, candidate_kind)
    return {
        "schema": SCHEMA,
        "status": "fail",
        "artifact": str(path),
        "artifactType": "unsupported",
        "candidateKind": candidate_kind or "unknown",
        "links": None,
        "target": TARGET,
        "metrics": {},
        "promotion": {"eligible": False, "reason": "unsupported-artifact-type"},
        "failures": ["unsupported-artifact-type"],
        "warnings": [],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument(
        "--candidate-kind",
        choices=[
            "learned-policy-rollout",
            "learned-policy",
            "teacher-trajectory",
            "optimizer-trajectory",
            "training-summary",
            "browser-policy",
            "summary-json",
            "unknown",
        ],
    )
    parser.add_argument("--write-result", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = validate_artifact(args.artifact, args.candidate_kind)
    if args.write_result:
        save_json(args.write_result, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
