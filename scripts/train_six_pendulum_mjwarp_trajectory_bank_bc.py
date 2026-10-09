#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from six_pendulum_mjwarp_env import DEFAULT_ACTION_SCALE
from train_six_pendulum_mjwarp_local_ppo import (
    TinyRecurrentPolicy,
    evaluate,
    save_checkpoint,
    train_sequence_bc_epoch,
)


def load_bank(path: Path) -> tuple[torch.Tensor, torch.Tensor, dict]:
    with np.load(path) as data:
        observations = np.asarray(data["observations"], dtype=np.float32)
        actions = np.asarray(data["actions"], dtype=np.float32)
        metadata_json = str(data["metadataJson"]) if "metadataJson" in data else "{}"
    if observations.ndim != 3:
        raise ValueError(f"Expected observations with shape [steps, nworld, obs], got {observations.shape}")
    if actions.ndim == 2:
        actions = actions[..., None]
    if actions.ndim != 3:
        raise ValueError(f"Expected actions with shape [steps, nworld, action], got {actions.shape}")
    if observations.shape[:2] != actions.shape[:2]:
        raise ValueError(f"Observation/action rollout axes differ: {observations.shape} vs {actions.shape}")
    return torch.as_tensor(observations), torch.as_tensor(actions), json.loads(metadata_json)


def behavior_clone_trajectory_bank(
    mjcf_xml: str,
    links: int,
    nworld: int,
    eval_steps: int,
    epochs: int,
    sequence_length: int,
    seed: int,
    trajectory_bank: Path,
    write_checkpoint: Path | None,
    force_scale: float,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    started = time.time()

    observations, actions, source_metadata = load_bank(trajectory_bank)
    policy = TinyRecurrentPolicy()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=5e-4, weight_decay=1e-5)
    trajectories = [(observations, actions)]
    losses = []

    for epoch in range(epochs):
        loss = train_sequence_bc_epoch(policy, optimizer, trajectories, sequence_length)
        if epoch in {0, epochs // 4, epochs // 2, epochs - 1}:
            losses.append({"epoch": epoch + 1, "loss": float(loss)})

    hold_eval = evaluate(policy, mjcf_xml, links, nworld, eval_steps, "hold", force_scale)
    down_eval = evaluate(policy, mjcf_xml, links, nworld, eval_steps, "down", force_scale)
    result = {
        "schema": "six-pendulum-mjwarp-trajectory-bank-sequence-bc-v1",
        "status": "trajectory-bank-sequence-bc-smoke-passed",
        "algorithm": "tiny-gru-trajectory-bank-sequence-behavior-clone",
        "links": links,
        "nworld": nworld,
        "evalSteps": eval_steps,
        "evalSeconds": eval_steps * 0.0025,
        "epochs": epochs,
        "sequenceLength": max(8, min(sequence_length, int(observations.shape[0]))),
        "seed": seed,
        "elapsedSeconds": time.time() - started,
        "parameterCount": int(sum(parameter.numel() for parameter in policy.parameters())),
        "forceScale": force_scale,
        "trajectoryBank": str(trajectory_bank),
        "source": {
            "schema": source_metadata.get("schema"),
            "steps": source_metadata.get("steps"),
            "nworld": source_metadata.get("nworld"),
            "forceScale": source_metadata.get("forceScale"),
            "teacherSolvedOneSecond": bool(source_metadata.get("solvedOneSecond", False)),
            "teacherMaxHeldSeconds": source_metadata.get("maxHeldSeconds"),
            "teacherSolvedWorldCount": source_metadata.get("solvedWorldCount"),
        },
        "losses": losses,
        "evaluation": {
            "hold": hold_eval,
            "down": down_eval,
        },
        "gates": {
            "strictOneSecondRequired": True,
            "teacherBankSolvesDownStart": bool(source_metadata.get("solvedOneSecond", False)),
            "learnedPolicySolvesDownStart": bool(down_eval["solvedOneSecond"]),
            "promoteToNextLink": bool(down_eval["solvedOneSecond"]),
        },
        "nextRequiredWork": [
            "If held-out down-start is still below one second, add learner-state DAgger labels to the bank.",
            "Use a MinGRU/Puffer policy at GPU scale after the trajectory-bank loop is mechanically valid.",
            "Promote only after learned held-out exact down-start validation exceeds one second.",
        ],
    }
    result["checkpointPath"] = save_checkpoint(write_checkpoint, policy, {key: value for key, value in result.items() if key != "losses"})
    return result


def main():
    parser = argparse.ArgumentParser(description="Train sequence BC from exported MJWarp teacher trajectory banks.")
    parser.add_argument("--links", type=int, default=1)
    parser.add_argument("--nworld", type=int, default=4)
    parser.add_argument("--eval-steps", type=int, default=4000)
    parser.add_argument("--bc-epochs", type=int, default=500)
    parser.add_argument("--bc-sequence-length", type=int, default=160)
    parser.add_argument("--seed", type=int, default=426210)
    parser.add_argument("--force-scale", type=float, default=DEFAULT_ACTION_SCALE)
    parser.add_argument("--trajectory-bank", type=Path, required=True)
    parser.add_argument("--write-checkpoint", type=Path)
    parser.add_argument("--write-result", type=Path, required=True)
    args = parser.parse_args()

    mjcf_path = Path(f"app/ailab/six-pendulum-cartpole/mjcf/cartpole_{args.links}_link.xml")
    if not mjcf_path.exists():
        raise FileNotFoundError(f"Missing MJCF file: {mjcf_path}")

    result = behavior_clone_trajectory_bank(
        mjcf_path.read_text(),
        args.links,
        args.nworld,
        args.eval_steps,
        args.bc_epochs,
        args.bc_sequence_length,
        args.seed,
        args.trajectory_bank,
        args.write_checkpoint,
        args.force_scale,
    )
    args.write_result.parent.mkdir(parents=True, exist_ok=True)
    args.write_result.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
