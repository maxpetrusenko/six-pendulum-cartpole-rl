#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path

import numpy as np

from six_pendulum_mjwarp_env import DEFAULT_ACTION_SCALE, OBS_DIM, SixPendulumMJWarpPufferEnv
from train_six_pendulum_mjwarp_local_ppo import energy_teacher_action_from_state


def infos_to_array(infos, key: str) -> np.ndarray:
    return np.asarray([info.get(key, 0.0) for info in infos], dtype=np.float32)


def export_energy_teacher_trajectories(
    mjcf_xml: str,
    links: int,
    nworld: int,
    steps: int,
    seed: int,
    force_scale: float,
    write_trajectories: Path,
    write_summary: Path,
) -> dict:
    if links != 1:
        raise ValueError("Energy teacher trajectory export is currently defined for one link")

    started = time.time()
    env = SixPendulumMJWarpPufferEnv(
        mjcf_xml,
        links=links,
        nworld=nworld,
        horizon=steps + 1,
        pose="down",
        force_scale=force_scale,
        seed=seed,
    )
    obs, _ = env.reset(seed)

    observations = np.zeros((steps, nworld, OBS_DIM), dtype=np.float32)
    actions = np.zeros((steps, nworld, 1), dtype=np.float32)
    qpos = np.zeros((steps, nworld, env.d.qpos.shape[1]), dtype=np.float32)
    qvel = np.zeros((steps, nworld, env.d.qvel.shape[1]), dtype=np.float32)
    rewards = np.zeros((steps, nworld), dtype=np.float32)
    strict_scores = np.zeros((steps, nworld), dtype=np.float32)
    held_seconds = np.zeros((steps, nworld), dtype=np.float32)
    max_held_seconds = np.zeros((steps, nworld), dtype=np.float32)
    catch_basin = np.zeros((steps, nworld), dtype=np.float32)
    near_top_fast = np.zeros((steps, nworld), dtype=np.float32)
    terminals = np.zeros((steps, nworld), dtype=np.bool_)
    truncations = np.zeros((steps, nworld), dtype=np.bool_)

    for step_index in range(steps):
        teacher_action = energy_teacher_action_from_state(env.d.qpos.numpy(), env.d.qvel.numpy(), force_scale)
        observations[step_index] = obs
        actions[step_index] = teacher_action
        qpos[step_index] = env.d.qpos.numpy()
        qvel[step_index] = env.d.qvel.numpy()

        obs, reward, terminal, truncation, infos = env.step(teacher_action)
        rewards[step_index] = reward
        strict_scores[step_index] = infos_to_array(infos, "strictScore")
        held_seconds[step_index] = infos_to_array(infos, "heldSeconds")
        max_held_seconds[step_index] = infos_to_array(infos, "maxHeldSeconds")
        catch_basin[step_index] = infos_to_array(infos, "catchBasin")
        near_top_fast[step_index] = infos_to_array(infos, "nearTopFast")
        terminals[step_index] = terminal
        truncations[step_index] = truncation

    env.close()

    best_by_world = np.max(max_held_seconds, axis=0)
    solved_worlds = best_by_world >= 1.0
    metadata = {
        "schema": "six-pendulum-mjwarp-energy-teacher-trajectory-bank-v1",
        "status": "energy-teacher-trajectories-exported",
        "algorithm": "energy-pump-plus-stabilizer-teacher",
        "links": links,
        "nworld": nworld,
        "steps": steps,
        "seconds": steps * 0.0025,
        "seed": seed,
        "forceScale": force_scale,
        "elapsedSeconds": time.time() - started,
        "trajectoryPath": str(write_trajectories),
        "observationShape": list(observations.shape),
        "actionShape": list(actions.shape),
        "qposShape": list(qpos.shape),
        "qvelShape": list(qvel.shape),
        "rewardMean": float(np.mean(np.sum(rewards, axis=0))),
        "maxStrictScore": float(np.max(strict_scores)),
        "maxHeldSeconds": float(np.max(best_by_world)),
        "solvedOneSecond": bool(np.any(solved_worlds)),
        "solvedWorldCount": int(np.sum(solved_worlds)),
        "catchEvents": int(np.sum(catch_basin > 0.0)),
        "nearTopFastEvents": int(np.sum(near_top_fast > 0.0)),
        "doneEvents": int(np.sum(terminals | truncations)),
        "arrays": {
            "observations": "pre-action policy observations",
            "actions": "normalized teacher cart force labels in [-1, 1]",
            "qpos": "pre-action MuJoCo positions",
            "qvel": "pre-action MuJoCo velocities",
            "rewards": "post-action shaped rewards",
            "strictScores": "post-action strict scores",
            "heldSeconds": "post-action current held duration",
            "maxHeldSeconds": "post-action max held duration",
            "terminals": "post-action terminal flags before automatic env reset",
            "truncations": "post-action truncation flags before automatic env reset",
        },
        "gates": {
            "strictOneSecondRequired": True,
            "teacherSolvesDownStart": bool(np.any(solved_worlds)),
            "teacherOnlyDoesNotPromote": True,
            "linkTwoLockedUntilLearnedPolicyPasses": True,
        },
        "nextRequiredWork": [
            "Train sequence BC from this bank, preserving recurrent hidden state over contiguous windows.",
            "Use learner-state DAgger to add labels from states visited by the learned policy.",
            "Promote only after a learned held-out exact down-start policy holds for at least one second.",
        ],
    }

    write_trajectories.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        write_trajectories,
        observations=observations,
        actions=actions,
        qpos=qpos,
        qvel=qvel,
        rewards=rewards,
        strictScores=strict_scores,
        heldSeconds=held_seconds,
        maxHeldSeconds=max_held_seconds,
        catchBasin=catch_basin,
        nearTopFast=near_top_fast,
        terminals=terminals,
        truncations=truncations,
        metadataJson=np.asarray(json.dumps(metadata, sort_keys=True)),
    )

    write_summary.parent.mkdir(parents=True, exist_ok=True)
    write_summary.write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main():
    parser = argparse.ArgumentParser(description="Export one-link MJWarp energy-teacher trajectories for BC/DAgger.")
    parser.add_argument("--links", type=int, default=1)
    parser.add_argument("--nworld", type=int, default=8)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=426210)
    parser.add_argument("--force-scale", type=float, default=DEFAULT_ACTION_SCALE)
    parser.add_argument("--write-trajectories", type=Path, required=True)
    parser.add_argument("--write-summary", type=Path)
    args = parser.parse_args()

    mjcf_path = Path(f"app/ailab/six-pendulum-cartpole/mjcf/cartpole_{args.links}_link.xml")
    if not mjcf_path.exists():
        raise FileNotFoundError(f"Missing MJCF file: {mjcf_path}")

    write_summary = args.write_summary or args.write_trajectories.with_suffix(".json")
    result = export_energy_teacher_trajectories(
        mjcf_path.read_text(),
        args.links,
        args.nworld,
        args.steps,
        args.seed,
        args.force_scale,
        args.write_trajectories,
        write_summary,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
