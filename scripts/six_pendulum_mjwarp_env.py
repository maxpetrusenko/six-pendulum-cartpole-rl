#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pufferlib


DEFAULT_ACTION_SCALE = 32.0
MAX_LINKS = 6
OBS_DIM = 3 + MAX_LINKS * 5
SCORE_MAX_UPRIGHT_ANGLE = 0.16
SCORE_MAX_CHAIN_BEND = 0.14


def score_batch(qpos, qvel, last_action, links: int, action_scale: float = DEFAULT_ACTION_SCALE) -> dict:
    safe_links = max(1, min(MAX_LINKS, int(links)))
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
    energy = np.mean(0.5 * (abs_vel * 0.28) ** 2 + np.cos(absolute), axis=1)
    energy_error = np.abs(energy - 1.0)
    near_top = (max_upright_error < SCORE_MAX_UPRIGHT_ANGLE) & (max_bend_error < SCORE_MAX_CHAIN_BEND)
    catch_basin = (max_upright_error < 0.55) & (max_bend_error < 0.34) & (mean_speed < 3.2)
    near_top_fast = (max_upright_error < 0.72) & (mean_speed > 1.2)
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
    low_height = 1.0 - mean_tip_height
    pump_reward = np.minimum(np.abs(qvel[:, 0]) * low_height, 3.0) * 0.08
    pump_reward += np.minimum(mean_speed * low_height, 3.0) * 0.04
    reward = mean_tip_height * 0.3 + dense_alignment * 0.1 + (strict_score / 100.0) ** 2 * 2.5
    reward += pump_reward
    reward += np.where(whip, 0.2, 0.0)
    reward += np.where(near_top_fast, 0.35, 0.0)
    reward += np.where(catch_basin, 1.0, 0.0)
    cart_fraction = np.abs(qpos[:, 0]) / 2.35
    reward -= cart_fraction**2 * 0.25
    reward -= np.maximum(cart_fraction - 0.8, 0.0) * 1.5
    reward -= (last_action / action_scale) ** 2 * 0.015
    potential = mean_tip_height - 0.02 * energy_error - 0.01 * np.abs(qpos[:, 0])

    obs = np.zeros((qpos.shape[0], OBS_DIM), dtype=np.float32)
    obs[:, 0] = qpos[:, 0]
    obs[:, 1] = qvel[:, 0] / 5.0
    obs[:, 2] = last_action / action_scale
    cursor = 3
    for index in range(MAX_LINKS):
        if index < safe_links:
            obs[:, cursor] = np.sin(absolute[:, index])
            obs[:, cursor + 1] = np.cos(absolute[:, index])
            obs[:, cursor + 2] = np.sin(relative[:, index])
            obs[:, cursor + 3] = np.cos(relative[:, index])
            obs[:, cursor + 4] = abs_vel[:, index] / 8.0
        cursor += 5

    return {
        "observation": obs,
        "reward": reward.astype(np.float32),
        "strictScore": strict_score.astype(np.float32),
        "meanTipHeight": mean_tip_height.astype(np.float32),
        "energyError": energy_error.astype(np.float32),
        "potential": potential.astype(np.float32),
        "maxUprightError": max_upright_error.astype(np.float32),
        "maxChainBend": max_bend_error.astype(np.float32),
        "catchBasin": catch_basin.astype(np.float32),
        "nearTopFast": near_top_fast.astype(np.float32),
        "whip": whip.astype(np.float32),
    }


class SixPendulumMJWarpPufferEnv(pufferlib.PufferEnv):
    def __init__(
        self,
        mjcf_xml: str,
        links: int = 1,
        nworld: int = 4,
        horizon: int = 400,
        pose: str = "down",
        force_scale: float = DEFAULT_ACTION_SCALE,
        seed: int | None = 426210,
        buf=None,
    ):
        import mujoco
        import mujoco_warp as mjw
        from six_pendulum_mjwarp_gpu_kernels import WarpScoreKernel

        self.links = max(1, min(MAX_LINKS, int(links)))
        self.nworld = max(1, int(nworld))
        self.horizon = max(1, int(horizon))
        self.pose = pose
        self.force_scale = float(force_scale)
        self.reset_seed = int(seed if seed is not None else 426210)
        self.rng = np.random.default_rng(seed)
        self.num_agents = self.nworld
        self.single_observation_space = pufferlib.gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32)
        self.single_action_space = pufferlib.gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.mujoco = mujoco
        self.mjw = mjw
        self.mjm = mujoco.MjModel.from_xml_string(mjcf_xml)
        self.m = mjw.put_model(self.mjm)
        self.d = mjw.make_data(self.mjm, nworld=self.nworld)
        self.score_kernel = WarpScoreKernel(
            self.nworld,
            self.links,
            self.force_scale,
            device=str(getattr(self.d.qpos, "device", "cpu")),
            terminal_boundary=2.35,
        )
        self.elapsed = np.zeros(self.nworld, dtype=np.int32)
        self.held_steps = np.zeros(self.nworld, dtype=np.int32)
        self.max_held_steps = np.zeros(self.nworld, dtype=np.int32)
        self.last_action = np.zeros(self.nworld, dtype=np.float32)
        self.prev_potential = np.zeros(self.nworld, dtype=np.float32)
        self.control_dt = float(self.mjm.opt.timestep)
        self.observations = np.zeros((self.nworld, OBS_DIM), dtype=np.float32)
        self.rewards = np.zeros(self.nworld, dtype=np.float32)
        self.terminals = np.zeros(self.nworld, dtype=bool)
        self.truncations = np.zeros(self.nworld, dtype=bool)
        self.masks = np.ones(self.nworld, dtype=bool)
        super().__init__(buf=buf)

    def _score_metrics(self):
        return self.score_kernel.score_from_warp_arrays(self.d.qpos, self.d.qvel, self.last_action)

    def _score_metrics_current_action(self):
        return self.score_kernel.score_from_current_last_action(self.d.qpos, self.d.qvel)

    def _reset_worlds(self, reset_all: bool = False):
        self.score_kernel.reset_worlds(self.d.qpos, self.d.qvel, self.d.ctrl, self.pose, self.reset_seed, reset_all=reset_all)

    def reset(self, seed=None):
        import warp as wp

        if seed is not None:
            self.reset_seed = int(seed)
            self.rng = np.random.default_rng(seed)
        self._reset_worlds(reset_all=True)
        self.elapsed.fill(0)
        self.held_steps.fill(0)
        self.max_held_steps.fill(0)
        self.last_action.fill(0.0)
        self.mjw.forward(self.m, self.d)
        metrics = self._score_metrics_current_action()
        self.prev_potential[:] = metrics["potential"]
        self.score_kernel.reset_rollout_state(self.prev_potential, self.elapsed, self.held_steps, self.max_held_steps)
        self.observations[:] = metrics["observation"]
        self.rewards.fill(0.0)
        self.terminals.fill(False)
        self.truncations.fill(False)
        return self.observations, self._infos(metrics)

    def step(self, actions):
        import warp as wp

        self.last_action = self.score_kernel.apply_actions(actions, self.d.ctrl)
        self.mjw.step(self.m, self.d)
        metrics = self._score_metrics_current_action()
        rollout = self.score_kernel.post_step(self.pose == "hold", self.horizon)
        self.observations[:] = metrics["observation"]
        self.rewards[:] = rollout["reward"]
        self.prev_potential[:] = rollout["prevPotential"]
        self.elapsed[:] = rollout["elapsed"]
        self.held_steps[:] = rollout["heldSteps"]
        self.max_held_steps[:] = rollout["maxHeldSteps"]
        self.terminals[:] = metrics["terminal"] > 0.5
        self.truncations[:] = rollout["truncation"] > 0.5
        done = self.terminals | self.truncations
        if np.any(done):
            self._reset_worlds(reset_all=False)
            self.mjw.forward(self.m, self.d)
            self._score_metrics_current_action()
            self.score_kernel.sync_reset_potential()
            self.prev_potential[:] = self.score_kernel.prev_potential_wp.numpy()
        return self.observations, self.rewards, self.terminals, self.truncations, self._infos(metrics)

    def _infos(self, metrics):
        return [
            {
                "strictScore": float(metrics["strictScore"][index]),
                "heldSeconds": float(self.held_steps[index] * self.control_dt),
                "maxHeldSeconds": float(self.max_held_steps[index] * self.control_dt),
                "whip": float(metrics["whip"][index]),
                "catchBasin": float(metrics["catchBasin"][index]),
                "nearTopFast": float(metrics["nearTopFast"][index]),
                "energyError": float(metrics["energyError"][index]),
            }
            for index in range(self.nworld)
        ]

    def close(self):
        return None


def run_driver_smoke(
    mjcf_xml: str,
    links: int = 1,
    nworld: int = 4,
    steps: int = 128,
    pose: str = "down",
    force_scale: float = DEFAULT_ACTION_SCALE,
) -> dict:
    started = time.time()
    env = SixPendulumMJWarpPufferEnv(mjcf_xml, links=links, nworld=nworld, horizon=steps + 1, pose=pose, force_scale=force_scale)
    obs, infos = env.reset()
    max_score = 0.0
    max_held = 0.0
    reward_sum = np.zeros(env.nworld, dtype=np.float32)
    for step_index in range(steps):
        if pose == "hold":
            force = np.clip(-18.0 * obs[:, 0] - 25.0 * obs[:, 1], -force_scale, force_scale)
        else:
            phase = step_index / max(1, steps - 1)
            force = np.full(env.nworld, np.sin(phase * np.pi * 4.0) * 18.0, dtype=np.float32)
        actions = np.asarray(force, dtype=np.float32).reshape(env.nworld, 1) / force_scale
        obs, rewards, terminals, truncations, infos = env.step(actions)
        reward_sum += rewards
        max_score = max(max_score, max(info["strictScore"] for info in infos))
        max_held = max(max_held, max(info["maxHeldSeconds"] for info in infos))
    env.close()
    return {
        "schema": "six-pendulum-mjwarp-puffer-env-driver-smoke-v1",
        "status": "puffer-env-driver-smoke-passed",
        "links": links,
        "nworld": nworld,
        "steps": steps,
        "pose": pose,
        "elapsedSeconds": time.time() - started,
        "observationShape": list(obs.shape),
        "actionSpace": {"shape": list(env.single_action_space.shape), "low": -1.0, "high": 1.0},
        "forceScale": force_scale,
        "scoreBackend": "warp-score-kernel",
        "rolloutBackend": "warp-post-step-kernel",
        "resetBackend": "warp-reset-kernel",
        "rewardMean": float(np.mean(reward_sum)),
        "maxStrictScore": float(max_score),
        "maxHeldSeconds": float(max_held),
        "solvedOneSecond": bool(max_held >= 1.0),
        "firstInfo": infos[0] if infos else {},
        "notes": [
            "This is an environment-driver smoke, not training.",
            f"Actions are normalized to [-1, 1] and scaled to cart force +/-{force_scale:g}.",
            "Strict score and one-second gate are preserved for lower-link promotion.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="Run the six-pendulum MJWarp PufferEnv driver smoke.")
    parser.add_argument("--links", type=int, default=1)
    parser.add_argument("--nworld", type=int, default=4)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--pose", choices=["down", "hold", "mixed", "down-heavy"], default="down")
    parser.add_argument("--force-scale", type=float, default=DEFAULT_ACTION_SCALE)
    parser.add_argument(
        "--write-result",
        type=Path,
        default=Path("local-artifacts/outputs/training-checkpoints/puffer-mjwarp-env-driver.json"),
    )
    args = parser.parse_args()

    mjcf_path = Path(f"app/ailab/six-pendulum-cartpole/mjcf/cartpole_{args.links}_link.xml")
    if not mjcf_path.exists():
        raise FileNotFoundError(f"Missing MJCF file: {mjcf_path}")
    result = run_driver_smoke(mjcf_path.read_text(), args.links, args.nworld, args.steps, args.pose, args.force_scale)
    args.write_result.parent.mkdir(parents=True, exist_ok=True)
    args.write_result.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
