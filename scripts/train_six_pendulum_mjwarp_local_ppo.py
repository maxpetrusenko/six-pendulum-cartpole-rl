#!/usr/bin/env python3
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from six_pendulum_mjwarp_env import DEFAULT_ACTION_SCALE, OBS_DIM, SixPendulumMJWarpPufferEnv


class TinyRecurrentPolicy(nn.Module):
    def __init__(self, obs_dim: int = OBS_DIM, hidden_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(obs_dim, hidden_dim), nn.Tanh())
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.actor = nn.Linear(hidden_dim, 1)
        self.critic = nn.Linear(hidden_dim, 1)
        self.log_std = nn.Parameter(torch.tensor([-0.35]))

    def forward(self, obs, hidden):
        encoded = self.encoder(obs)
        next_hidden = self.gru(encoded, hidden)
        mean = torch.tanh(self.actor(next_hidden))
        value = self.critic(next_hidden).squeeze(-1)
        std = self.log_std.exp().expand_as(mean)
        return mean, std, value, next_hidden


def save_checkpoint(path: Path | None, policy, metadata: dict) -> str | None:
    if path is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"stateDict": policy.state_dict(), "metadata": metadata}, path)
    return str(path)


def load_checkpoint(path: Path | None, policy) -> dict | None:
    if path is None:
        return None
    payload = torch.load(path, map_location="cpu")
    policy.load_state_dict(payload["stateDict"])
    return payload.get("metadata", {})


def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def evaluate(policy, mjcf_xml: str, links: int, nworld: int, steps: int, pose: str, force_scale: float = DEFAULT_ACTION_SCALE) -> dict:
    env = SixPendulumMJWarpPufferEnv(mjcf_xml, links=links, nworld=nworld, horizon=steps + 1, pose=pose, force_scale=force_scale)
    obs, infos = env.reset()
    hidden = torch.zeros(nworld, policy.gru.hidden_size)
    reward_sum = np.zeros(nworld, dtype=np.float32)
    max_score = 0.0
    max_held = 0.0
    with torch.no_grad():
        for _ in range(steps):
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
            mean, _, _, hidden = policy(obs_tensor, hidden)
            obs, rewards, terminals, truncations, infos = env.step(mean.numpy())
            done = torch.as_tensor(terminals | truncations, dtype=torch.bool)
            if bool(done.any()):
                hidden[done] = 0.0
            reward_sum += rewards
            max_score = max(max_score, max(info["strictScore"] for info in infos))
            max_held = max(max_held, max(info["maxHeldSeconds"] for info in infos))
    env.close()
    return {
        "pose": pose,
        "steps": steps,
        "forceScale": force_scale,
        "rewardMean": float(np.mean(reward_sum)),
        "maxStrictScore": float(max_score),
        "maxHeldSeconds": float(max_held),
        "solvedOneSecond": bool(max_held >= 1.0),
    }


def energy_teacher_action_from_state(qpos, qvel, force_scale: float) -> np.ndarray:
    theta = wrap_angle(qpos[:, 1])
    omega = qvel[:, 1]
    x = qpos[:, 0]
    v = qvel[:, 0]
    energy = (1.0 / 6.0) * omega**2 + (9.8 / 2.0) * np.cos(theta)
    target_energy = 9.8 / 2.0
    pump_acc = 12.0 * (energy - target_energy) * omega * np.cos(theta)
    near_bottom = (np.abs(omega) < 0.2) & (np.abs(theta) > np.pi - 0.5)
    pump_acc = np.where(near_bottom, 30.0, pump_acc)
    acceleration = np.clip(pump_acc - 0.6 * v - 0.05 * x, -30.0, 30.0)
    catch = (np.abs(theta) < 0.36) & (np.abs(omega) < 3.0)
    stabilizer_force = -(8.0 * x + 4.0 * v - 60.0 * theta - 16.0 * omega)
    swing_force = acceleration * 6.75
    force = np.where(catch, stabilizer_force, swing_force)
    return np.clip(force, -force_scale, force_scale).reshape(len(qpos), 1).astype(np.float32) / force_scale


def evaluate_energy_teacher(mjcf_xml: str, links: int, nworld: int, steps: int, seed: int, force_scale: float) -> dict:
    if links != 1:
        raise ValueError("Energy teacher is only defined for one link")
    started = time.time()
    env = SixPendulumMJWarpPufferEnv(mjcf_xml, links=links, nworld=nworld, horizon=steps + 1, pose="down", force_scale=force_scale, seed=seed)
    obs, infos = env.reset(seed)
    reward_sum = np.zeros(nworld, dtype=np.float32)
    max_score = 0.0
    max_held = 0.0
    catch_events = 0
    near_top_events = 0
    for _ in range(steps):
        action = energy_teacher_action_from_state(env.d.qpos.numpy(), env.d.qvel.numpy(), force_scale)
        obs, rewards, terminals, truncations, infos = env.step(action)
        reward_sum += rewards
        max_score = max(max_score, max(info["strictScore"] for info in infos))
        max_held = max(max_held, max(info["maxHeldSeconds"] for info in infos))
        catch_events += sum(1 for info in infos if info.get("catchBasin", 0.0) > 0.0)
        near_top_events += sum(1 for info in infos if info.get("nearTopFast", 0.0) > 0.0)
    env.close()
    return {
        "schema": "six-pendulum-mjwarp-energy-teacher-v1",
        "status": "teacher-eval-passed",
        "algorithm": "energy-pump-plus-stabilizer-teacher",
        "links": links,
        "nworld": nworld,
        "steps": steps,
        "evalSeconds": steps * 0.0025,
        "seed": seed,
        "forceScale": force_scale,
        "elapsedSeconds": time.time() - started,
        "rewardMean": float(np.mean(reward_sum)),
        "maxStrictScore": float(max_score),
        "maxHeldSeconds": float(max_held),
        "solvedOneSecond": bool(max_held >= 1.0),
        "catchEvents": int(catch_events),
        "nearTopFastEvents": int(near_top_events),
        "teacher": {
            "energy": "E = (1/6) * omega^2 + (g/2) * cos(theta), target E* = g/2",
            "pump": "a = kE * (E - E*) * omega * cos(theta) - kv*v - kx*x",
            "gains": {"kE": 12.0, "kv": 0.6, "kx": 0.05, "aMax": 30.0, "forcePerAcceleration": 6.75},
            "catch": {"angle": 0.36, "rate": 3.0, "stabilizer": "force = -(8*x + 4*v - 60*theta - 16*omega)"},
        },
        "gates": {
            "strictOneSecondRequired": True,
            "teacherSolvesDownStart": bool(max_held >= 1.0),
            "promoteToNextLink": False,
        },
        "nextRequiredWork": [
            "Behavior-clone this teacher into a neural policy, then fine-tune with RL.",
            "Keep link two locked until a learned held-out down-start policy passes one second.",
        ],
    }


def train(
    mjcf_xml: str,
    links: int,
    nworld: int,
    rollout_steps: int,
    eval_steps: int,
    updates: int,
    pose: str,
    seed: int,
    warmstart_checkpoint: Path | None = None,
    write_checkpoint: Path | None = None,
    force_scale: float = DEFAULT_ACTION_SCALE,
    energy_teacher_anchor_weight: float = 0.0,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    started = time.time()
    env = SixPendulumMJWarpPufferEnv(mjcf_xml, links=links, nworld=nworld, horizon=rollout_steps + 1, pose=pose, force_scale=force_scale)
    policy = TinyRecurrentPolicy()
    warmstart_metadata = load_checkpoint(warmstart_checkpoint, policy)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=3e-4, weight_decay=1e-5)
    obs, infos = env.reset()
    hidden = torch.zeros(nworld, policy.gru.hidden_size)
    history = []
    gamma = 0.985

    for update_index in range(updates):
        logprobs = []
        values = []
        rewards = []
        entropies = []
        action_means = []
        teacher_actions = []
        max_score = 0.0
        max_held = 0.0
        reward_sum = np.zeros(nworld, dtype=np.float32)
        hidden = hidden.detach()

        for _ in range(rollout_steps):
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
            mean, std, value, next_hidden = policy(obs_tensor, hidden)
            dist = Normal(mean, std)
            action = dist.sample()
            logprob = dist.log_prob(action).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
            if energy_teacher_anchor_weight > 0 and links == 1:
                teacher_action = energy_teacher_action_from_state(env.d.qpos.numpy(), env.d.qvel.numpy(), force_scale)
                teacher_actions.append(torch.as_tensor(teacher_action[:, 0], dtype=torch.float32))
                action_means.append(mean[:, 0])
            obs, reward, terminals, truncations, infos = env.step(action.detach().numpy())
            done = torch.as_tensor(terminals | truncations, dtype=torch.bool)
            hidden_for_next_step = next_hidden.detach().clone()
            if bool(done.any()):
                hidden_for_next_step[done] = 0.0
            hidden = hidden_for_next_step
            logprobs.append(logprob)
            values.append(value)
            rewards.append(torch.as_tensor(reward, dtype=torch.float32))
            entropies.append(entropy)
            reward_sum += reward
            max_score = max(max_score, max(info["strictScore"] for info in infos))
            max_held = max(max_held, max(info["maxHeldSeconds"] for info in infos))

        returns = []
        running = torch.zeros(nworld)
        for reward in reversed(rewards):
            running = reward + gamma * running
            returns.append(running)
        returns.reverse()
        returns_tensor = torch.stack(returns)
        values_tensor = torch.stack(values)
        logprob_tensor = torch.stack(logprobs)
        entropy_tensor = torch.stack(entropies)
        advantage = returns_tensor - values_tensor.detach()
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-6)

        policy_loss = -(logprob_tensor * advantage).mean()
        value_loss = (returns_tensor - values_tensor).pow(2).mean()
        entropy_bonus = entropy_tensor.mean()
        teacher_anchor_loss = torch.tensor(0.0)
        if energy_teacher_anchor_weight > 0 and action_means:
            teacher_anchor_loss = (torch.stack(action_means) - torch.stack(teacher_actions)).pow(2).mean()
        loss = policy_loss + 0.35 * value_loss - 0.004 * entropy_bonus + energy_teacher_anchor_weight * teacher_anchor_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.7)
        optimizer.step()

        history.append(
            {
                "update": update_index + 1,
                "loss": float(loss.detach()),
                "policyLoss": float(policy_loss.detach()),
                "valueLoss": float(value_loss.detach()),
                "entropy": float(entropy_bonus.detach()),
                "teacherAnchorLoss": float(teacher_anchor_loss.detach()),
                "rewardMean": float(np.mean(reward_sum)),
                "maxStrictScore": float(max_score),
                "maxHeldSeconds": float(max_held),
                "solvedOneSecond": bool(max_held >= 1.0),
            }
        )

    env.close()
    hold_eval = evaluate(policy, mjcf_xml, links, nworld, eval_steps, "hold", force_scale)
    down_eval = evaluate(policy, mjcf_xml, links, nworld, eval_steps, "down", force_scale)
    result = {
        "schema": "six-pendulum-mjwarp-local-recurrent-ppo-smoke-v1",
        "status": "training-smoke-passed",
        "algorithm": "tiny-local-recurrent-policy-gradient",
        "links": links,
        "nworld": nworld,
        "rolloutSteps": rollout_steps,
        "rolloutSeconds": rollout_steps * 0.0025,
        "evalSteps": eval_steps,
        "evalSeconds": eval_steps * 0.0025,
        "updates": updates,
        "pose": pose,
        "seed": seed,
        "elapsedSeconds": time.time() - started,
        "parameterCount": int(sum(parameter.numel() for parameter in policy.parameters())),
        "forceScale": force_scale,
        "energyTeacherAnchorWeight": energy_teacher_anchor_weight,
        "warmstartCheckpoint": str(warmstart_checkpoint) if warmstart_checkpoint else None,
        "warmstartMetadata": warmstart_metadata,
        "history": history,
        "evaluation": {
            "hold": hold_eval,
            "down": down_eval,
        },
        "gates": {
            "strictOneSecondRequired": True,
            "trainingHorizonMeetsGate": bool(rollout_steps * 0.0025 >= 1.0),
            "promoteToNextLink": bool(down_eval["solvedOneSecond"]),
        },
        "nextRequiredWork": [
            "Run this loop on GPU with larger nworld and proper PPO minibatch updates.",
            "Replace this tiny GRU with Puffer MinGRU/PufferNet when GPU is available.",
            "Promote link count only after down-start held-out validation exceeds one second.",
        ],
    }
    result["checkpointPath"] = save_checkpoint(write_checkpoint, policy, {key: value for key, value in result.items() if key != "history"})
    return result


def expert_action_from_obs(obs, gains, force_scale: float = DEFAULT_ACTION_SCALE):
    kx, kv, kt, kw = gains
    x = obs[:, 0]
    v = obs[:, 1] * 5.0
    theta = np.arctan2(obs[:, 3], obs[:, 4])
    omega = obs[:, 7] * 8.0
    force = -(kx * x + kv * v + kt * theta + kw * omega)
    return np.clip(force / force_scale, -1.0, 1.0).reshape(len(obs), 1).astype(np.float32)


def behavior_clone(
    mjcf_xml: str,
    links: int,
    nworld: int,
    rollout_steps: int,
    eval_steps: int,
    epochs: int,
    seed: int,
    write_checkpoint: Path | None = None,
    force_scale: float = DEFAULT_ACTION_SCALE,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    started = time.time()
    gains = (8.0, 4.0, -60.0, -16.0)
    env = SixPendulumMJWarpPufferEnv(mjcf_xml, links=links, nworld=nworld, horizon=rollout_steps + 1, pose="hold", force_scale=force_scale)
    obs, _ = env.reset()
    obs_batches = []
    action_batches = []
    expert_max_held = 0.0
    expert_max_score = 0.0
    for _ in range(rollout_steps):
        action = expert_action_from_obs(obs, gains, force_scale)
        obs_batches.append(obs.copy())
        action_batches.append(action.copy())
        obs, _, _, _, infos = env.step(action)
        expert_max_held = max(expert_max_held, max(info["maxHeldSeconds"] for info in infos))
        expert_max_score = max(expert_max_score, max(info["strictScore"] for info in infos))
    env.close()

    observations = torch.as_tensor(np.concatenate(obs_batches), dtype=torch.float32)
    actions = torch.as_tensor(np.concatenate(action_batches), dtype=torch.float32)
    policy = TinyRecurrentPolicy()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3, weight_decay=1e-5)
    losses = []
    batch_size = min(512, len(observations))
    for epoch in range(epochs):
        indices = torch.randperm(len(observations))[:batch_size]
        hidden = torch.zeros(len(indices), policy.gru.hidden_size)
        mean, _, _, _ = policy(observations[indices], hidden)
        loss = (mean - actions[indices]).pow(2).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.7)
        optimizer.step()
        if epoch in {0, epochs // 4, epochs // 2, epochs - 1}:
            losses.append({"epoch": epoch + 1, "loss": float(loss.detach())})

    hold_eval = evaluate(policy, mjcf_xml, links, nworld, eval_steps, "hold", force_scale)
    down_eval = evaluate(policy, mjcf_xml, links, nworld, eval_steps, "down", force_scale)
    result = {
        "schema": "six-pendulum-mjwarp-local-stabilizer-bc-v1",
        "status": "behavior-clone-smoke-passed",
        "algorithm": "tiny-gru-stabilizer-behavior-clone",
        "links": links,
        "nworld": nworld,
        "rolloutSteps": rollout_steps,
        "evalSteps": eval_steps,
        "evalSeconds": eval_steps * 0.0025,
        "epochs": epochs,
        "seed": seed,
        "elapsedSeconds": time.time() - started,
        "parameterCount": int(sum(parameter.numel() for parameter in policy.parameters())),
        "forceScale": force_scale,
        "expert": {
            "controller": "force = -(kx*x + kv*v + kt*theta + kw*omega)",
            "gains": {"kx": gains[0], "kv": gains[1], "kt": gains[2], "kw": gains[3]},
            "maxStrictScore": float(expert_max_score),
            "maxHeldSeconds": float(expert_max_held),
            "solvedOneSecond": bool(expert_max_held >= 1.0),
        },
        "losses": losses,
        "evaluation": {
            "hold": hold_eval,
            "down": down_eval,
        },
        "gates": {
            "strictOneSecondRequired": True,
            "holdStartGatePassed": bool(hold_eval["solvedOneSecond"]),
            "promoteToNextLink": bool(down_eval["solvedOneSecond"]),
        },
        "nextRequiredWork": [
            "Use this learned stabilizer as a curriculum bootstrap, not as a final down-start solve.",
            "Train swing-up into this catch policy from down start.",
            "Replace the hand-designed stabilizer target with RL-discovered behavior at GPU scale.",
        ],
    }
    result["checkpointPath"] = save_checkpoint(write_checkpoint, policy, {key: value for key, value in result.items() if key != "losses"})
    return result


def behavior_clone_energy_teacher(
    mjcf_xml: str,
    links: int,
    nworld: int,
    rollout_steps: int,
    eval_steps: int,
    epochs: int,
    seed: int,
    write_checkpoint: Path | None = None,
    force_scale: float = 240.0,
) -> dict:
    if links != 1:
        raise ValueError("Energy teacher behavior cloning is only defined for one link")
    torch.manual_seed(seed)
    np.random.seed(seed)
    started = time.time()
    env = SixPendulumMJWarpPufferEnv(mjcf_xml, links=links, nworld=nworld, horizon=rollout_steps + 1, pose="down", force_scale=force_scale, seed=seed)
    obs, _ = env.reset(seed)
    obs_batches = []
    action_batches = []
    teacher_reward_sum = np.zeros(nworld, dtype=np.float32)
    teacher_max_held = 0.0
    teacher_max_score = 0.0
    teacher_catch_events = 0
    for _ in range(rollout_steps):
        action = energy_teacher_action_from_state(env.d.qpos.numpy(), env.d.qvel.numpy(), force_scale)
        obs_batches.append(obs.copy())
        action_batches.append(action.copy())
        obs, rewards, _, _, infos = env.step(action)
        teacher_reward_sum += rewards
        teacher_max_held = max(teacher_max_held, max(info["maxHeldSeconds"] for info in infos))
        teacher_max_score = max(teacher_max_score, max(info["strictScore"] for info in infos))
        teacher_catch_events += sum(1 for info in infos if info.get("catchBasin", 0.0) > 0.0)
    env.close()

    observations = torch.as_tensor(np.concatenate(obs_batches), dtype=torch.float32)
    actions = torch.as_tensor(np.concatenate(action_batches), dtype=torch.float32)
    policy = TinyRecurrentPolicy()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3, weight_decay=1e-5)
    losses = []
    batch_size = min(1024, len(observations))
    for epoch in range(epochs):
        indices = torch.randperm(len(observations))[:batch_size]
        hidden = torch.zeros(len(indices), policy.gru.hidden_size)
        mean, _, _, _ = policy(observations[indices], hidden)
        loss = (mean - actions[indices]).pow(2).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.7)
        optimizer.step()
        if epoch in {0, epochs // 4, epochs // 2, epochs - 1}:
            losses.append({"epoch": epoch + 1, "loss": float(loss.detach())})

    hold_eval = evaluate(policy, mjcf_xml, links, nworld, eval_steps, "hold", force_scale)
    down_eval = evaluate(policy, mjcf_xml, links, nworld, eval_steps, "down", force_scale)
    result = {
        "schema": "six-pendulum-mjwarp-local-energy-teacher-bc-v1",
        "status": "energy-teacher-bc-smoke-passed",
        "algorithm": "tiny-gru-energy-teacher-behavior-clone",
        "links": links,
        "nworld": nworld,
        "rolloutSteps": rollout_steps,
        "rolloutSeconds": rollout_steps * 0.0025,
        "evalSteps": eval_steps,
        "evalSeconds": eval_steps * 0.0025,
        "epochs": epochs,
        "seed": seed,
        "elapsedSeconds": time.time() - started,
        "parameterCount": int(sum(parameter.numel() for parameter in policy.parameters())),
        "forceScale": force_scale,
        "teacher": {
            "maxStrictScore": float(teacher_max_score),
            "maxHeldSeconds": float(teacher_max_held),
            "solvedOneSecond": bool(teacher_max_held >= 1.0),
            "rewardMean": float(np.mean(teacher_reward_sum)),
            "catchEvents": int(teacher_catch_events),
        },
        "losses": losses,
        "evaluation": {
            "hold": hold_eval,
            "down": down_eval,
        },
        "gates": {
            "strictOneSecondRequired": True,
            "teacherSolvesDownStart": bool(teacher_max_held >= 1.0),
            "learnedPolicySolvesDownStart": bool(down_eval["solvedOneSecond"]),
            "promoteToNextLink": bool(down_eval["solvedOneSecond"]),
        },
        "nextRequiredWork": [
            "Use this as PufferPPO initialization/reward scaffold, not as final proof.",
            "Run a PufferPPO/MJWarp sweep with MinGRU-sized policies once GPU spend is available.",
            "Promote only after learned held-out down-start validation exceeds one second.",
        ],
    }
    result["checkpointPath"] = save_checkpoint(write_checkpoint, policy, {key: value for key, value in result.items() if key != "losses"})
    return result


def behavior_clone_energy_teacher_sequence(
    mjcf_xml: str,
    links: int,
    nworld: int,
    rollout_steps: int,
    eval_steps: int,
    epochs: int,
    seed: int,
    write_checkpoint: Path | None = None,
    force_scale: float = 240.0,
    sequence_length: int = 160,
) -> dict:
    if links != 1:
        raise ValueError("Energy teacher sequence behavior cloning is only defined for one link")
    torch.manual_seed(seed)
    np.random.seed(seed)
    started = time.time()
    env = SixPendulumMJWarpPufferEnv(mjcf_xml, links=links, nworld=nworld, horizon=rollout_steps + 1, pose="down", force_scale=force_scale, seed=seed)
    obs, _ = env.reset(seed)
    obs_batches = []
    action_batches = []
    teacher_reward_sum = np.zeros(nworld, dtype=np.float32)
    teacher_max_held = 0.0
    teacher_max_score = 0.0
    teacher_catch_events = 0
    teacher_near_top_fast_events = 0
    for _ in range(rollout_steps):
        action = energy_teacher_action_from_state(env.d.qpos.numpy(), env.d.qvel.numpy(), force_scale)
        obs_batches.append(obs.copy())
        action_batches.append(action.copy())
        obs, rewards, _, _, infos = env.step(action)
        teacher_reward_sum += rewards
        teacher_max_held = max(teacher_max_held, max(info["maxHeldSeconds"] for info in infos))
        teacher_max_score = max(teacher_max_score, max(info["strictScore"] for info in infos))
        teacher_catch_events += sum(1 for info in infos if info.get("catchBasin", 0.0) > 0.0)
        teacher_near_top_fast_events += sum(1 for info in infos if info.get("nearTopFast", 0.0) > 0.0)
    env.close()

    observations = torch.as_tensor(np.stack(obs_batches), dtype=torch.float32)
    actions = torch.as_tensor(np.stack(action_batches), dtype=torch.float32)
    policy = TinyRecurrentPolicy()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=5e-4, weight_decay=1e-5)
    losses = []
    safe_sequence_length = max(8, min(sequence_length, rollout_steps))
    max_start = max(0, rollout_steps - safe_sequence_length)
    batch_worlds = min(nworld, 8)
    for epoch in range(epochs):
        start = int(torch.randint(0, max_start + 1, (1,)).item()) if max_start > 0 else 0
        world_indices = torch.randperm(nworld)[:batch_worlds]
        hidden = torch.zeros(len(world_indices), policy.gru.hidden_size)
        sequence_losses = []
        for offset in range(safe_sequence_length):
            mean, _, _, hidden = policy(observations[start + offset, world_indices], hidden)
            sequence_losses.append((mean - actions[start + offset, world_indices]).pow(2).mean())
        loss = torch.stack(sequence_losses).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.7)
        optimizer.step()
        if epoch in {0, epochs // 4, epochs // 2, epochs - 1}:
            losses.append({"epoch": epoch + 1, "loss": float(loss.detach())})

    hold_eval = evaluate(policy, mjcf_xml, links, nworld, eval_steps, "hold", force_scale)
    down_eval = evaluate(policy, mjcf_xml, links, nworld, eval_steps, "down", force_scale)
    result = {
        "schema": "six-pendulum-mjwarp-local-energy-teacher-sequence-bc-v1",
        "status": "energy-teacher-sequence-bc-smoke-passed",
        "algorithm": "tiny-gru-energy-teacher-sequence-behavior-clone",
        "links": links,
        "nworld": nworld,
        "rolloutSteps": rollout_steps,
        "rolloutSeconds": rollout_steps * 0.0025,
        "evalSteps": eval_steps,
        "evalSeconds": eval_steps * 0.0025,
        "epochs": epochs,
        "sequenceLength": safe_sequence_length,
        "seed": seed,
        "elapsedSeconds": time.time() - started,
        "parameterCount": int(sum(parameter.numel() for parameter in policy.parameters())),
        "forceScale": force_scale,
        "teacher": {
            "maxStrictScore": float(teacher_max_score),
            "maxHeldSeconds": float(teacher_max_held),
            "solvedOneSecond": bool(teacher_max_held >= 1.0),
            "rewardMean": float(np.mean(teacher_reward_sum)),
            "catchEvents": int(teacher_catch_events),
            "nearTopFastEvents": int(teacher_near_top_fast_events),
        },
        "losses": losses,
        "evaluation": {
            "hold": hold_eval,
            "down": down_eval,
        },
        "gates": {
            "strictOneSecondRequired": True,
            "teacherSolvesDownStart": bool(teacher_max_held >= 1.0),
            "learnedPolicySolvesDownStart": bool(down_eval["solvedOneSecond"]),
            "promoteToNextLink": bool(down_eval["solvedOneSecond"]),
        },
        "nextRequiredWork": [
            "Use sequence BC results to warm-start RL if held-out down-start is still below one second.",
            "Do not add randomized episode length until whip and catch appear in learned-policy evaluation.",
            "Promote only after learned held-out down-start validation exceeds one second.",
        ],
    }
    result["checkpointPath"] = save_checkpoint(write_checkpoint, policy, {key: value for key, value in result.items() if key != "losses"})
    return result


def collect_energy_teacher_trajectory(
    mjcf_xml: str,
    links: int,
    nworld: int,
    steps: int,
    seed: int,
    force_scale: float,
    policy: TinyRecurrentPolicy | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    env = SixPendulumMJWarpPufferEnv(mjcf_xml, links=links, nworld=nworld, horizon=steps + 1, pose="down", force_scale=force_scale, seed=seed)
    obs, _ = env.reset(seed)
    hidden = torch.zeros(nworld, policy.gru.hidden_size) if policy is not None else None
    obs_batches = []
    action_batches = []
    reward_sum = np.zeros(nworld, dtype=np.float32)
    max_held = 0.0
    max_score = 0.0
    catch_events = 0
    near_top_fast_events = 0
    for _ in range(steps):
        teacher_action = energy_teacher_action_from_state(env.d.qpos.numpy(), env.d.qvel.numpy(), force_scale)
        obs_batches.append(obs.copy())
        action_batches.append(teacher_action.copy())
        if policy is None:
            action = teacher_action
        else:
            with torch.no_grad():
                mean, _, _, hidden = policy(torch.as_tensor(obs, dtype=torch.float32), hidden)
            action = mean.numpy()
        obs, rewards, terminals, truncations, infos = env.step(action)
        if policy is not None:
            done = torch.as_tensor(terminals | truncations, dtype=torch.bool)
            if bool(done.any()):
                hidden[done] = 0.0
        reward_sum += rewards
        max_held = max(max_held, max(info["maxHeldSeconds"] for info in infos))
        max_score = max(max_score, max(info["strictScore"] for info in infos))
        catch_events += sum(1 for info in infos if info.get("catchBasin", 0.0) > 0.0)
        near_top_fast_events += sum(1 for info in infos if info.get("nearTopFast", 0.0) > 0.0)
    env.close()
    metadata = {
        "steps": steps,
        "controlledBy": "teacher" if policy is None else "learner",
        "rewardMean": float(np.mean(reward_sum)),
        "maxStrictScore": float(max_score),
        "maxHeldSeconds": float(max_held),
        "solvedOneSecond": bool(max_held >= 1.0),
        "catchEvents": int(catch_events),
        "nearTopFastEvents": int(near_top_fast_events),
    }
    return torch.as_tensor(np.stack(obs_batches), dtype=torch.float32), torch.as_tensor(np.stack(action_batches), dtype=torch.float32), metadata


def train_sequence_bc_epoch(policy, optimizer, trajectories, sequence_length: int) -> torch.Tensor:
    observations, actions = trajectories[int(torch.randint(0, len(trajectories), (1,)).item())]
    rollout_steps, nworld, _ = observations.shape
    safe_sequence_length = max(8, min(sequence_length, rollout_steps))
    max_start = max(0, rollout_steps - safe_sequence_length)
    start = int(torch.randint(0, max_start + 1, (1,)).item()) if max_start > 0 else 0
    batch_worlds = min(nworld, 8)
    world_indices = torch.randperm(nworld)[:batch_worlds]
    hidden = torch.zeros(len(world_indices), policy.gru.hidden_size)
    losses = []
    for offset in range(safe_sequence_length):
        mean, _, _, hidden = policy(observations[start + offset, world_indices], hidden)
        losses.append((mean - actions[start + offset, world_indices]).pow(2).mean())
    loss = torch.stack(losses).mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.7)
    optimizer.step()
    return loss.detach()


def behavior_clone_energy_teacher_dagger(
    mjcf_xml: str,
    links: int,
    nworld: int,
    rollout_steps: int,
    eval_steps: int,
    epochs: int,
    seed: int,
    write_checkpoint: Path | None = None,
    force_scale: float = 240.0,
    sequence_length: int = 160,
    dagger_iterations: int = 4,
) -> dict:
    if links != 1:
        raise ValueError("Energy teacher DAgger is only defined for one link")
    torch.manual_seed(seed)
    np.random.seed(seed)
    started = time.time()
    policy = TinyRecurrentPolicy()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=5e-4, weight_decay=1e-5)
    trajectories = []
    trajectory_metadata = []
    observations, actions, metadata = collect_energy_teacher_trajectory(
        mjcf_xml,
        links,
        nworld,
        rollout_steps,
        seed,
        force_scale,
        None,
    )
    trajectories.append((observations, actions))
    trajectory_metadata.append(metadata)
    losses = []
    epochs_per_iteration = max(1, epochs // max(1, dagger_iterations + 1))
    for iteration in range(dagger_iterations + 1):
        for epoch in range(epochs_per_iteration):
            loss = train_sequence_bc_epoch(policy, optimizer, trajectories, sequence_length)
            if (iteration == 0 and epoch == 0) or (iteration == dagger_iterations and epoch == epochs_per_iteration - 1):
                losses.append({"iteration": iteration, "epoch": epoch + 1, "loss": float(loss)})
        if iteration < dagger_iterations:
            learner_obs, learner_actions, learner_metadata = collect_energy_teacher_trajectory(
                mjcf_xml,
                links,
                nworld,
                rollout_steps,
                seed + iteration + 1,
                force_scale,
                policy,
            )
            trajectories.append((learner_obs, learner_actions))
            trajectory_metadata.append(learner_metadata)
    hold_eval = evaluate(policy, mjcf_xml, links, nworld, eval_steps, "hold", force_scale)
    down_eval = evaluate(policy, mjcf_xml, links, nworld, eval_steps, "down", force_scale)
    teacher_metadata = trajectory_metadata[0]
    learner_metadata = trajectory_metadata[1:]
    result = {
        "schema": "six-pendulum-mjwarp-local-energy-teacher-dagger-v1",
        "status": "energy-teacher-dagger-smoke-passed",
        "algorithm": "tiny-gru-energy-teacher-dagger",
        "links": links,
        "nworld": nworld,
        "rolloutSteps": rollout_steps,
        "rolloutSeconds": rollout_steps * 0.0025,
        "evalSteps": eval_steps,
        "evalSeconds": eval_steps * 0.0025,
        "epochs": epochs_per_iteration * (dagger_iterations + 1),
        "sequenceLength": max(8, min(sequence_length, rollout_steps)),
        "daggerIterations": dagger_iterations,
        "seed": seed,
        "elapsedSeconds": time.time() - started,
        "parameterCount": int(sum(parameter.numel() for parameter in policy.parameters())),
        "forceScale": force_scale,
        "teacher": teacher_metadata,
        "learnerRolloutLabels": learner_metadata,
        "losses": losses,
        "evaluation": {
            "hold": hold_eval,
            "down": down_eval,
        },
        "gates": {
            "strictOneSecondRequired": True,
            "teacherSolvesDownStart": bool(teacher_metadata["solvedOneSecond"]),
            "learnedPolicySolvesDownStart": bool(down_eval["solvedOneSecond"]),
            "promoteToNextLink": bool(down_eval["solvedOneSecond"]),
        },
        "nextRequiredWork": [
            "If DAgger reaches near-catch but not one-second hold, warm-start PPO from this checkpoint.",
            "Do not add randomized episode length until learned whip and catch are stable.",
            "Promote only after learned held-out down-start validation exceeds one second.",
        ],
    }
    result["checkpointPath"] = save_checkpoint(
        write_checkpoint,
        policy,
        {key: value for key, value in result.items() if key not in {"losses", "learnerRolloutLabels"}},
    )
    return result


def main():
    parser = argparse.ArgumentParser(description="Run a tiny local recurrent PPO smoke on the MJWarp env adapter.")
    parser.add_argument("--links", type=int, default=1)
    parser.add_argument("--nworld", type=int, default=4)
    parser.add_argument("--rollout-steps", type=int, default=96)
    parser.add_argument("--eval-steps", type=int, default=480)
    parser.add_argument("--updates", type=int, default=3)
    parser.add_argument("--bc-epochs", type=int, default=600)
    parser.add_argument("--pose", choices=["down", "hold", "mixed", "down-heavy"], default="hold")
    parser.add_argument("--behavior-clone-stabilizer", action="store_true")
    parser.add_argument("--energy-teacher-eval", action="store_true")
    parser.add_argument("--behavior-clone-energy-teacher", action="store_true")
    parser.add_argument("--behavior-clone-energy-teacher-sequence", action="store_true")
    parser.add_argument("--behavior-clone-energy-teacher-dagger", action="store_true")
    parser.add_argument("--warmstart-checkpoint", type=Path)
    parser.add_argument("--write-checkpoint", type=Path)
    parser.add_argument("--force-scale", type=float, default=DEFAULT_ACTION_SCALE)
    parser.add_argument("--energy-teacher-anchor-weight", type=float, default=0.0)
    parser.add_argument("--bc-sequence-length", type=int, default=160)
    parser.add_argument("--dagger-iterations", type=int, default=4)
    parser.add_argument("--seed", type=int, default=426210)
    parser.add_argument(
        "--write-result",
        type=Path,
        default=Path("local-artifacts/outputs/training-checkpoints/puffer-mjwarp-local-ppo-smoke.json"),
    )
    args = parser.parse_args()

    mjcf_path = Path(f"app/ailab/six-pendulum-cartpole/mjcf/cartpole_{args.links}_link.xml")
    if not mjcf_path.exists():
        raise FileNotFoundError(f"Missing MJCF file: {mjcf_path}")
    if args.energy_teacher_eval:
        result = evaluate_energy_teacher(
            mjcf_path.read_text(),
            args.links,
            args.nworld,
            args.eval_steps,
            args.seed,
            args.force_scale,
        )
    elif args.behavior_clone_energy_teacher:
        result = behavior_clone_energy_teacher(
            mjcf_path.read_text(),
            args.links,
            args.nworld,
            args.rollout_steps,
            args.eval_steps,
            args.bc_epochs,
            args.seed,
            args.write_checkpoint,
            args.force_scale,
        )
    elif args.behavior_clone_energy_teacher_sequence:
        result = behavior_clone_energy_teacher_sequence(
            mjcf_path.read_text(),
            args.links,
            args.nworld,
            args.rollout_steps,
            args.eval_steps,
            args.bc_epochs,
            args.seed,
            args.write_checkpoint,
            args.force_scale,
            args.bc_sequence_length,
        )
    elif args.behavior_clone_energy_teacher_dagger:
        result = behavior_clone_energy_teacher_dagger(
            mjcf_path.read_text(),
            args.links,
            args.nworld,
            args.rollout_steps,
            args.eval_steps,
            args.bc_epochs,
            args.seed,
            args.write_checkpoint,
            args.force_scale,
            args.bc_sequence_length,
            args.dagger_iterations,
        )
    elif args.behavior_clone_stabilizer:
        result = behavior_clone(
            mjcf_path.read_text(),
            args.links,
            args.nworld,
            args.rollout_steps,
            args.eval_steps,
            args.bc_epochs,
            args.seed,
            args.write_checkpoint,
            args.force_scale,
        )
    else:
        result = train(
            mjcf_path.read_text(),
            args.links,
            args.nworld,
            args.rollout_steps,
            args.eval_steps,
            args.updates,
            args.pose,
            args.seed,
            args.warmstart_checkpoint,
            args.write_checkpoint,
            args.force_scale,
            args.energy_teacher_anchor_weight,
        )
    args.write_result.parent.mkdir(parents=True, exist_ok=True)
    args.write_result.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
