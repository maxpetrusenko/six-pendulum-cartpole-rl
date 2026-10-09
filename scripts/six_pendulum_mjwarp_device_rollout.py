#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path

import numpy as np

from six_pendulum_mjwarp_gpu_kernels import (
    DEFAULT_ACTION_SCALE,
    OBS_DIM,
    WarpScoreKernel,
    record_policy_scalars_kernel,
    record_rollout_obs_kernel,
    record_rollout_scalars_kernel,
)


DEFAULT_OUTPUT = Path(
    "local-artifacts/outputs/training-checkpoints/puffer-mjwarp-device-rollout.json"
)


def build_deterministic_action_plan(steps: int, nworld: int, force_scale: float) -> np.ndarray:
    step_axis = np.linspace(0.0, 4.0 * np.pi, int(steps), endpoint=False, dtype=np.float32)
    world_phase = np.linspace(0.0, 0.75 * np.pi, int(nworld), endpoint=False, dtype=np.float32)
    force = np.sin(step_axis[:, None] + world_phase[None, :]) * 18.0
    return np.clip(force / float(force_scale), -1.0, 1.0).astype(np.float32)


def build_torch_policy(obs_dim: int, hidden_dim: int, seed: int, recurrent: bool = False):
    import torch

    torch.manual_seed(int(seed))
    if recurrent:
        class TinyRecurrentActorCritic(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = torch.nn.Linear(int(obs_dim), int(hidden_dim))
                self.rnn = torch.nn.GRUCell(int(hidden_dim), int(hidden_dim))
                self.actor = torch.nn.Linear(int(hidden_dim), 1)
                self.critic = torch.nn.Linear(int(hidden_dim), 1)
                self.log_std = torch.nn.Parameter(torch.tensor([-0.5], dtype=torch.float32))

            def forward(self, obs, hidden, deterministic: bool = True):
                encoded = torch.tanh(self.encoder(obs))
                next_hidden = self.rnn(encoded, hidden)
                mean = self.actor(next_hidden).reshape(-1)
                std = torch.exp(torch.clamp(self.log_std, -2.0, 0.5)).reshape(())
                dist = torch.distributions.Normal(mean, std)
                raw_action = mean if deterministic else dist.sample()
                action = torch.tanh(raw_action).contiguous()
                logprob = self.squashed_logprob(dist, raw_action, action)
                value = self.critic(next_hidden).reshape(-1).contiguous()
                return action, logprob, value, next_hidden

            def squashed_logprob(self, dist, raw_action, action):
                return (dist.log_prob(raw_action) - torch.log(1.0 - action * action + 1e-6)).contiguous()

            def evaluate_actions(self, obs, hidden, action):
                encoded = torch.tanh(self.encoder(obs))
                next_hidden = self.rnn(encoded, hidden)
                mean = self.actor(next_hidden).reshape(-1)
                std = torch.exp(torch.clamp(self.log_std, -2.0, 0.5)).reshape(())
                dist = torch.distributions.Normal(mean, std)
                clamped = torch.clamp(action.reshape(-1), -0.999, 0.999)
                raw_action = 0.5 * (torch.log1p(clamped) - torch.log1p(-clamped))
                logprob = self.squashed_logprob(dist, raw_action, clamped)
                entropy = dist.entropy().reshape(-1).contiguous()
                value = self.critic(next_hidden).reshape(-1).contiguous()
                return logprob, entropy, value, next_hidden

        policy = TinyRecurrentActorCritic()
        policy.eval()
        return policy

    policy = torch.nn.Sequential(
        torch.nn.Linear(int(obs_dim), int(hidden_dim)),
        torch.nn.Tanh(),
        torch.nn.Linear(int(hidden_dim), 1),
        torch.nn.Tanh(),
    )
    policy.eval()
    return policy


def run_recurrent_ppo_update_smoke(
    policy,
    obs_np: np.ndarray,
    normalized_action_np: np.ndarray,
    old_logprob_np: np.ndarray,
    old_value_np: np.ndarray,
    reward_np: np.ndarray,
    terminal_np: np.ndarray,
    truncation_np: np.ndarray,
    hidden_dim: int,
    update_epochs: int = 1,
) -> dict:
    import torch

    obs = torch.as_tensor(obs_np, dtype=torch.float32)
    actions = torch.as_tensor(normalized_action_np, dtype=torch.float32)
    old_logprob = torch.as_tensor(old_logprob_np, dtype=torch.float32)
    old_value = torch.as_tensor(old_value_np, dtype=torch.float32)
    rewards = torch.as_tensor(reward_np, dtype=torch.float32)
    done = torch.as_tensor((terminal_np > 0.5) | (truncation_np > 0.5), dtype=torch.float32)
    steps, nworld, _ = obs.shape

    gamma = 0.995
    gae_lambda = 0.95
    advantages = torch.zeros(steps, nworld, dtype=torch.float32)
    last_gae = torch.zeros(nworld, dtype=torch.float32)
    next_value = torch.zeros(nworld, dtype=torch.float32)
    for step in range(steps - 1, -1, -1):
        next_nonterminal = 1.0 - done[step]
        delta = rewards[step] + gamma * next_value * next_nonterminal - old_value[step]
        last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
        advantages[step] = last_gae
        next_value = old_value[step]
    returns = advantages + old_value
    normalized_advantage = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

    before = torch.cat([parameter.detach().flatten() for parameter in policy.parameters()])
    optimizer = torch.optim.AdamW(policy.parameters(), lr=3e-4, weight_decay=1e-5)
    clip_coef = 0.2
    epoch_history = []
    grad_norm = 0.0
    parameter_delta_l2 = 0.0
    policy_loss = torch.tensor(0.0)
    value_loss = torch.tensor(0.0)
    entropy_loss = torch.tensor(0.0)
    loss = torch.tensor(0.0)
    ratio = torch.ones_like(old_logprob)

    for epoch_index in range(max(1, int(update_epochs))):
        hidden = torch.zeros(nworld, int(hidden_dim), dtype=torch.float32)
        logprob_steps = []
        value_steps = []
        entropy_steps = []
        for step in range(steps):
            logprob, entropy, value, hidden = policy.evaluate_actions(obs[step], hidden, actions[step])
            logprob_steps.append(logprob)
            value_steps.append(value)
            entropy_steps.append(entropy)
            if step < steps - 1 and bool(done[step].any()):
                hidden = hidden.clone()
                hidden[done[step] > 0.5] = 0.0

        new_logprob = torch.stack(logprob_steps)
        new_value = torch.stack(value_steps)
        entropy = torch.stack(entropy_steps)
        ratio = torch.exp(new_logprob - old_logprob)
        pg_loss_unclipped = -normalized_advantage * ratio
        pg_loss_clipped = -normalized_advantage * torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
        policy_loss = torch.max(pg_loss_unclipped, pg_loss_clipped).mean()
        value_clipped = old_value + torch.clamp(new_value - old_value, -clip_coef, clip_coef)
        value_loss = 0.5 * torch.max((new_value - returns).pow(2), (value_clipped - returns).pow(2)).mean()
        entropy_loss = entropy.mean()
        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.7).detach())
        optimizer.step()
        after_epoch = torch.cat([parameter.detach().flatten() for parameter in policy.parameters()])
        parameter_delta_l2 = float(torch.linalg.vector_norm(after_epoch - before))
        epoch_history.append(
            {
                "epoch": epoch_index + 1,
                "policyLoss": float(policy_loss.detach()),
                "valueLoss": float(value_loss.detach()),
                "entropy": float(entropy_loss.detach()),
                "loss": float(loss.detach()),
                "ratioMean": float(ratio.detach().mean()),
                "ratioMax": float(ratio.detach().max()),
                "gradNorm": grad_norm,
                "parameterDeltaL2": parameter_delta_l2,
            }
        )

    after = torch.cat([parameter.detach().flatten() for parameter in policy.parameters()])
    parameter_delta_l2 = float(torch.linalg.vector_norm(after - before))

    return {
        "enabled": True,
        "optimizer": "AdamW",
        "learningRate": 3e-4,
        "minibatches": int(max(1, int(update_epochs))),
        "updateEpochs": int(max(1, int(update_epochs))),
        "sequenceShape": [int(steps), int(nworld), int(obs.shape[-1])],
        "actionShape": [int(steps), int(nworld)],
        "policyLoss": float(policy_loss.detach()),
        "valueLoss": float(value_loss.detach()),
        "entropy": float(entropy_loss.detach()),
        "loss": float(loss.detach()),
        "advantageMean": float(advantages.mean()),
        "advantageStd": float(advantages.std()),
        "ratioMean": float(ratio.detach().mean()),
        "ratioMax": float(ratio.detach().max()),
        "gradNorm": grad_norm,
        "parameterDeltaL2": parameter_delta_l2,
        "updatedParameters": bool(parameter_delta_l2 > 0.0),
        "epochHistory": epoch_history,
        "notes": [
            "This is a PPO update smoke over fixed recurrent rollout buffers.",
            "It proves gradients flow from buffered observations/actions/logprobs/values into the recurrent actor-critic.",
            "It is not a trained policy and does not count toward solve.",
        ],
    }


def run_device_rollout(
    mjcf_xml: str,
    links: int = 1,
    nworld: int = 128,
    steps: int = 256,
    pose: str = "down",
    force_scale: float = DEFAULT_ACTION_SCALE,
    seed: int = 426210,
    random_horizon: bool = False,
    min_horizon: int = 160,
    max_horizon: int = 512,
    record_buffer: bool = False,
    action_source: str = "scripted",
    action_plan: np.ndarray | None = None,
    policy_hidden_dim: int = 64,
    recurrent_policy: bool = False,
    ppo_update_smoke: bool = False,
    ppo_update_epochs: int = 1,
) -> dict:
    import mujoco
    import mujoco_warp as mjw
    import warp as wp

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
    horizon = max(1, int(steps) + 1)
    random_horizon_enabled = bool(random_horizon)
    if random_horizon_enabled:
        min_horizon = max(1, int(min_horizon))
        max_horizon = max(min_horizon, int(max_horizon))
    pose_hold = pose == "hold"
    if action_source not in {"scripted", "buffer", "torch-policy"}:
        raise ValueError(f"Unsupported action_source: {action_source}")
    action_plan_wp = None
    action_plan_shape = None
    torch_policy = None
    torch_policy_parameters = 0
    torch_interop = False
    if action_source == "torch-policy":
        import torch

        torch_policy = build_torch_policy(OBS_DIM, policy_hidden_dim, seed, recurrent_policy)
        torch_policy_parameters = int(sum(parameter.numel() for parameter in torch_policy.parameters()))
        torch_interop = True
        torch_hidden = torch.zeros(int(nworld), int(policy_hidden_dim), dtype=torch.float32) if recurrent_policy else None
    if action_source == "buffer":
        if action_plan is None:
            action_plan = build_deterministic_action_plan(steps, nworld, force_scale)
        action_plan = np.asarray(action_plan, dtype=np.float32)
        expected_shape = (int(steps), int(nworld))
        if action_plan.shape != expected_shape:
            raise ValueError(f"action_plan must be time-major shape {expected_shape}, got {action_plan.shape}")
        if not np.isfinite(action_plan).all():
            raise ValueError("action_plan contains non-finite values")
        action_plan = np.ascontiguousarray(action_plan, dtype=np.float32)
        action_plan_shape = [int(steps), int(nworld)]
        action_plan_wp = wp.array(action_plan.reshape(int(steps) * int(nworld)), dtype=wp.float32, device=device)
    obs_buffer = None
    reward_buffer = None
    terminal_buffer = None
    truncation_buffer = None
    action_buffer = None
    normalized_action_buffer = None
    logprob_buffer = None
    value_buffer = None
    if record_buffer:
        obs_buffer = wp.zeros(int(steps) * int(nworld) * OBS_DIM, dtype=wp.float32, device=device)
        reward_buffer = wp.zeros(int(steps) * int(nworld), dtype=wp.float32, device=device)
        terminal_buffer = wp.zeros(int(steps) * int(nworld), dtype=wp.float32, device=device)
        truncation_buffer = wp.zeros(int(steps) * int(nworld), dtype=wp.float32, device=device)
        action_buffer = wp.zeros(int(steps) * int(nworld), dtype=wp.float32, device=device)
        if action_source == "torch-policy":
            normalized_action_buffer = wp.zeros(int(steps) * int(nworld), dtype=wp.float32, device=device)
            logprob_buffer = wp.zeros(int(steps) * int(nworld), dtype=wp.float32, device=device)
            value_buffer = wp.zeros(int(steps) * int(nworld), dtype=wp.float32, device=device)

    runner.reset_worlds(
        data.qpos,
        data.qvel,
        data.ctrl,
        pose,
        seed,
        reset_all=True,
        synchronize=False,
        random_horizon=random_horizon_enabled,
        min_horizon=min_horizon,
        max_horizon=max_horizon,
    )
    mjw.forward(model, data)
    runner.score_device(data.qpos, data.qvel, synchronize=False)
    runner.initialize_prev_potential_from_current(synchronize=False)

    for step_index in range(int(steps)):
        if record_buffer:
            wp.launch(
                record_rollout_obs_kernel,
                dim=(int(nworld), OBS_DIM),
                inputs=[runner.obs_wp, int(step_index), int(nworld), obs_buffer],
                device=device,
            )
        action_vector_wp = None
        logprob_wp = None
        value_wp = None
        if action_source == "torch-policy":
            obs_torch = wp.to_torch(runner.obs_wp).reshape(int(nworld), OBS_DIM)
            with torch.no_grad():
                if recurrent_policy:
                    action_torch, logprob_torch, value_torch, torch_hidden = torch_policy(obs_torch, torch_hidden)
                else:
                    action_torch = torch_policy(obs_torch).reshape(int(nworld)).contiguous()
                    logprob_torch = torch.zeros_like(action_torch)
                    value_torch = torch.zeros_like(action_torch)
            action_vector_wp = wp.from_torch(action_torch, dtype=wp.float32)
            logprob_wp = wp.from_torch(logprob_torch, dtype=wp.float32)
            value_wp = wp.from_torch(value_torch, dtype=wp.float32)
            runner.apply_action_vector(action_vector_wp, data.ctrl, synchronize=False)
        elif action_source == "buffer":
            runner.apply_action_buffer(action_plan_wp, data.ctrl, step_index, synchronize=False)
        else:
            runner.apply_scripted_actions(data.ctrl, step_index, steps, synchronize=False)
        mjw.step(model, data)
        runner.score_device(data.qpos, data.qvel, synchronize=False)
        runner.post_step_device(pose_hold, horizon, synchronize=False)
        if record_buffer:
            wp.launch(
                record_rollout_scalars_kernel,
                dim=int(nworld),
                inputs=[
                    runner.final_reward_wp,
                    runner.terminal_wp,
                    runner.truncation_wp,
                    runner.last_action_wp,
                    int(step_index),
                    int(nworld),
                    reward_buffer,
                    terminal_buffer,
                    truncation_buffer,
                    action_buffer,
                ],
                device=device,
            )
            if action_source == "torch-policy":
                wp.launch(
                    record_policy_scalars_kernel,
                    dim=int(nworld),
                    inputs=[
                        action_vector_wp,
                        logprob_wp,
                        value_wp,
                        int(step_index),
                        int(nworld),
                        normalized_action_buffer,
                        logprob_buffer,
                        value_buffer,
                    ],
                    device=device,
                )
        runner.reset_worlds(
            data.qpos,
            data.qvel,
            data.ctrl,
            pose,
            seed,
            reset_all=False,
            synchronize=False,
            random_horizon=random_horizon_enabled,
            min_horizon=min_horizon,
            max_horizon=max_horizon,
        )
        mjw.forward(model, data)
        runner.score_device(data.qpos, data.qvel, synchronize=False)
        runner.sync_reset_potential(synchronize=False)

    wp.synchronize()
    elapsed = time.time() - started
    strict_score = runner.rollout_max_strict_score_wp.numpy()
    max_held_steps = runner.rollout_max_held_steps_wp.numpy()
    final_reward = runner.final_reward_wp.numpy()
    reset_count = runner.reset_count_wp.numpy()
    terminal = runner.terminal_wp.numpy()
    truncation = runner.truncation_wp.numpy()
    horizon_steps = runner.horizon_steps_wp.numpy()
    control_dt = float(mjm.opt.timestep)
    simulated_steps = int(nworld) * int(steps)
    rollout_buffer_summary = {
        "enabled": False,
        "fixedShape": True,
    }
    ppo_update_summary = {"enabled": False}
    if record_buffer:
        obs_np = obs_buffer.numpy().reshape(int(steps), int(nworld), OBS_DIM)
        reward_np = reward_buffer.numpy().reshape(int(steps), int(nworld))
        terminal_np = terminal_buffer.numpy().reshape(int(steps), int(nworld))
        truncation_np = truncation_buffer.numpy().reshape(int(steps), int(nworld))
        action_np = action_buffer.numpy().reshape(int(steps), int(nworld))
        normalized_action_np = None
        logprob_np = None
        value_np = None
        if action_source == "torch-policy":
            normalized_action_np = normalized_action_buffer.numpy().reshape(int(steps), int(nworld))
            logprob_np = logprob_buffer.numpy().reshape(int(steps), int(nworld))
            value_np = value_buffer.numpy().reshape(int(steps), int(nworld))
        rollout_buffer_summary = {
            "enabled": True,
            "fixedShape": True,
            "observationShape": [int(steps), int(nworld), OBS_DIM],
            "rewardShape": [int(steps), int(nworld)],
            "terminalShape": [int(steps), int(nworld)],
            "truncationShape": [int(steps), int(nworld)],
            "actionShape": [int(steps), int(nworld)],
            "actionUnits": "scaled cart force",
            "normalizedActionShape": [int(steps), int(nworld)] if action_source == "torch-policy" else [],
            "normalizedActionUnits": "normalized policy action [-1, 1]" if action_source == "torch-policy" else "",
            "logprobShape": [int(steps), int(nworld)] if action_source == "torch-policy" else [],
            "valueShape": [int(steps), int(nworld)] if action_source == "torch-policy" else [],
            "observationFinite": bool(np.isfinite(obs_np).all()),
            "rewardFinite": bool(np.isfinite(reward_np).all()),
            "actionFinite": bool(np.isfinite(action_np).all()),
            "normalizedActionFinite": bool(np.isfinite(normalized_action_np).all()) if normalized_action_np is not None else None,
            "logprobFinite": bool(np.isfinite(logprob_np).all()) if logprob_np is not None else None,
            "valueFinite": bool(np.isfinite(value_np).all()) if value_np is not None else None,
            "rewardMean": float(np.mean(reward_np)),
            "cartAbsMean": float(np.mean(np.abs(obs_np[:, :, 0]))) if obs_np.size else 0.0,
            "cartAbsMax": float(np.max(np.abs(obs_np[:, :, 0]))) if obs_np.size else 0.0,
            "logprobMean": float(np.mean(logprob_np)) if logprob_np is not None else None,
            "valueMean": float(np.mean(value_np)) if value_np is not None else None,
            "terminalCount": int(np.sum(terminal_np > 0.5)),
            "truncationCount": int(np.sum(truncation_np > 0.5)),
            "actionAbsMax": float(np.max(np.abs(action_np))) if action_np.size else 0.0,
            "normalizedActionAbsMax": float(np.max(np.abs(normalized_action_np))) if normalized_action_np is not None and normalized_action_np.size else None,
            "bytes": int(
                obs_np.nbytes
                + reward_np.nbytes
                + terminal_np.nbytes
                + truncation_np.nbytes
                + action_np.nbytes
                + (0 if normalized_action_np is None else normalized_action_np.nbytes + logprob_np.nbytes + value_np.nbytes)
            ),
            "cpuReads": "rollout buffers copied once after final synchronize",
        }
        if ppo_update_smoke:
            if action_source != "torch-policy" or not recurrent_policy:
                raise ValueError("ppo_update_smoke requires --action-source torch-policy --recurrent-policy")
            ppo_update_summary = run_recurrent_ppo_update_smoke(
                torch_policy,
                obs_np,
                normalized_action_np,
                logprob_np,
                value_np,
                reward_np,
                terminal_np,
                truncation_np,
                policy_hidden_dim,
                ppo_update_epochs,
            )

    return {
        "schema": "six-pendulum-mjwarp-device-rollout-smoke-v1",
        "status": "device-rollout-smoke-passed",
        "links": int(links),
        "nworld": int(nworld),
        "steps": int(steps),
        "pose": pose,
        "forceScale": float(force_scale),
        "seed": int(seed),
        "device": device,
        "elapsedSeconds": elapsed,
        "simulatedSteps": simulated_steps,
        "sps": simulated_steps / elapsed if elapsed > 0 else 0.0,
        "controlDt": control_dt,
        "scoreBackend": "warp-score-kernel",
        "rolloutBackend": "warp-post-step-kernel-device-loop",
        "resetBackend": "warp-reset-kernel",
        "actionBackend": "warp-policy-action-vector-kernel"
        if action_source == "torch-policy"
        else "warp-action-buffer-kernel"
        if action_source == "buffer"
        else "warp-scripted-action-kernel",
        "actionSource": "torch-policy-vector"
        if action_source == "torch-policy"
        else "external-action-buffer"
        if action_source == "buffer"
        else "scripted-kernel",
        "actionPlanShape": action_plan_shape or [],
        "actionPlanLayout": "time-major [steps, nworld]" if action_source == "buffer" else "",
        "actionPlanUnits": "normalized policy action [-1, 1]" if action_source == "buffer" else "",
        "actionPlanCopiedBeforeRollout": bool(action_source == "buffer"),
        "actionPlanCpuWritesPerStep": 0,
        "policyReadyActionInterface": bool(action_source in {"buffer", "torch-policy"}),
        "torchPolicyInterop": torch_interop,
        "torchPolicyParameters": torch_policy_parameters,
        "torchPolicyHiddenDim": int(policy_hidden_dim) if action_source == "torch-policy" else 0,
        "torchPolicyRecurrent": bool(recurrent_policy) if action_source == "torch-policy" else False,
        "torchPolicyHiddenShape": [int(nworld), int(policy_hidden_dim)] if action_source == "torch-policy" and recurrent_policy else [],
        "torchPolicyObsInterop": "wp.to_torch(runner.obs_wp)" if action_source == "torch-policy" else "",
        "torchPolicyActionInterop": "wp.from_torch(action_torch)" if action_source == "torch-policy" else "",
        "torchPolicyLogprobInterop": "wp.from_torch(logprob_torch)" if action_source == "torch-policy" and recurrent_policy else "",
        "torchPolicyValueInterop": "wp.from_torch(value_torch)" if action_source == "torch-policy" and recurrent_policy else "",
        "torchPolicyCpuActionWritesPerStep": 0 if action_source == "torch-policy" else None,
        "torchPolicyLearned": False if action_source == "torch-policy" else None,
        "randomHorizonEnabled": random_horizon_enabled,
        "randomHorizonMinSteps": int(min_horizon) if random_horizon_enabled else 0,
        "randomHorizonMaxSteps": int(max_horizon) if random_horizon_enabled else 0,
        "randomHorizonCurrentMin": int(np.min(horizon_steps)) if random_horizon_enabled and horizon_steps.size else 0,
        "randomHorizonCurrentMax": int(np.max(horizon_steps)) if random_horizon_enabled and horizon_steps.size else 0,
        "rolloutBuffer": rollout_buffer_summary,
        "ppoUpdateSmoke": ppo_update_summary,
        "cpuMetricReadsPerStep": 0,
        "cpuStateWritesPerStep": 0,
        "cpuReads": "summary arrays only after final synchronize",
        "maxStrictScore": float(np.max(strict_score)) if strict_score.size else 0.0,
        "maxHeldSeconds": float(np.max(max_held_steps) * control_dt) if max_held_steps.size else 0.0,
        "solvedOneSecond": bool(max_held_steps.size and np.max(max_held_steps) * control_dt >= 1.0),
        "rewardMean": float(np.mean(final_reward)) if final_reward.size else 0.0,
        "resetCountMean": float(np.mean(reset_count)) if reset_count.size else 0.0,
        "resetCountMax": int(np.max(reset_count)) if reset_count.size else 0,
        "terminalWorlds": int(np.sum(terminal > 0.5)) if terminal.size else 0,
        "truncatedWorlds": int(np.sum(truncation > 0.5)) if truncation.size else 0,
        "notes": [
            "This is a device-rollout substrate smoke, not training and not a learned policy.",
            "The buffer action source is a precomputed deterministic tensor consumed by a Warp kernel; it is policy-interface plumbing, not a learned policy.",
            "The torch-policy action source uses Torch/Warp tensor interop for policy output plumbing, but the policy is untrained.",
            "The recurrent torch-policy smoke records normalized actions, logprobs, and values for PPO-style rollout plumbing.",
            "The PPO update smoke performs one minibatch update from fixed recurrent buffers only.",
            "Strict score still requires continuous upright hold; subsecond flashes do not count.",
            "Randomized per-world horizons are opt-in and should stay disabled until a learned policy shows whip behavior.",
            "Rollout buffer recording is fixed-shape plumbing for a future trainer, not policy learning.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="Run a six-pendulum MJWarp rollout smoke without per-step CPU metric reads.")
    parser.add_argument("--links", type=int, default=1)
    parser.add_argument("--nworld", type=int, default=128)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--pose", choices=["down", "hold", "mixed", "down-heavy"], default="down")
    parser.add_argument("--force-scale", type=float, default=DEFAULT_ACTION_SCALE)
    parser.add_argument("--seed", type=int, default=426210)
    parser.add_argument("--random-horizon", action="store_true")
    parser.add_argument("--min-horizon", type=int, default=160)
    parser.add_argument("--max-horizon", type=int, default=512)
    parser.add_argument("--record-buffer", action="store_true")
    parser.add_argument("--action-source", choices=["scripted", "buffer", "torch-policy"], default="scripted")
    parser.add_argument("--policy-hidden-dim", type=int, default=64)
    parser.add_argument("--recurrent-policy", action="store_true")
    parser.add_argument("--ppo-update-smoke", action="store_true")
    parser.add_argument("--ppo-update-epochs", type=int, default=1)
    parser.add_argument("--write-result", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    mjcf_path = Path(f"app/ailab/six-pendulum-cartpole/mjcf/cartpole_{args.links}_link.xml")
    if not mjcf_path.exists():
        raise FileNotFoundError(f"Missing MJCF file: {mjcf_path}")
    result = run_device_rollout(
        mjcf_xml=mjcf_path.read_text(),
        links=args.links,
        nworld=args.nworld,
        steps=args.steps,
        pose=args.pose,
        force_scale=args.force_scale,
        seed=args.seed,
        random_horizon=args.random_horizon,
        min_horizon=args.min_horizon,
        max_horizon=args.max_horizon,
        record_buffer=args.record_buffer,
        action_source=args.action_source,
        policy_hidden_dim=args.policy_hidden_dim,
        recurrent_policy=args.recurrent_policy,
        ppo_update_smoke=args.ppo_update_smoke,
        ppo_update_epochs=args.ppo_update_epochs,
    )
    args.write_result.parent.mkdir(parents=True, exist_ok=True)
    args.write_result.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
