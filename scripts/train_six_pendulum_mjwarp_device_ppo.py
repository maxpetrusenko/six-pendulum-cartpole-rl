#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path

import numpy as np

from six_pendulum_mjwarp_device_rollout import build_torch_policy
from six_pendulum_mjwarp_gpu_kernels import (
    DEFAULT_ACTION_SCALE,
    OBS_DIM,
    WarpScoreKernel,
    record_policy_scalars_kernel,
    record_rollout_obs_kernel,
    record_rollout_scalars_kernel,
)


DEFAULT_OUTPUT = Path(
    "local-artifacts/outputs/training-checkpoints/puffer-mjwarp-device-ppo-train.json"
)


def expert_stabilizer_action_from_obs(obs: np.ndarray, force_scale: float) -> np.ndarray:
    x = obs[:, 0]
    cart_velocity = obs[:, 1] * 5.0
    theta = np.arctan2(obs[:, 3], obs[:, 4])
    angular_velocity = obs[:, 7] * 8.0
    force = -(8.0 * x + 4.0 * cart_velocity - 60.0 * theta - 16.0 * angular_velocity)
    return np.clip(force / force_scale, -1.0, 1.0).astype(np.float32)


def warmup_with_stabilizer_bc(
    mjcf_xml: str,
    policy,
    links: int,
    nworld: int,
    steps: int,
    epochs: int,
    seed: int,
    hidden_dim: int,
    force_scale: float,
) -> dict:
    import torch
    from six_pendulum_mjwarp_env import SixPendulumMJWarpPufferEnv

    if epochs <= 0:
        return {"enabled": False}
    started = time.time()
    env = SixPendulumMJWarpPufferEnv(
        mjcf_xml,
        links=links,
        nworld=nworld,
        horizon=steps + 1,
        pose="hold",
        force_scale=force_scale,
        seed=seed,
    )
    obs, _ = env.reset(seed)
    obs_batches = []
    action_batches = []
    expert_max_held = 0.0
    expert_max_score = 0.0
    for _ in range(int(steps)):
        action = expert_stabilizer_action_from_obs(obs, force_scale).reshape(nworld, 1)
        obs_batches.append(obs.copy())
        action_batches.append(action[:, 0].copy())
        obs, _, _, _, infos = env.step(action)
        expert_max_held = max(expert_max_held, max(info["maxHeldSeconds"] for info in infos))
        expert_max_score = max(expert_max_score, max(info["strictScore"] for info in infos))
    env.close()

    observations = torch.as_tensor(np.concatenate(obs_batches), dtype=torch.float32)
    actions = torch.as_tensor(np.concatenate(action_batches), dtype=torch.float32)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3, weight_decay=1e-5)
    losses = []
    batch_size = min(1024, len(observations))
    marker_epochs = {0, max(0, epochs // 4), max(0, epochs // 2), max(0, epochs - 1)}
    for epoch in range(int(epochs)):
        indices = torch.randperm(len(observations))[:batch_size]
        hidden = torch.zeros(len(indices), int(hidden_dim), dtype=torch.float32)
        predicted, _, _, _ = policy(observations[indices], hidden, deterministic=True)
        loss = (predicted.reshape(-1) - actions[indices]).pow(2).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.7)
        optimizer.step()
        if epoch in marker_epochs:
            losses.append({"epoch": epoch + 1, "loss": float(loss.detach())})

    return {
        "enabled": True,
        "epochs": int(epochs),
        "steps": int(steps),
        "samples": int(len(observations)),
        "batchSize": int(batch_size),
        "elapsedSeconds": time.time() - started,
        "expert": {
            "controller": "force = -(8*x + 4*v - 60*theta - 16*omega)",
            "maxStrictScore": float(expert_max_score),
            "maxHeldSeconds": float(expert_max_held),
            "solvedOneSecond": bool(expert_max_held >= 1.0),
        },
        "losses": losses,
        "notes": [
            "This is learned stabilizer warmup for the policy, not a score-counting teacher solve.",
            "Down-start promotion still requires held-out learned policy one-second hold from pure down-start.",
        ],
    }


def collect_recurrent_rollout(
    mjcf_xml: str,
    policy,
    links: int,
    nworld: int,
    steps: int,
    pose: str,
    force_scale: float,
    seed: int,
    hidden_dim: int,
    stochastic: bool,
) -> dict:
    import mujoco
    import mujoco_warp as mjw
    import torch
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
    pose_hold = pose == "hold"
    obs_buffer = wp.zeros(int(steps) * int(nworld) * OBS_DIM, dtype=wp.float32, device=device)
    reward_buffer = wp.zeros(int(steps) * int(nworld), dtype=wp.float32, device=device)
    terminal_buffer = wp.zeros(int(steps) * int(nworld), dtype=wp.float32, device=device)
    truncation_buffer = wp.zeros(int(steps) * int(nworld), dtype=wp.float32, device=device)
    action_buffer = wp.zeros(int(steps) * int(nworld), dtype=wp.float32, device=device)
    normalized_action_buffer = wp.zeros(int(steps) * int(nworld), dtype=wp.float32, device=device)
    logprob_buffer = wp.zeros(int(steps) * int(nworld), dtype=wp.float32, device=device)
    value_buffer = wp.zeros(int(steps) * int(nworld), dtype=wp.float32, device=device)
    torch_hidden = torch.zeros(int(nworld), int(hidden_dim), dtype=torch.float32)

    runner.reset_worlds(data.qpos, data.qvel, data.ctrl, pose, seed, reset_all=True, synchronize=False)
    mjw.forward(model, data)
    runner.score_device(data.qpos, data.qvel, synchronize=False)
    runner.initialize_prev_potential_from_current(synchronize=False)

    for step_index in range(int(steps)):
        obs_torch = wp.to_torch(runner.obs_wp).reshape(int(nworld), OBS_DIM)
        wp.launch(
            record_rollout_obs_kernel,
            dim=(int(nworld), OBS_DIM),
            inputs=[runner.obs_wp, int(step_index), int(nworld), obs_buffer],
            device=device,
        )
        with torch.no_grad():
            action_torch, logprob_torch, value_torch, torch_hidden = policy(
                obs_torch,
                torch_hidden,
                deterministic=not stochastic,
            )
        action_vector_wp = wp.from_torch(action_torch, dtype=wp.float32)
        logprob_wp = wp.from_torch(logprob_torch, dtype=wp.float32)
        value_wp = wp.from_torch(value_torch, dtype=wp.float32)
        runner.apply_action_vector(action_vector_wp, data.ctrl, synchronize=False)
        mjw.step(model, data)
        runner.score_device(data.qpos, data.qvel, synchronize=False)
        runner.post_step_device(pose_hold, horizon, synchronize=False)
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
        done_torch = (wp.to_torch(runner.terminal_wp) > 0.5) | (wp.to_torch(runner.truncation_wp) > 0.5)
        if bool(done_torch.any()):
            torch_hidden = torch_hidden.clone()
            torch_hidden[done_torch] = 0.0
        runner.reset_worlds(data.qpos, data.qvel, data.ctrl, pose, seed, reset_all=False, synchronize=False)
        mjw.forward(model, data)
        runner.score_device(data.qpos, data.qvel, synchronize=False)
        runner.sync_reset_potential(synchronize=False)

    wp.synchronize()
    control_dt = float(mjm.opt.timestep)
    strict_score = runner.rollout_max_strict_score_wp.numpy()
    max_held_steps = runner.rollout_max_held_steps_wp.numpy()
    reward_np = reward_buffer.numpy().reshape(int(steps), int(nworld))
    terminal_np = terminal_buffer.numpy().reshape(int(steps), int(nworld))
    truncation_np = truncation_buffer.numpy().reshape(int(steps), int(nworld))
    action_np = action_buffer.numpy().reshape(int(steps), int(nworld))
    normalized_action_np = normalized_action_buffer.numpy().reshape(int(steps), int(nworld))
    obs_np = obs_buffer.numpy().reshape(int(steps), int(nworld), OBS_DIM)
    cart_abs_np = np.abs(obs_np[:, :, 0])

    return {
        "summary": {
            "pose": pose,
            "steps": int(steps),
            "nworld": int(nworld),
            "stochastic": bool(stochastic),
            "elapsedSeconds": time.time() - started,
            "simulatedSteps": int(steps) * int(nworld),
            "sps": (int(steps) * int(nworld)) / max(time.time() - started, 1e-9),
            "rewardMean": float(np.mean(reward_np)),
            "actionAbsMean": float(np.mean(np.abs(action_np))),
            "actionAbsMax": float(np.max(np.abs(action_np))) if action_np.size else 0.0,
            "normalizedActionAbsMean": float(np.mean(np.abs(normalized_action_np))),
            "normalizedActionAbsMax": float(np.max(np.abs(normalized_action_np))) if normalized_action_np.size else 0.0,
            "cartAbsMean": float(np.mean(cart_abs_np)) if cart_abs_np.size else 0.0,
            "cartAbsMax": float(np.max(cart_abs_np)) if cart_abs_np.size else 0.0,
            "terminalCount": int(np.sum(terminal_np > 0.5)),
            "truncationCount": int(np.sum(truncation_np > 0.5)),
            "maxStrictScore": float(np.max(strict_score)) if strict_score.size else 0.0,
            "maxHeldSeconds": float(np.max(max_held_steps) * control_dt) if max_held_steps.size else 0.0,
            "solvedOneSecond": bool(max_held_steps.size and np.max(max_held_steps) * control_dt >= 1.0),
        },
        "buffers": {
            "obs": obs_np,
            "actions": normalized_action_np,
            "logprobs": logprob_buffer.numpy().reshape(int(steps), int(nworld)),
            "values": value_buffer.numpy().reshape(int(steps), int(nworld)),
            "rewards": reward_np,
            "terminals": terminal_np,
            "truncations": truncation_np,
        },
    }


def ppo_update(
    policy,
    optimizer,
    buffers: dict,
    hidden_dim: int,
    epochs: int,
    gamma: float,
    gae_lambda: float,
    clip_coef: float,
    entropy_coef: float,
) -> dict:
    import torch

    obs = torch.as_tensor(buffers["obs"], dtype=torch.float32)
    actions = torch.as_tensor(buffers["actions"], dtype=torch.float32)
    old_logprob = torch.as_tensor(buffers["logprobs"], dtype=torch.float32)
    old_value = torch.as_tensor(buffers["values"], dtype=torch.float32)
    rewards = torch.as_tensor(buffers["rewards"], dtype=torch.float32)
    done = torch.as_tensor((buffers["terminals"] > 0.5) | (buffers["truncations"] > 0.5), dtype=torch.float32)
    steps, nworld, _ = obs.shape
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
    history = []
    for epoch_index in range(max(1, int(epochs))):
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
        logratio = new_logprob - old_logprob
        approx_kl = ((ratio - 1.0) - logratio).mean()
        clipfrac = ((ratio - 1.0).abs() > float(clip_coef)).float().mean()
        policy_loss = torch.max(
            -normalized_advantage * ratio,
            -normalized_advantage * torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef),
        ).mean()
        value_clipped = old_value + torch.clamp(new_value - old_value, -clip_coef, clip_coef)
        value_loss = 0.5 * torch.max((new_value - returns).pow(2), (value_clipped - returns).pow(2)).mean()
        entropy_loss = entropy.mean()
        loss = policy_loss + 0.5 * value_loss - float(entropy_coef) * entropy_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.7).detach())
        optimizer.step()
        after_epoch = torch.cat([parameter.detach().flatten() for parameter in policy.parameters()])
        history.append(
            {
                "epoch": epoch_index + 1,
                "policyLoss": float(policy_loss.detach()),
                "valueLoss": float(value_loss.detach()),
                "entropy": float(entropy_loss.detach()),
                "loss": float(loss.detach()),
                "ratioMean": float(ratio.detach().mean()),
                "ratioMax": float(ratio.detach().max()),
                "approxKl": float(approx_kl.detach()),
                "clipFrac": float(clipfrac.detach()),
                "logStd": float(getattr(policy, "log_std").detach().reshape(-1)[0]) if hasattr(policy, "log_std") else None,
                "gradNorm": grad_norm,
                "parameterDeltaL2": float(torch.linalg.vector_norm(after_epoch - before)),
            }
        )
    after = torch.cat([parameter.detach().flatten() for parameter in policy.parameters()])
    return {
        "updateEpochs": int(max(1, int(epochs))),
        "gamma": float(gamma),
        "gaeLambda": float(gae_lambda),
        "clipCoef": float(clip_coef),
        "entropyCoef": float(entropy_coef),
        "updatedParameters": bool(float(torch.linalg.vector_norm(after - before)) > 0.0),
        "parameterDeltaL2": float(torch.linalg.vector_norm(after - before)),
        "advantageMean": float(advantages.mean()),
        "advantageStd": float(advantages.std()),
        "history": history,
    }


def train_device_ppo(
    mjcf_xml: str,
    links: int,
    nworld: int,
    rollout_steps: int,
    eval_steps: int,
    updates: int,
    update_epochs: int,
    pose: str,
    force_scale: float,
    seed: int,
    hidden_dim: int,
    eval_interval: int = 1,
    write_progress: Path | None = None,
    bc_stabilizer_epochs: int = 0,
    bc_stabilizer_steps: int = 1024,
    learning_rate: float = 3e-4,
    entropy_coef: float = 0.01,
    clip_coef: float = 0.2,
    gamma: float = 0.995,
    gae_lambda: float = 0.95,
) -> dict:
    import torch

    started = time.time()
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    policy = build_torch_policy(OBS_DIM, hidden_dim, seed, recurrent=True)
    bc_warmup = warmup_with_stabilizer_bc(
        mjcf_xml,
        policy,
        links,
        nworld,
        bc_stabilizer_steps,
        bc_stabilizer_epochs,
        seed,
        hidden_dim,
        force_scale,
    )
    optimizer = torch.optim.AdamW(policy.parameters(), lr=float(learning_rate), weight_decay=1e-5)
    history = []
    best_down = {"maxHeldSeconds": 0.0, "maxStrictScore": 0.0, "solvedOneSecond": False}
    best_hold = {"maxHeldSeconds": 0.0, "maxStrictScore": 0.0, "solvedOneSecond": False}

    def build_result(status: str) -> dict:
        return {
            "schema": "six-pendulum-mjwarp-device-ppo-training-v1",
            "status": status,
            "algorithm": "local-recurrent-ppo-on-mjwarp-rollout-buffers",
            "links": int(links),
            "nworld": int(nworld),
            "rolloutSteps": int(rollout_steps),
            "rolloutSeconds": float(rollout_steps) * 0.0025,
            "evalSteps": int(eval_steps),
            "evalSeconds": float(eval_steps) * 0.0025,
            "updates": int(updates),
            "completedUpdates": len(history),
            "updateEpochs": int(update_epochs),
            "evalInterval": max(1, int(eval_interval)),
            "pose": pose,
            "seed": int(seed),
            "forceScale": float(force_scale),
            "bcStabilizerWarmup": bc_warmup,
            "ppoHyperparameters": {
                "learningRate": float(learning_rate),
                "entropyCoef": float(entropy_coef),
                "clipCoef": float(clip_coef),
                "gamma": float(gamma),
                "gaeLambda": float(gae_lambda),
            },
            "policyParameters": int(sum(parameter.numel() for parameter in policy.parameters())),
            "elapsedSeconds": time.time() - started,
            "history": history,
            "bestDownEvaluation": best_down,
            "bestHoldEvaluation": best_hold,
            "gates": {
                "learnedPolicyOnly": True,
                "strictOneSecondRequired": True,
                "subsecondDoesNotCount": True,
                "holdStartSolvedOneSecond": bool(best_hold["solvedOneSecond"]),
                "promoteToNextLink": bool(best_down["solvedOneSecond"]),
            },
            "notes": [
                "This repeatedly collects stochastic recurrent policy rollouts, updates the same policy with PPO, then evaluates deterministic down-start and hold-start behavior.",
                "Local Mac execution still reports a CPU Warp/MJWarp device; this is a correctness path before Modal/GPU scale.",
                "A score is counted only when held-out down-start maxHeldSeconds is at least one second.",
                "Partial progress is written after each update when writeProgress/writeResult is provided.",
            ],
        }

    def write_progress_result():
        if write_progress is None:
            return
        write_progress.parent.mkdir(parents=True, exist_ok=True)
        write_progress.write_text(json.dumps(build_result("device-ppo-training-running"), indent=2) + "\n")

    for update_index in range(int(updates)):
        rollout = collect_recurrent_rollout(
            mjcf_xml,
            policy,
            links,
            nworld,
            rollout_steps,
            pose,
            force_scale,
            seed + update_index,
            hidden_dim,
            stochastic=True,
        )
        update = ppo_update(
            policy,
            optimizer,
            rollout["buffers"],
            hidden_dim,
            update_epochs,
            gamma,
            gae_lambda,
            clip_coef,
            entropy_coef,
        )
        evaluation = {}
        should_eval = (update_index + 1) % max(1, int(eval_interval)) == 0 or update_index + 1 == int(updates)
        if should_eval:
            down_eval = collect_recurrent_rollout(
                mjcf_xml,
                policy,
                links,
                nworld,
                eval_steps,
                "down",
                force_scale,
                seed + 10_000 + update_index,
                hidden_dim,
                stochastic=False,
            )["summary"]
            hold_eval = collect_recurrent_rollout(
                mjcf_xml,
                policy,
                links,
                nworld,
                eval_steps,
                "hold",
                force_scale,
                seed + 20_000 + update_index,
                hidden_dim,
                stochastic=False,
            )["summary"]
            if down_eval["maxHeldSeconds"] > best_down["maxHeldSeconds"]:
                best_down = down_eval
            if hold_eval["maxHeldSeconds"] > best_hold["maxHeldSeconds"]:
                best_hold = hold_eval
            evaluation = {
                "down": down_eval,
                "hold": hold_eval,
            }
        history.append(
            {
                "update": update_index + 1,
                "rollout": rollout["summary"],
                "ppo": update,
                "evaluation": evaluation,
                "countsTowardSolve": bool(evaluation.get("down", {}).get("solvedOneSecond", False)),
            }
        )
        print(json.dumps({"update": update_index + 1, "rollout": rollout["summary"], "evaluation": evaluation}, sort_keys=True), flush=True)
        write_progress_result()

    return build_result("device-ppo-training-finished")


def main():
    parser = argparse.ArgumentParser(description="Train a recurrent PPO policy on MJWarp device rollout buffers.")
    parser.add_argument("--links", type=int, default=1)
    parser.add_argument("--nworld", type=int, default=8)
    parser.add_argument("--rollout-steps", type=int, default=96)
    parser.add_argument("--eval-steps", type=int, default=480)
    parser.add_argument("--updates", type=int, default=4)
    parser.add_argument("--update-epochs", type=int, default=2)
    parser.add_argument("--pose", choices=["down", "hold", "mixed", "down-heavy"], default="down")
    parser.add_argument("--force-scale", type=float, default=DEFAULT_ACTION_SCALE)
    parser.add_argument("--policy-hidden-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=426210)
    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument("--bc-stabilizer-epochs", type=int, default=0)
    parser.add_argument("--bc-stabilizer-steps", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--write-result", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    mjcf_path = Path(f"app/ailab/six-pendulum-cartpole/mjcf/cartpole_{args.links}_link.xml")
    if not mjcf_path.exists():
        raise FileNotFoundError(f"Missing MJCF file: {mjcf_path}")
    result = train_device_ppo(
        mjcf_path.read_text(),
        args.links,
        args.nworld,
        args.rollout_steps,
        args.eval_steps,
        args.updates,
        args.update_epochs,
        args.pose,
        args.force_scale,
        args.seed,
        args.policy_hidden_dim,
        args.eval_interval,
        args.write_result,
        args.bc_stabilizer_epochs,
        args.bc_stabilizer_steps,
        args.learning_rate,
        args.entropy_coef,
        args.clip_coef,
        args.gamma,
        args.gae_lambda,
    )
    args.write_result.parent.mkdir(parents=True, exist_ok=True)
    args.write_result.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
