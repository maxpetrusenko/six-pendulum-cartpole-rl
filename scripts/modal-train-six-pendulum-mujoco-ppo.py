import json
import math
import time
from pathlib import Path

import modal


app = modal.App("six-pendulum-mujoco-ppo-train")
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "mujoco==3.3.7",
    "numpy==2.2.6",
    "torch==2.7.1",
)


@app.function(image=image, gpu="L4", timeout=7200)
def train_policy(mjcf_xml: str, links: int = 1, smoke: bool = True, seed: int = 426210, curriculum: str = "down") -> str:
    import mujoco
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.distributions import Normal

    started = time.time()
    safe_links = max(1, min(6, int(links)))
    safe_curriculum = curriculum if curriculum in {"down", "hold", "mixed"} else "down"
    torch.manual_seed(seed + safe_links)
    rng = np.random.default_rng(seed + safe_links)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    max_links = 6
    obs_dim = 3 + max_links * 5
    hidden_dim = 96
    action_scale = 32.0
    control_skip = 10
    control_dt = 0.0025 * control_skip
    horizon = 240 if smoke else 420
    num_envs = 32 if smoke and safe_curriculum in {"down", "hold"} else (16 if smoke else 48)
    rollout_steps = 128 if smoke and safe_curriculum in {"down", "hold"} else (96 if smoke else 160)
    updates = 28 if smoke and safe_curriculum == "down" else (24 if smoke and safe_curriculum == "hold" else (10 if smoke else 80))
    minibatch_size = 384 if smoke else 768
    train_epochs = 3 if smoke else 4
    gamma = 0.985
    gae_lambda = 0.92
    clip_coef = 0.2
    entropy_coef = 0.003 if safe_curriculum == "hold" else 0.01
    value_coef = 0.45
    max_grad_norm = 0.7
    score_max_upright_angle = 0.16
    score_max_chain_bend = 0.14

    model = mujoco.MjModel.from_xml_string(mjcf_xml)

    class MujocoCartpoleEnv:
        def __init__(self, env_index: int):
            self.data = mujoco.MjData(model)
            self.env_index = env_index
            self.elapsed = 0
            self.pose = "hold"
            self.consecutive_held_steps = 0
            self.max_consecutive_held_steps = 0
            self.last_action = 0.0
            self.reset("hold")

        def reset(self, pose: str = "mixed"):
            mujoco.mj_resetData(model, self.data)
            self.elapsed = 0
            self.pose = pose
            self.consecutive_held_steps = 0
            self.max_consecutive_held_steps = 0
            self.last_action = 0.0
            self.data.qpos[0] = rng.normal(0.0, 0.015)
            self.data.qvel[0] = rng.normal(0.0, 0.03)
            if pose == "hold":
                hinge = rng.normal(0.0, 0.045, size=safe_links)
            elif pose == "down":
                hinge = np.array([math.pi - index * 0.05 for index in range(safe_links)])
                hinge += rng.normal(0.0, 0.045, size=safe_links)
            else:
                if rng.random() < 0.45:
                    hinge = rng.normal(0.0, 0.055, size=safe_links)
                else:
                    hinge = np.array([math.pi - index * 0.05 for index in range(safe_links)])
                    hinge += rng.normal(0.0, 0.06, size=safe_links)
            self.data.qpos[1 : 1 + safe_links] = hinge
            self.data.qvel[1 : 1 + safe_links] = rng.normal(0.0, 0.05, size=safe_links)
            mujoco.mj_forward(model, self.data)
            return self.obs()

        def angles(self):
            relative = np.asarray(self.data.qpos[1 : 1 + safe_links], dtype=np.float32)
            relative = (relative + np.pi) % (2 * np.pi) - np.pi
            absolute = np.cumsum(relative)
            absolute = (absolute + np.pi) % (2 * np.pi) - np.pi
            rel_vel = np.asarray(self.data.qvel[1 : 1 + safe_links], dtype=np.float32)
            abs_vel = np.cumsum(rel_vel)
            return relative, absolute, abs_vel

        def obs(self):
            relative, absolute, abs_vel = self.angles()
            out = np.zeros(obs_dim, dtype=np.float32)
            out[0] = float(self.data.qpos[0])
            out[1] = float(self.data.qvel[0]) / 5.0
            out[2] = self.last_action / action_scale
            cursor = 3
            for index in range(max_links):
                if index < safe_links:
                    out[cursor] = math.sin(float(absolute[index]))
                    out[cursor + 1] = math.cos(float(absolute[index]))
                    out[cursor + 2] = math.sin(float(relative[index]))
                    out[cursor + 3] = math.cos(float(relative[index]))
                    out[cursor + 4] = float(abs_vel[index]) / 8.0
                cursor += 5
            return out

        def strict_score(self):
            relative, absolute, abs_vel = self.angles()
            max_upright_error = float(np.max(np.abs(absolute)))
            max_bend_error = float(np.max(np.abs(relative[1:]))) if safe_links > 1 else 0.0
            if max_upright_error > score_max_upright_angle or max_bend_error > score_max_chain_bend:
                return 0.0
            weights = np.arange(1, safe_links + 1, dtype=np.float32)
            angle_error = float(np.sum(np.abs(absolute) * weights))
            velocity_error = float(np.sum(np.abs(abs_vel)))
            raw = 100.0 - angle_error * 12.0 - max_bend_error * 30.0 - velocity_error * 2.5 - abs(float(self.data.qpos[0])) * 8.0
            return float(np.clip(raw, 0.0, 100.0))

        def step(self, action: float):
            self.data.ctrl[0] = float(np.clip(action, -action_scale, action_scale))
            self.last_action = float(self.data.ctrl[0])
            for _ in range(control_skip):
                mujoco.mj_step(model, self.data)
            self.elapsed += 1
            relative, absolute, abs_vel = self.angles()
            score = self.strict_score()
            if score > 82:
                self.consecutive_held_steps += 1
            else:
                self.consecutive_held_steps = 0
            self.max_consecutive_held_steps = max(self.max_consecutive_held_steps, self.consecutive_held_steps)
            mean_upright_error = float(np.mean(np.abs(absolute)))
            max_upright_error = float(np.max(np.abs(absolute)))
            mean_speed = float(np.mean(np.abs(abs_vel)))
            max_bend_error = float(np.max(np.abs(relative[1:]))) if safe_links > 1 else 0.0
            mean_tip_height = float(np.mean((np.cos(absolute) + 1.0) * 0.5))
            near_top = 1.0 if max_upright_error < 0.35 and max_bend_error < 0.25 else 0.0
            dense_alignment = math.exp(-mean_upright_error * 1.15 - max_bend_error * 2.0 - mean_speed * 0.08)
            whip = 1.0 if abs(float(absolute[-1])) < 0.65 and mean_speed > 1.0 else 0.0
            late = self.elapsed / max(horizon, 1)
            reward = mean_tip_height * (0.04 + late * 0.18)
            reward += dense_alignment * (0.05 + late * 0.18)
            reward += near_top * (0.2 - mean_speed * 0.015)
            reward += whip * (0.10 if self.elapsed < horizon * 0.65 else 0.03)
            reward += (score / 100.0) ** 2 * (0.8 + late * 2.2)
            reward += min(self.consecutive_held_steps * control_dt, 2.0) ** 2 * 1.6
            reward += 1.0 if score > 82 else 0.0
            reward += 2.4 if score > 92 else 0.0
            reward -= abs(float(self.data.qpos[0])) * 0.015
            reward -= (float(self.data.ctrl[0]) / action_scale) ** 2 * 0.002
            done = self.elapsed >= horizon or abs(float(self.data.qpos[0])) > 2.35
            return self.obs(), float(reward), bool(done), {
                "score": score,
                "held": 1.0 if score > 82 else 0.0,
                "whip": whip,
                "maxHeldSeconds": self.max_consecutive_held_steps * control_dt,
            }

    class RecurrentActorCritic(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(nn.Linear(obs_dim, 128), nn.Tanh(), nn.Linear(128, hidden_dim), nn.Tanh())
            self.gru = nn.GRU(hidden_dim, hidden_dim)
            self.actor = nn.Linear(hidden_dim, 1)
            self.critic = nn.Linear(hidden_dim, 1)
            self.log_std = nn.Parameter(torch.tensor([math.log(3.0)]))

        def forward(self, obs, hidden):
            x = self.encoder(obs)
            y, next_hidden = self.gru(x.unsqueeze(0), hidden)
            y = y.squeeze(0)
            mean = torch.tanh(self.actor(y)) * action_scale
            value = self.critic(y).squeeze(-1)
            return mean.squeeze(-1), self.log_std.exp().expand_as(mean.squeeze(-1)), value, next_hidden

        def evaluate_actions(self, obs, hidden, action):
            mean, std, value, _ = self.forward(obs, hidden)
            dist = Normal(mean, std)
            logprob = dist.log_prob(action)
            entropy = dist.entropy()
            return logprob, entropy, value

    policy = RecurrentActorCritic().to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=3e-4, weight_decay=1e-5)
    envs = [MujocoCartpoleEnv(index) for index in range(num_envs)]
    obs_np = np.stack([env.reset("hold") for env in envs])
    hidden = torch.zeros(1, num_envs, hidden_dim, device=device)
    episode_returns = np.zeros(num_envs, dtype=np.float32)
    completed_returns = []
    history = []

    def reset_pose(update_index: int):
        if safe_curriculum == "down":
            return "down"
        if safe_curriculum == "hold":
            return "hold"
        if safe_links == 1:
            return "mixed" if update_index > updates * 0.75 else "hold"
        if update_index < updates * 0.35:
            return "hold"
        return "mixed"

    def collect_rollout(update_index: int):
        nonlocal obs_np, hidden
        obs_buf = []
        hidden_buf = []
        action_buf = []
        logprob_buf = []
        reward_buf = []
        done_buf = []
        value_buf = []
        score_buf = []
        held_buf = []
        whip_buf = []
        for _ in range(rollout_steps):
            obs_t = torch.tensor(obs_np, dtype=torch.float32, device=device)
            hidden_in = hidden.detach()
            with torch.no_grad():
                mean, std, value, next_hidden = policy(obs_t, hidden_in)
                dist = Normal(mean, std)
                action = dist.sample().clamp(-action_scale, action_scale)
                logprob = dist.log_prob(action)
            next_obs = []
            rewards = []
            dones = []
            scores = []
            helds = []
            whips = []
            for env_index, env in enumerate(envs):
                ob, reward, done, info = env.step(float(action[env_index].detach().cpu()))
                episode_returns[env_index] += reward
                if done:
                    completed_returns.append(float(episode_returns[env_index]))
                    episode_returns[env_index] = 0.0
                    ob = env.reset(reset_pose(update_index))
                    next_hidden[:, env_index, :] = 0
                next_obs.append(ob)
                rewards.append(reward)
                dones.append(done)
                scores.append(info["score"])
                helds.append(info["held"])
                whips.append(info["whip"])
            obs_buf.append(obs_t.detach())
            hidden_buf.append(hidden_in.squeeze(0).detach())
            action_buf.append(action.detach())
            logprob_buf.append(logprob.detach())
            reward_buf.append(torch.tensor(rewards, dtype=torch.float32, device=device))
            done_buf.append(torch.tensor(dones, dtype=torch.float32, device=device))
            value_buf.append(value.detach())
            score_buf.append(torch.tensor(scores, dtype=torch.float32, device=device))
            held_buf.append(torch.tensor(helds, dtype=torch.float32, device=device))
            whip_buf.append(torch.tensor(whips, dtype=torch.float32, device=device))
            obs_np = np.stack(next_obs)
            hidden = next_hidden.detach()
        with torch.no_grad():
            _, _, next_value, _ = policy(torch.tensor(obs_np, dtype=torch.float32, device=device), hidden)
        return {
            "obs": torch.stack(obs_buf),
            "hidden": torch.stack(hidden_buf),
            "action": torch.stack(action_buf),
            "logprob": torch.stack(logprob_buf),
            "reward": torch.stack(reward_buf),
            "done": torch.stack(done_buf),
            "value": torch.stack(value_buf),
            "score": torch.stack(score_buf),
            "held": torch.stack(held_buf),
            "whip": torch.stack(whip_buf),
            "next_value": next_value.detach(),
        }

    def advantages(batch):
        rewards = batch["reward"]
        dones = batch["done"]
        values = batch["value"]
        adv = torch.zeros_like(rewards, device=device)
        lastgaelam = torch.zeros(num_envs, device=device)
        for step in reversed(range(rollout_steps)):
            next_nonterminal = 1.0 - dones[step]
            next_values = batch["next_value"] if step == rollout_steps - 1 else values[step + 1]
            delta = rewards[step] + gamma * next_values * next_nonterminal - values[step]
            lastgaelam = delta + gamma * gae_lambda * next_nonterminal * lastgaelam
            adv[step] = lastgaelam
        returns = adv + values
        return adv, returns

    def update_policy(batch):
        adv, returns = advantages(batch)
        flat_obs = batch["obs"].reshape(-1, obs_dim)
        flat_hidden = batch["hidden"].reshape(-1, hidden_dim).unsqueeze(0)
        flat_action = batch["action"].reshape(-1)
        flat_logprob = batch["logprob"].reshape(-1)
        flat_adv = adv.reshape(-1)
        flat_returns = returns.reshape(-1)
        flat_values = batch["value"].reshape(-1)
        flat_adv = (flat_adv - flat_adv.mean()) / (flat_adv.std().clamp(min=1e-6))
        count = flat_obs.shape[0]
        indices = torch.randperm(count, device=device)
        losses = []
        for _ in range(train_epochs):
            for start in range(0, count, minibatch_size):
                mb = indices[start : start + minibatch_size]
                logprob, entropy, value = policy.evaluate_actions(flat_obs[mb], flat_hidden[:, mb, :].contiguous(), flat_action[mb])
                ratio = (logprob - flat_logprob[mb]).exp()
                pg_loss_1 = -flat_adv[mb] * ratio
                pg_loss_2 = -flat_adv[mb] * torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
                pg_loss = torch.max(pg_loss_1, pg_loss_2).mean()
                value_clipped = flat_values[mb] + torch.clamp(value - flat_values[mb], -clip_coef, clip_coef)
                value_loss = 0.5 * torch.max((value - flat_returns[mb]).pow(2), (value_clipped - flat_returns[mb]).pow(2)).mean()
                entropy_loss = entropy.mean()
                loss = pg_loss + value_coef * value_loss - entropy_coef * entropy_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
        return float(np.mean(losses)) if losses else 0.0

    def validate(pose: str, episodes: int = 24):
        total_score = []
        total_held = []
        total_whip = []
        total_max_held_seconds = []
        returns = []
        for episode in range(episodes):
            env = MujocoCartpoleEnv(1000 + episode)
            obs = env.reset(pose)
            h = torch.zeros(1, 1, hidden_dim, device=device)
            ep_score = []
            ep_held = []
            ep_whip = []
            ep_max_held_seconds = 0.0
            ep_return = 0.0
            for _ in range(horizon):
                with torch.no_grad():
                    obs_t = torch.tensor(obs, dtype=torch.float32, device=device).view(1, -1)
                    mean, _, _, h = policy(obs_t, h)
                obs, reward, done, info = env.step(float(mean[0].detach().cpu()))
                ep_return += reward
                ep_score.append(info["score"])
                ep_held.append(info["held"])
                ep_whip.append(info["whip"])
                ep_max_held_seconds = max(ep_max_held_seconds, info["maxHeldSeconds"])
                if done:
                    break
            total_score.append(ep_score[-1] if ep_score else 0.0)
            total_held.append(float(np.mean(ep_held)) if ep_held else 0.0)
            total_whip.append(float(np.mean(ep_whip)) if ep_whip else 0.0)
            total_max_held_seconds.append(ep_max_held_seconds)
            returns.append(ep_return)
        return {
            "pose": pose,
            "episodes": episodes,
            "score": float(np.mean(total_score)),
            "scoreP10": float(np.quantile(total_score, 0.1)),
            "held": float(np.mean(total_held)),
            "heldP10": float(np.quantile(total_held, 0.1)),
            "maxHeldSeconds": float(np.mean(total_max_held_seconds)),
            "maxHeldSecondsP10": float(np.quantile(total_max_held_seconds, 0.1)),
            "solvedOneSecondRate": float(np.mean(np.asarray(total_max_held_seconds) >= 1.0)),
            "whip": float(np.mean(total_whip)),
            "return": float(np.mean(returns)),
        }

    best_state = None
    best_validation = None
    best_validation_score = -1.0
    for update_index in range(updates):
        batch = collect_rollout(update_index)
        loss = update_policy(batch)
        if update_index % 2 == 0 or update_index == updates - 1:
            selection_pose = "down" if safe_curriculum == "down" else ("hold" if safe_curriculum == "hold" else "mixed")
            selection_validation = validate(selection_pose, 8 if smoke else 16)
            selection_score = (
                selection_validation["solvedOneSecondRate"] * 1000.0
                + selection_validation["maxHeldSeconds"] * 100.0
                + selection_validation["score"]
                + selection_validation["held"] * 10.0
            )
            if selection_score > best_validation_score:
                best_validation_score = selection_score
                best_validation = selection_validation
                best_state = {key: value.detach().cpu().clone() for key, value in policy.state_dict().items()}
            line = {
                "update": update_index,
                "links": safe_links,
                "loss": loss,
                "meanReward": float(batch["reward"].mean().detach().cpu()),
                "meanScore": float(batch["score"].mean().detach().cpu()),
                "held": float(batch["held"].mean().detach().cpu()),
                "whip": float(batch["whip"].mean().detach().cpu()),
                "completedReturn": float(np.mean(completed_returns[-16:])) if completed_returns else 0.0,
                "selectionPose": selection_pose,
                "selectionMaxHeldSeconds": selection_validation["maxHeldSeconds"],
                "selectionSolvedOneSecondRate": selection_validation["solvedOneSecondRate"],
                "selectionScore": selection_validation["score"],
            }
            print(json.dumps(line), flush=True)
            history.append(line)

    if best_state is not None:
        policy.load_state_dict({key: value.to(device) for key, value in best_state.items()})
    down_validation = validate("down", 16 if smoke else 48)
    hold_validation = validate("hold", 16 if smoke else 48)
    mixed_validation = validate("mixed", 16 if smoke else 48)
    print(json.dumps({"downValidation": down_validation, "holdValidation": hold_validation, "mixedValidation": mixed_validation}), flush=True)
    output = {
        "trainedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "algorithm": "modal-mujoco-recurrent-ppo",
        "environment": f"mujoco-cartpole-{safe_links}-link",
        "seed": seed + safe_links,
        "links": safe_links,
        "modelType": "gruActorCritic",
        "obsDim": obs_dim,
        "hiddenDim": hidden_dim,
        "actionScale": action_scale,
        "training": {
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "torch": torch.__version__,
            "mujoco": mujoco.__version__,
            "elapsedSeconds": round(time.time() - started, 3),
            "smoke": smoke,
            "numEnvs": num_envs,
            "rolloutSteps": rollout_steps,
            "updates": updates,
            "horizon": horizon,
            "controlSkip": control_skip,
            "curriculum": safe_curriculum,
            "history": history,
            "validation": down_validation if safe_curriculum == "down" else mixed_validation,
            "bestValidation": best_validation,
            "downValidation": down_validation,
            "holdValidation": hold_validation,
            "mixedValidation": mixed_validation,
            "strictScore": {
                "maxUprightAngleRad": score_max_upright_angle,
                "maxChainBendRad": score_max_chain_bend,
            },
            "sourceClues": [
                "Yacine thread: lower link curriculum before six links.",
                "Yacine thread: recurrent policies benefit from randomized episode length after whip behavior appears.",
                "PufferLib clue: PPO plus small recurrent policy, not fixed time-knot search.",
                "MuJoCo Python bindings load MJCF and step the simulator directly.",
            ],
        },
        "stateDict": {key: value.detach().cpu().tolist() for key, value in policy.state_dict().items()},
    }
    return json.dumps(output, indent=2)


@app.local_entrypoint()
def main(links: int = 1, smoke: bool = True, curriculum: str = "down"):
    path = Path(f"app/ailab/six-pendulum-cartpole/mjcf/cartpole_{links}_link.xml")
    if not path.exists():
        raise FileNotFoundError(f"Missing MJCF file: {path}")
    return train_policy.remote(path.read_text(), links, smoke, 426210, curriculum)
