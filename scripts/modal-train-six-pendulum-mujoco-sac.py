import json
import math
import time
from pathlib import Path

import modal


app = modal.App("six-pendulum-mujoco-sac-train")
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "gymnasium==1.1.1",
    "mujoco==3.3.7",
    "numpy==2.2.6",
    "stable-baselines3==2.6.0",
    "torch==2.7.1",
)


@app.function(image=image, gpu="L4", timeout=7200)
def train_policy(
    mjcf_xml: str,
    links: int = 1,
    smoke: bool = True,
    seed: int = 426210,
    curriculum: str = "down",
    algo: str = "sac",
) -> str:
    import gymnasium as gym
    import mujoco
    import numpy as np
    import torch
    import torch.nn as nn
    from gymnasium import spaces
    from stable_baselines3 import SAC, TD3
    from stable_baselines3.common.noise import NormalActionNoise
    from stable_baselines3.common.vec_env import DummyVecEnv

    started = time.time()
    safe_links = max(1, min(6, int(links)))
    safe_curriculum = curriculum if curriculum in {"down", "hold", "stabilize-down"} else "down"
    safe_algo = algo if algo in {"sac", "td3"} else "sac"
    max_links = 6
    obs_dim = 3 + max_links * 5
    action_scale = 32.0
    control_skip = 10
    control_dt = 0.0025 * control_skip
    horizon = 360 if smoke else 560
    total_timesteps = 80_000 if smoke else 420_000
    eval_every = 10_000 if smoke else 35_000
    score_max_upright_angle = 0.16
    score_max_chain_bend = 0.14

    model_spec = mujoco.MjModel.from_xml_string(mjcf_xml)

    class OneLinkCartpoleEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self, env_index: int = 0, eval_mode: bool = False, pose_mode: str = "down"):
            super().__init__()
            self.model = model_spec
            self.data = mujoco.MjData(self.model)
            self.env_index = env_index
            self.eval_mode = eval_mode
            self.pose_mode = pose_mode
            self.rng = np.random.default_rng(seed + env_index * 9973 + (1_000_000 if eval_mode else 0))
            self.elapsed = 0
            self.last_action = 0.0
            self.consecutive_held_steps = 0
            self.max_consecutive_held_steps = 0
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
            self.action_space = spaces.Box(low=np.array([-action_scale], dtype=np.float32), high=np.array([action_scale], dtype=np.float32))

        def sample_hinge(self):
            if self.pose_mode == "hold":
                return self.rng.normal(0.0, 0.045 if self.eval_mode else 0.08, size=safe_links)
            if self.pose_mode == "mixed" and self.rng.random() < 0.45:
                return self.rng.normal(0.0, 0.07 if self.eval_mode else 0.11, size=safe_links)
            hinge = np.array([math.pi - index * 0.05 for index in range(safe_links)], dtype=np.float64)
            hinge += self.rng.normal(0.0, 0.035 if self.eval_mode else 0.055, size=safe_links)
            return hinge

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            mujoco.mj_resetData(self.model, self.data)
            self.elapsed = 0
            self.last_action = 0.0
            self.consecutive_held_steps = 0
            self.max_consecutive_held_steps = 0
            self.data.qpos[0] = self.rng.normal(0.0, 0.012)
            self.data.qvel[0] = self.rng.normal(0.0, 0.02)
            self.data.qpos[1 : 1 + safe_links] = self.sample_hinge()
            self.data.qvel[1 : 1 + safe_links] = self.rng.normal(0.0, 0.04 if self.eval_mode else 0.08, size=safe_links)
            mujoco.mj_forward(self.model, self.data)
            return self.obs(), {}

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

        def step(self, action):
            ctrl = float(np.clip(np.asarray(action, dtype=np.float32)[0], -action_scale, action_scale))
            self.data.ctrl[0] = ctrl
            self.last_action = ctrl
            for _ in range(control_skip):
                mujoco.mj_step(self.model, self.data)
            self.elapsed += 1
            relative, absolute, abs_vel = self.angles()
            score = self.strict_score()
            if score > 82:
                self.consecutive_held_steps += 1
            else:
                self.consecutive_held_steps = 0
            self.max_consecutive_held_steps = max(self.max_consecutive_held_steps, self.consecutive_held_steps)

            max_upright_error = float(np.max(np.abs(absolute)))
            max_bend_error = float(np.max(np.abs(relative[1:]))) if safe_links > 1 else 0.0
            mean_speed = float(np.mean(np.abs(abs_vel)))
            tip_height = float(np.mean((np.cos(absolute) + 1.0) * 0.5))
            upright_shape = math.exp(-max_upright_error * 1.7 - max_bend_error * 2.5 - mean_speed * 0.06)
            held_seconds = self.consecutive_held_steps * control_dt

            reward = tip_height * 1.8
            reward += upright_shape * 0.8
            reward += (score / 100.0) ** 2 * 8.0
            reward += min(held_seconds, 2.0) ** 2 * 18.0
            reward += 6.0 if score > 82 else 0.0
            reward += 20.0 if held_seconds >= 1.0 else 0.0
            reward -= abs(float(self.data.qpos[0])) * 0.03
            reward -= (ctrl / action_scale) ** 2 * 0.01

            terminated = abs(float(self.data.qpos[0])) > 2.35
            truncated = self.elapsed >= horizon
            info = {
                "score": score,
                "held": 1.0 if score > 82 else 0.0,
                "maxHeldSeconds": self.max_consecutive_held_steps * control_dt,
                "maxUprightError": max_upright_error,
                "tipHeight": tip_height,
            }
            return self.obs(), float(reward), bool(terminated), bool(truncated), info

    def make_env(env_index: int, eval_mode: bool = False, pose_mode: str = "down"):
        return lambda: OneLinkCartpoleEnv(env_index, eval_mode, pose_mode)

    def validate(model, episodes: int = 24, pose_mode: str = "down"):
        total_score = []
        total_held = []
        total_max_held_seconds = []
        total_tip_height = []
        total_upright_error = []
        returns = []
        for episode in range(episodes):
            env = OneLinkCartpoleEnv(10_000 + episode, eval_mode=True, pose_mode=pose_mode)
            obs, _ = env.reset()
            ep_score = []
            ep_held = []
            ep_tip_height = []
            ep_upright_error = []
            ep_return = 0.0
            ep_max_held_seconds = 0.0
            for _ in range(horizon):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_return += reward
                ep_score.append(info["score"])
                ep_held.append(info["held"])
                ep_tip_height.append(info["tipHeight"])
                ep_upright_error.append(info["maxUprightError"])
                ep_max_held_seconds = max(ep_max_held_seconds, info["maxHeldSeconds"])
                if terminated or truncated:
                    break
            total_score.append(ep_score[-1] if ep_score else 0.0)
            total_held.append(float(np.mean(ep_held)) if ep_held else 0.0)
            total_max_held_seconds.append(ep_max_held_seconds)
            total_tip_height.append(float(np.mean(ep_tip_height)) if ep_tip_height else 0.0)
            total_upright_error.append(float(np.min(ep_upright_error)) if ep_upright_error else math.pi)
            returns.append(ep_return)
        return {
            "pose": pose_mode,
            "episodes": episodes,
            "score": float(np.mean(total_score)),
            "scoreP10": float(np.quantile(total_score, 0.1)),
            "held": float(np.mean(total_held)),
            "heldP10": float(np.quantile(total_held, 0.1)),
            "maxHeldSeconds": float(np.mean(total_max_held_seconds)),
            "maxHeldSecondsP10": float(np.quantile(total_max_held_seconds, 0.1)),
            "solvedOneSecondRate": float(np.mean(np.asarray(total_max_held_seconds) >= 1.0)),
            "tipHeight": float(np.mean(total_tip_height)),
            "bestUprightError": float(np.mean(total_upright_error)),
            "return": float(np.mean(returns)),
        }

    def validate_all(model, episodes: int = 24):
        return {
            "down": validate(model, episodes, "down"),
            "hold": validate(model, episodes, "hold"),
            "mixed": validate(model, episodes, "mixed"),
        }

    env_count = 8 if smoke else 16
    vec_env = DummyVecEnv([make_env(index, pose_mode="down") for index in range(env_count)])
    common_kwargs = {
        "policy": "MlpPolicy",
        "env": vec_env,
        "seed": seed + safe_links,
        "learning_rate": 3e-4,
        "buffer_size": 180_000 if smoke else 700_000,
        "learning_starts": 2_000,
        "batch_size": 256,
        "tau": 0.02,
        "gamma": 0.99,
        "train_freq": (1, "step"),
        "gradient_steps": 1,
        "policy_kwargs": {"net_arch": [128, 128], "activation_fn": nn.ReLU},
        "verbose": 0,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    if safe_algo == "td3":
        action_noise = NormalActionNoise(mean=np.zeros(1), sigma=np.ones(1) * 5.0)
        model = TD3(**common_kwargs, action_noise=action_noise, policy_delay=2, target_policy_noise=2.0, target_noise_clip=6.0)
    else:
        model = SAC(**common_kwargs, ent_coef="auto")

    if safe_curriculum == "hold":
        phases = [("hold", total_timesteps)]
    elif safe_curriculum == "stabilize-down":
        phases = [("hold", 40_000), ("mixed", 40_000), ("down", 80_000)] if smoke else [
            ("hold", 120_000),
            ("mixed", 160_000),
            ("down", 320_000),
        ]
    else:
        phases = [("down", total_timesteps)]

    history = []
    best_score = -1.0
    best_state = None
    completed = 0
    for phase_name, phase_steps in phases:
        phase_env = DummyVecEnv([make_env(index, pose_mode=phase_name) for index in range(env_count)])
        model.set_env(phase_env)
        phase_completed = 0
        while phase_completed < phase_steps:
            chunk = min(eval_every, phase_steps - phase_completed)
            model.learn(total_timesteps=chunk, reset_num_timesteps=completed == 0, progress_bar=False)
            completed += chunk
            phase_completed += chunk
            all_validation = validate_all(model, episodes=12 if smoke else 24)
            validation = all_validation["down"]
            phase_validation = all_validation[phase_name] if phase_name in all_validation else validation
            selection = (
                validation["solvedOneSecondRate"] * 1000.0
                + validation["maxHeldSeconds"] * 100.0
                + validation["score"]
                + validation["tipHeight"] * 10.0
                + phase_validation["maxHeldSeconds"] * 25.0
            )
            line = {
                "phase": phase_name,
                "timesteps": completed,
                "selection": selection,
                "down": all_validation["down"],
                "hold": all_validation["hold"],
                "mixed": all_validation["mixed"],
            }
            print(json.dumps(line), flush=True)
            history.append(line)
            if selection > best_score:
                best_score = selection
                best_state = {key: value.detach().cpu().clone() for key, value in model.policy.state_dict().items()}

    if best_state is not None:
        model.policy.load_state_dict({key: value.to(model.device) for key, value in best_state.items()})

    final_validation = validate_all(model, episodes=32 if smoke else 96)

    def export_linear_layers():
        actor = model.policy.actor
        modules = []

        def add_linear(module, activation):
            modules.append({
                "weights": module.weight.detach().cpu().numpy().T.tolist(),
                "bias": module.bias.detach().cpu().numpy().tolist(),
                "activation": activation,
            })

        if hasattr(actor, "latent_pi"):
            for module in actor.latent_pi:
                if isinstance(module, nn.Linear):
                    add_linear(module, "relu")
            add_linear(actor.mu, "tanh")
            return modules

        sequential = actor.mu if hasattr(actor, "mu") else actor
        pending_linear = None
        for module in sequential:
            if isinstance(module, nn.Linear):
                if pending_linear is not None:
                    add_linear(pending_linear, "linear")
                pending_linear = module
            elif isinstance(module, nn.ReLU) and pending_linear is not None:
                add_linear(pending_linear, "relu")
                pending_linear = None
            elif isinstance(module, nn.Tanh) and pending_linear is not None:
                add_linear(pending_linear, "tanh")
                pending_linear = None
        if pending_linear is not None:
            add_linear(pending_linear, "linear")
        return modules

    output = {
        "trainedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "algorithm": f"modal-mujoco-{safe_algo}",
        "environment": f"mujoco-cartpole-{safe_links}-link-down-start",
        "seed": seed + safe_links,
        "links": safe_links,
        "modelType": "sacMlp",
        "inputCount": obs_dim,
        "forceScale": action_scale,
        "layers": export_linear_layers(),
        "training": {
            "device": str(model.device),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "torch": torch.__version__,
            "mujoco": mujoco.__version__,
            "stableBaselines3": "2.6.0",
            "algo": safe_algo,
            "elapsedSeconds": round(time.time() - started, 3),
            "smoke": smoke,
            "totalTimesteps": total_timesteps,
            "phaseTimesteps": [{"phase": phase, "timesteps": steps} for phase, steps in phases],
            "horizon": horizon,
            "controlSkip": control_skip,
            "controlDt": control_dt,
            "curriculum": safe_curriculum,
            "history": history,
            "validation": final_validation["down"],
            "validationByPose": final_validation,
            "strictScore": {
                "maxUprightAngleRad": score_max_upright_angle,
                "maxChainBendRad": score_max_chain_bend,
                "minimumConsecutiveHoldSeconds": 1.0,
            },
            "sourceClues": [
                "Gymnasium Pendulum uses angle, angular velocity, and torque shaping for swing-up.",
                "Gymnasium MuJoCo InvertedPendulum uses an alive reward only while upright.",
                "Stable-Baselines3 SAC is a continuous-control baseline for Pendulum-style tasks.",
            ],
        },
    }
    return json.dumps(output, indent=2)


@app.local_entrypoint()
def main(links: int = 1, smoke: bool = True, curriculum: str = "down", algo: str = "sac"):
    if links != 1:
        raise ValueError("SAC gate starts with exactly one pendulum. Solve link 1 before links 2-6.")
    path = Path("app/ailab/six-pendulum-cartpole/mjcf/cartpole_1_link.xml")
    if not path.exists():
        raise FileNotFoundError(f"Missing MJCF file: {path}")
    return train_policy.remote(path.read_text(), links, smoke, 426210, curriculum, algo)
