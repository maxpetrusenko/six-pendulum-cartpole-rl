import json
import math
import time
from pathlib import Path

import modal


app = modal.App("six-pendulum-pezzza-evolution-train")
image = modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.7.1", "numpy==2.2.6")


@app.function(image=image, gpu="L4", timeout=7200)
def train_policy(
    smoke: bool = True,
    seed: int = 426310,
    control_hz: int = 240,
    population: int = 1536,
    generations: int = 32,
    profile: str = "curriculum",
) -> str:
    import torch

    started = time.time()
    safe_control_hz = max(60, min(480, int(control_hz)))
    safe_population = max(256, min(8192, int(population)))
    safe_generations = max(4, min(240, int(generations)))
    safe_profile = profile if profile in {"curriculum", "full-gravity", "velocity"} else "curriculum"

    torch.manual_seed(seed + safe_control_hz + safe_population + safe_generations)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dt = 1.0 / float(safe_control_hz)
    horizon_seconds = 8.0 if smoke else 16.0
    steps = int(horizon_seconds * safe_control_hz)
    hidden = 10
    knot_count = 24
    input_count = 7
    output_count = 1
    param_count = knot_count + input_count * hidden + hidden + hidden * output_count + output_count
    elite_count = max(32, min(384, safe_population // 16))
    action_scale = 38.0 if safe_profile != "velocity" else 5.5
    track_limit = 2.4
    hold_angle = 0.16
    near_top_angle = 0.65
    max_hold_seconds = 2.0

    mean = torch.zeros(param_count, device=device)
    sigma = torch.tensor(1.15, device=device)
    best_params = mean.clone()
    best_selection = torch.tensor(-1e9, device=device)

    def unpack(params):
        cursor = 0
        knots = params[:, cursor : cursor + knot_count]
        cursor += knot_count
        w1 = params[:, cursor : cursor + input_count * hidden].view(-1, input_count, hidden)
        cursor += input_count * hidden
        b1 = params[:, cursor : cursor + hidden]
        cursor += hidden
        w2 = params[:, cursor : cursor + hidden * output_count].view(-1, hidden, output_count)
        cursor += hidden * output_count
        b2 = params[:, cursor : cursor + output_count]
        return knots, w1, b1, w2, b2

    def policy(params, x, xdot, theta, thetadot, last_action, tick):
        knots, w1, b1, w2, b2 = unpack(params)
        t_norm = torch.full_like(x, float(tick) / max(steps - 1, 1))
        knot_pos = t_norm * float(knot_count - 1)
        left = torch.floor(knot_pos).long().clamp(0, knot_count - 1)
        right = (left + 1).clamp(0, knot_count - 1)
        mix = knot_pos - left.float()
        base = knots.gather(1, left.view(-1, 1)).squeeze(1) * (1.0 - mix) + knots.gather(1, right.view(-1, 1)).squeeze(1) * mix
        obs = torch.stack(
            [
                x / track_limit,
                xdot / 6.0,
                torch.sin(theta),
                torch.cos(theta),
                thetadot / 10.0,
                last_action / action_scale,
                t_norm,
            ],
            dim=1,
        )
        h = torch.tanh(torch.bmm(obs.unsqueeze(1), w1).squeeze(1) + b1)
        feedback = torch.bmm(h.unsqueeze(1), w2).squeeze(1).squeeze(1) + b2.squeeze(1)
        return torch.tanh(base + feedback) * action_scale

    def initial_state(batch):
        x = torch.randn(batch, device=device) * 0.025
        xdot = torch.randn(batch, device=device) * 0.04
        theta = math.pi + torch.randn(batch, device=device) * 0.035
        theta = torch.remainder(theta + math.pi, 2 * math.pi) - math.pi
        thetadot = torch.randn(batch, device=device) * 0.045
        return x, xdot, theta, thetadot

    def step_force(x, xdot, theta, thetadot, action, gravity, cart_damping, hinge_damping):
        cart_mass = 1.0
        pole_mass = 0.1
        length = 0.5
        total_mass = cart_mass + pole_mass
        force = action - cart_damping * xdot - 0.35 * x
        temp = (force + pole_mass * length * thetadot.square() * torch.sin(theta)) / total_mass
        theta_acc = (gravity * torch.sin(theta) - torch.cos(theta) * temp) / (
            length * (4.0 / 3.0 - pole_mass * torch.cos(theta).square() / total_mass)
        )
        theta_acc = theta_acc - hinge_damping * thetadot
        x_acc = temp - pole_mass * length * theta_acc * torch.cos(theta) / total_mass
        xdot = torch.clamp(xdot + x_acc * dt, -8.0, 8.0)
        x = torch.clamp(x + xdot * dt, -track_limit * 1.2, track_limit * 1.2)
        thetadot = torch.clamp(thetadot + theta_acc * dt, -22.0, 22.0)
        theta = torch.remainder(theta + thetadot * dt + math.pi, 2 * math.pi) - math.pi
        return x, xdot, theta, thetadot

    def step_velocity(x, xdot, theta, thetadot, action, gravity, cart_damping, hinge_damping):
        target_v = action
        xdot = target_v - 0.06 * x
        x = torch.clamp(x + xdot * dt, -track_limit * 1.2, track_limit * 1.2)
        theta_acc = gravity * torch.sin(theta) / 0.5 - 1.8 * xdot * torch.cos(theta) - hinge_damping * thetadot
        thetadot = torch.clamp(thetadot + theta_acc * dt, -22.0, 22.0)
        theta = torch.remainder(theta + thetadot * dt + math.pi, 2 * math.pi) - math.pi
        return x, xdot, theta, thetadot

    def evaluate(params, stage, validation=False):
        batch = params.shape[0]
        gravity = stage["gravity"]
        cart_damping = stage["cartDamping"]
        hinge_damping = stage["hingeDamping"]
        if stage.get("randomHorizon", False) and not validation:
            horizon_steps = torch.randint(int(steps * 0.65), steps + 1, (batch,), device=device)
        else:
            horizon_steps = torch.full((batch,), steps, device=device, dtype=torch.long)
        x, xdot, theta, thetadot = initial_state(batch)
        reward = torch.zeros(batch, device=device)
        hold_run = torch.zeros(batch, device=device)
        max_hold = torch.zeros(batch, device=device)
        above_time = torch.zeros(batch, device=device)
        whiplash = torch.zeros(batch, device=device)
        center_bonus = torch.zeros(batch, device=device)
        last_action = torch.zeros(batch, device=device)
        last_hold_run = torch.zeros(batch, device=device)
        action_delta_sum = torch.zeros(batch, device=device)

        for tick in range(steps):
            active = (tick < horizon_steps).float()
            action = policy(params, x, xdot, theta, thetadot, last_action, tick) * active
            if safe_profile == "velocity":
                x, xdot, theta, thetadot = step_velocity(x, xdot, theta, thetadot, action, gravity, cart_damping, hinge_damping)
            else:
                x, xdot, theta, thetadot = step_force(x, xdot, theta, thetadot, action, gravity, cart_damping, hinge_damping)

            upright_error = theta.abs()
            strict = (upright_error < hold_angle) & (x.abs() < 1.2) & (thetadot.abs() < 4.5)
            near_top = upright_error < near_top_angle
            swing_with_gravity = torch.relu(torch.cos(theta) - 0.35) * torch.relu(thetadot.abs() - 0.8)
            whiplash_event = near_top.float() * (thetadot.abs() > 1.2).float() * (x.abs() < 1.55).float() * active

            hold_run = torch.where(strict & active.bool(), hold_run + dt, torch.zeros_like(hold_run))
            max_hold = torch.maximum(max_hold, hold_run)
            above_time = torch.where(near_top & active.bool(), above_time + dt, torch.zeros_like(above_time))
            whiplash += whiplash_event * dt
            center_bonus += torch.relu(1.0 - x.abs() / track_limit) * dt * active

            loss_of_balance = ((last_hold_run > 0.12) & (~strict) & (upright_error < 0.38)).float()
            reward += torch.exp(-upright_error * 1.8 - thetadot.abs() * 0.04) * 0.025 * active
            reward += swing_with_gravity * (0.018 if tick < steps * 0.65 else 0.006) * active
            reward += whiplash_event * (0.08 if tick < steps * 0.65 else 0.02)
            reward += strict.float() * (0.5 + hold_run.pow(2.0) * 2.5) * active
            reward += (above_time.clamp(max=max_hold_seconds).pow(8) / max_hold_seconds**8) * 2.0 * active
            reward -= x.abs() * 0.006 * active
            reward -= (action / action_scale).square() * 0.002 * active
            reward -= loss_of_balance * 0.25 * active
            action_delta_sum += (action - last_action).abs() * active
            last_action = action
            last_hold_run = hold_run

        smooth_penalty = action_delta_sum / max(steps, 1) / action_scale
        mean_horizon_seconds = horizon_steps.float().mean() * dt
        selection = (
            reward
            + max_hold * 145.0
            + (max_hold >= 1.0).float() * 420.0
            + whiplash.clamp(max=1.5) * 12.0
            + center_bonus / horizon_seconds * 3.0
            - smooth_penalty * 4.0
        )
        if validation:
            return {
                "reward": reward,
                "selection": selection,
                "maxHoldSeconds": max_hold,
                "solvedOneSecond": (max_hold >= 1.0).float(),
                "whiplashSeconds": whiplash,
                "centerRatio": center_bonus / horizon_seconds,
                "smoothPenalty": smooth_penalty,
                "horizonSeconds": mean_horizon_seconds,
            }
        return selection

    if safe_profile == "full-gravity":
        stages = [
            {"name": "full-gravity", "gravity": 9.81, "cartDamping": 0.08, "hingeDamping": 0.02, "generations": safe_generations, "randomHorizon": True}
        ]
    elif safe_profile == "velocity":
        stages = [
            {"name": "velocity-easy", "gravity": 3.0, "cartDamping": 0.18, "hingeDamping": 0.10, "generations": max(4, safe_generations // 3), "randomHorizon": False},
            {"name": "velocity-normal", "gravity": 9.81, "cartDamping": 0.08, "hingeDamping": 0.02, "generations": safe_generations, "randomHorizon": True},
        ]
    else:
        stages = [
            {"name": "low-gravity-high-friction", "gravity": 2.2, "cartDamping": 0.26, "hingeDamping": 0.22, "generations": max(4, safe_generations // 4), "randomHorizon": False},
            {"name": "medium-gravity", "gravity": 4.8, "cartDamping": 0.18, "hingeDamping": 0.12, "generations": max(4, safe_generations // 4), "randomHorizon": False},
            {"name": "near-normal", "gravity": 7.2, "cartDamping": 0.12, "hingeDamping": 0.06, "generations": max(4, safe_generations // 4), "randomHorizon": False},
            {"name": "normal", "gravity": 9.81, "cartDamping": 0.08, "hingeDamping": 0.02, "generations": safe_generations, "randomHorizon": True},
        ]

    def summarize_metrics(metrics):
        max_hold = float(metrics["maxHoldSeconds"].mean().detach().cpu())
        max_hold_p10 = float(torch.quantile(metrics["maxHoldSeconds"], 0.1).detach().cpu())
        solved_rate = float(metrics["solvedOneSecond"].mean().detach().cpu())
        center_ratio = float(metrics["centerRatio"].mean().detach().cpu())
        smooth_penalty = float(metrics["smoothPenalty"].mean().detach().cpu())
        strict_score = 0.0
        if max_hold >= 1.0:
            rail_penalty = max(0.0, 0.62 - center_ratio) * 10.0
            strict_score = (
                100.0 * solved_rate
                + 10.0 * min(max_hold, horizon_seconds)
                + 5.0 * min(max_hold_p10, horizon_seconds)
                - rail_penalty
                - smooth_penalty
            )
        return {
            "strictScore": strict_score,
            "maxHoldSeconds": max_hold,
            "maxHoldSecondsP10": max_hold_p10,
            "solvedOneSecondRate": solved_rate,
            "whiplashSeconds": float(metrics["whiplashSeconds"].mean().detach().cpu()),
            "centerRatio": center_ratio,
            "smoothPenalty": smooth_penalty,
            "selection": float(metrics["selection"].mean().detach().cpu()),
            "horizonSeconds": float(metrics["horizonSeconds"].detach().cpu()),
        }

    history = []
    experiment_dots = []
    randomized_horizon_ready = False
    for stage in stages:
        for generation in range(stage["generations"]):
            stage_for_eval = {**stage, "randomHorizon": bool(stage.get("randomHorizon", False) and randomized_horizon_ready)}
            params = mean.view(1, -1) + torch.randn(safe_population, param_count, device=device) * sigma
            selection = evaluate(params, stage_for_eval)
            values, indices = torch.topk(selection, elite_count)
            elites = params[indices]
            mean = elites.mean(dim=0)
            sigma = torch.maximum(sigma * 0.965, torch.tensor(0.045, device=device))
            if values[0] > best_selection:
                best_selection = values[0]
                best_params = elites[0].clone()
            if generation % 4 == 0 or generation == stage["generations"] - 1:
                eval_params = best_params.view(1, -1).repeat(96 if smoke else 192, 1)
                metrics = evaluate(eval_params, stage, validation=True)
                summary = summarize_metrics(metrics)
                if summary["whiplashSeconds"] >= 0.12 and summary["solvedOneSecondRate"] < 0.5:
                    randomized_horizon_ready = True
                line = {
                    "stage": stage["name"],
                    "gravity": stage["gravity"],
                    "randomizedHorizon": stage_for_eval["randomHorizon"],
                    "generation": generation,
                    "sigma": float(sigma.detach().cpu()),
                    "bestSelection": float(best_selection.detach().cpu()),
                    "eliteSelection": float(values[0].detach().cpu()),
                    "strictScore": summary["strictScore"],
                    "maxHoldSeconds": summary["maxHoldSeconds"],
                    "maxHoldSecondsP10": summary["maxHoldSecondsP10"],
                    "solvedOneSecondRate": summary["solvedOneSecondRate"],
                    "whiplashSeconds": summary["whiplashSeconds"],
                    "centerRatio": summary["centerRatio"],
                    "smoothPenalty": summary["smoothPenalty"],
                    "horizonSeconds": summary["horizonSeconds"],
                    "wallclockSeconds": round(time.time() - started, 3),
                    "score": summary["strictScore"],
                }
                print(json.dumps(line), flush=True)
                history.append(line)
                experiment_dots.append(
                    {
                        "runId": f"{seed}_{safe_profile}_{safe_control_hz}_{safe_population}_{safe_generations}_{stage['name']}_{generation}",
                        "algorithm": "modal-pezzza-style-evolution",
                        "modelType": "mlpPolicy",
                        "links": 1,
                        "stage": stage["name"],
                        "seed": seed,
                        "wallclockSeconds": line["wallclockSeconds"],
                        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
                        "train": {
                            "population": safe_population,
                            "eliteCount": elite_count,
                            "generations": safe_generations,
                            "controlHz": safe_control_hz,
                            "horizonSeconds": horizon_seconds,
                            "randomizedHorizon": stage_for_eval["randomHorizon"],
                        },
                        "hypers": {
                            "forceScale": action_scale,
                            "sigma": line["sigma"],
                            "knotCount": knot_count,
                            "hidden": hidden,
                            "profile": safe_profile,
                        },
                        "metrics": {
                            "strictScore": summary["strictScore"],
                            "maxHoldSeconds": summary["maxHoldSeconds"],
                            "maxHoldSecondsP10": summary["maxHoldSecondsP10"],
                            "solvedOneSecondRate": summary["solvedOneSecondRate"],
                            "whiplashSeconds": summary["whiplashSeconds"],
                            "centerRatio": summary["centerRatio"],
                            "selection": summary["selection"],
                        },
                    }
                )
        mean = best_params * 0.70 + mean * 0.30

    final_stage = {"name": "validation-normal-down", "gravity": 9.81, "cartDamping": 0.08, "hingeDamping": 0.02, "generations": 0, "randomHorizon": False}
    final_params = best_params.view(1, -1).repeat(256 if smoke else 1024, 1)
    final_metrics = evaluate(final_params, final_stage, validation=True)
    final_summary = summarize_metrics(final_metrics)

    knots, w1, b1, w2, b2 = unpack(best_params.view(1, -1))
    output = {
        "trainedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "policyVersion": 1,
        "algorithm": "modal-pezzza-style-evolution",
        "environment": "one-link-vectorized-cartpole-down-start",
        "modelType": "pezzzaKnotMlp",
        "links": 1,
        "inputCount": input_count,
        "knotCount": knot_count,
        "forceScale": action_scale,
        "controlHz": safe_control_hz,
        "dt": dt,
        "horizonSeconds": horizon_seconds,
        "observation": [
            "cartX/2.4",
            "cartV/6",
            "sin(theta0)",
            "cos(theta0)",
            "omega0/10",
            "lastAction/forceScale",
            "time/horizonSeconds",
        ],
        "knots": knots.squeeze(0).detach().cpu().tolist(),
        "layers": [
            {"weights": w1.squeeze(0).detach().cpu().tolist(), "bias": b1.squeeze(0).detach().cpu().tolist(), "activation": "tanh"},
            {"weights": w2.squeeze(0).detach().cpu().tolist(), "bias": b2.squeeze(0).detach().cpu().tolist(), "activation": "tanh"},
        ],
        "training": {
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "torch": torch.__version__,
            "elapsedSeconds": round(time.time() - started, 3),
            "smoke": smoke,
            "seed": seed,
            "profile": safe_profile,
            "population": safe_population,
            "eliteCount": elite_count,
            "generations": safe_generations,
            "controlHz": safe_control_hz,
            "dt": dt,
            "horizonSeconds": horizon_seconds,
            "score": "strictScore is zero unless mean maxHoldSeconds is at least 1.0; subsecond holds are diagnostic only",
            "movementTerms": {
                "compensation": "cart acceleration and velocity are policy inputs/effects",
                "lossOfBalance": "penalty when a recent hold collapses near threshold",
                "usingGravity": "early swing energy term from cos(theta) and angular speed",
                "whiplash": "near-top high-angular-speed event before stabilization",
                "stayCenter": "centerRatio reward plus rail penalties",
            },
            "stages": stages,
            "history": history,
            "experimentDots": experiment_dots,
            "validation": {
                "strictScore": final_summary["strictScore"],
                "maxHoldSeconds": final_summary["maxHoldSeconds"],
                "maxHoldSecondsP10": final_summary["maxHoldSecondsP10"],
                "solvedOneSecondRate": final_summary["solvedOneSecondRate"],
                "whiplashSeconds": final_summary["whiplashSeconds"],
                "centerRatio": final_summary["centerRatio"],
                "smoothPenalty": final_summary["smoothPenalty"],
                "selection": final_summary["selection"],
            },
            "sourceClues": [
                "Pendulum-NEAT: 1000-agent evolutionary population with one cart-control output.",
                "Pezzza video: reward time above threshold, then curriculum low gravity/high friction to normal gravity.",
                "Pezzza video: acceleration control adds cart velocity input and benefits from higher control frequency.",
            ],
        },
    }
    return json.dumps(output, indent=2)


@app.local_entrypoint()
def main(smoke: bool = True, control_hz: int = 240, population: int = 1536, generations: int = 32, profile: str = "curriculum"):
    return train_policy.remote(smoke, 426310, control_hz, population, generations, profile)
