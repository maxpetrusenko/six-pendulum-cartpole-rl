import json
import time

import modal


app = modal.App("six-pendulum-curriculum-train")
image = modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.7.1", "numpy==2.2.6")


@app.function(image=image, gpu="L4", timeout=7200)
def train_policy(
    initial_policy_json: str = "",
    smoke: bool = False,
    max_stage_links: int = 6,
    generation_scale: float = 1.0,
) -> str:
    import torch

    started = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_links = 6
    safe_max_stage_links = max(1, min(max_links, int(max_stage_links)))
    safe_generation_scale = max(0.25, float(generation_scale))
    torch.manual_seed((426120 if not smoke else 426121) + safe_max_stage_links)
    torch.set_float32_matmul_precision("high")

    dt = 0.025
    steps = 520 if smoke else 820
    knot_count = 48 if smoke else 72
    feedback_count = 2 + max_links * 2
    param_count = knot_count + feedback_count
    force_scale = 32.0
    score_max_upright_angle = 0.16
    score_max_chain_bend = 0.14
    population = 1536 if smoke else 8192
    elite_count = 96 if smoke else 192
    checkpoint_pool_count = 24 if smoke else 32
    transition_blend = 0.72
    sigma = 0.95 if initial_policy_json else 1.45
    mean = torch.zeros(param_count, device=device)
    best_params = mean.clone()
    best_fitness = torch.tensor(-1e9, device=device)
    link_index = torch.arange(max_links, device=device).float().view(1, max_links)
    link_weights = torch.arange(1, max_links + 1, device=device).float().view(1, max_links)

    if initial_policy_json:
        initial = json.loads(initial_policy_json)
        if initial.get("modelType") == "timeKnotFeedback":
            with torch.no_grad():
                knots = torch.tensor(initial.get("knots", []), device=device).float()
                feedback = torch.tensor(initial.get("feedback", []), device=device).float()
                mean[: min(knot_count, knots.numel())] = knots[:knot_count]
                mean[knot_count : knot_count + min(feedback_count, feedback.numel())] = feedback[:feedback_count]

    base_stages = [
        {"phase": "hold", "links": 1, "generations": 6 if smoke else 24, "pose": "hold", "random_horizon": False},
        {"phase": "swing", "links": 1, "generations": 8 if smoke else 36, "pose": "mixed", "random_horizon": False},
        {"phase": "hold", "links": 2, "generations": 8 if smoke else 36, "pose": "hold", "random_horizon": False},
        {"phase": "swing", "links": 2, "generations": 8 if smoke else 42, "pose": "mixed", "random_horizon": smoke},
        {"phase": "hold", "links": 3, "generations": 0 if smoke else 36, "pose": "hold", "random_horizon": False},
        {"phase": "swing", "links": 3, "generations": 0 if smoke else 48, "pose": "mixed", "random_horizon": True},
        {"phase": "hold", "links": 4, "generations": 0 if smoke else 40, "pose": "hold", "random_horizon": False},
        {"phase": "swing", "links": 4, "generations": 0 if smoke else 54, "pose": "down", "random_horizon": True},
        {"phase": "hold", "links": 5, "generations": 0 if smoke else 44, "pose": "hold", "random_horizon": False},
        {"phase": "swing", "links": 5, "generations": 0 if smoke else 60, "pose": "down", "random_horizon": True},
        {"phase": "hold", "links": 6, "generations": 0 if smoke else 48, "pose": "hold", "random_horizon": False},
        {"phase": "swing", "links": 6, "generations": 0 if smoke else 72, "pose": "down", "random_horizon": True},
    ]
    stages = []
    for stage in base_stages:
        if stage["links"] > safe_max_stage_links:
            continue
        stages.append(
            {
                **stage,
                "generations": int(round(stage["generations"] * safe_generation_scale)),
            }
        )

    def active_mask(active_links):
        return (torch.arange(max_links, device=device).view(1, max_links) < active_links).float()

    def make_initial(batch, active_links, pose):
        mask = active_mask(active_links)
        down = torch.pi - link_index * 0.08
        upright = -0.08 + link_index * 0.018
        if pose == "hold":
            theta = upright.repeat(batch, 1)
        elif pose == "mixed":
            selector = (torch.rand(batch, 1, device=device) > 0.42).float()
            theta = down * selector + upright * (1.0 - selector)
        else:
            theta = down.repeat(batch, 1)
        theta = theta + torch.randn(batch, max_links, device=device) * (0.035 + active_links * 0.004)
        omega = torch.randn(batch, max_links, device=device) * 0.045
        cart_x = torch.randn(batch, device=device) * 0.012
        cart_v = torch.randn(batch, device=device) * 0.035
        return cart_x, cart_v, theta * mask, omega * mask

    def policy_force(params, cart_x, cart_v, theta, omega, tick, horizon):
        knots = params[:, :knot_count]
        feedback = params[:, knot_count:]
        position = tick / torch.clamp(horizon - 1, min=1) * (knot_count - 1)
        left = torch.floor(position).long().clamp(0, knot_count - 1)
        right = (left + 1).clamp(0, knot_count - 1)
        mix = position - left.float()
        base = knots.gather(1, left.view(-1, 1)).squeeze(1) * (1 - mix) + knots.gather(1, right.view(-1, 1)).squeeze(1) * mix
        features = [cart_x, cart_v / 5.0]
        for link in range(max_links):
            features.extend([torch.sin(theta[:, link]), omega[:, link] / 8.0])
        obs = torch.stack(features, dim=1)
        correction = (feedback * obs).sum(dim=1)
        return torch.tanh(base + correction) * force_scale

    def strict_score(cart_x, theta, omega, active_links):
        active_theta = theta[:, :active_links]
        active_omega = omega[:, :active_links]
        weights = link_weights[:, :active_links]
        max_upright_error = active_theta.abs().max(dim=1).values
        if active_links > 1:
            max_bend_error = (active_theta[:, 1:] - active_theta[:, :-1]).abs().max(dim=1).values
        else:
            max_bend_error = torch.zeros_like(max_upright_error)
        strict_gate = (max_upright_error <= score_max_upright_angle) & (max_bend_error <= score_max_chain_bend)
        angle_error = (active_theta.abs() * weights).sum(dim=1)
        velocity_error = active_omega.abs().sum(dim=1)
        raw_score = torch.clamp(
            100.0 - angle_error * 12.0 - max_bend_error * 30.0 - velocity_error * 2.5 - cart_x.abs() * 8.0,
            0.0,
            100.0,
        )
        return torch.where(strict_gate, raw_score, torch.zeros_like(raw_score))

    def step(cart_x, cart_v, theta, omega, force, active_links):
        mask = active_mask(active_links)
        link_drag = (torch.sin(theta) * link_weights * mask).sum(dim=1) * 0.08
        cart_acc = force - 0.62 * cart_v - 1.2 * cart_x - link_drag
        cart_v = torch.clamp(cart_v + cart_acc * dt, -5.5, 5.5)
        cart_x = torch.clamp(cart_x + cart_v * dt, -2.4, 2.4)
        prev = torch.cat([theta[:, :1], theta[:, :-1]], dim=1)
        nxt = torch.cat([theta[:, 1:], theta[:, -1:]], dim=1)
        length = 0.62 + link_index * 0.035
        coupling = (prev + nxt - theta * 2.0) * (1.5 + link_index * 0.2)
        damping = 0.045 + link_index * 0.008
        gravity = 9.81 * torch.sin(theta) / length
        drive = (-cart_acc.view(-1, 1) * torch.cos(theta) * (0.42 + link_index * 0.04)) / length
        angular_acc = (gravity + drive + coupling - damping * omega) * mask
        omega = torch.clamp(omega + angular_acc * dt, -18.0, 18.0) * mask
        theta = (torch.remainder(theta + omega * dt + torch.pi, 2 * torch.pi) - torch.pi) * mask
        return cart_x, cart_v, theta, omega

    def evaluate(params, stage, validation=False):
        active_links = stage["links"]
        batch = params.shape[0]
        cart_x, cart_v, theta, omega = make_initial(batch, active_links, stage["pose"])
        if stage["random_horizon"] and not validation:
            horizon = torch.randint(int(steps * 0.62), steps + 1, (batch,), device=device).float()
        else:
            horizon = torch.full((batch,), float(steps), device=device)
        reward = torch.zeros(batch, device=device)
        held = torch.zeros(batch, device=device)
        whip = torch.zeros(batch, device=device)
        last = torch.zeros(batch, device=device)
        alive_steps = torch.zeros(batch, device=device)
        for tick in range(steps):
            active = (tick < horizon).float()
            force = policy_force(params, cart_x, cart_v, theta, omega, tick, horizon) * active
            cart_x, cart_v, theta, omega = step(cart_x, cart_v, theta, omega, force, active_links)
            score = strict_score(cart_x, theta, omega, active_links) * active
            active_theta = theta[:, :active_links]
            active_omega = omega[:, :active_links]
            if active_links > 1:
                max_bend_error = (active_theta[:, 1:] - active_theta[:, :-1]).abs().max(dim=1).values
            else:
                max_bend_error = torch.zeros(batch, device=device)
            max_upright_error = active_theta.abs().max(dim=1).values
            mean_upright_error = active_theta.abs().mean(dim=1)
            mean_speed = active_omega.abs().mean(dim=1)
            dense_alignment = torch.exp(-mean_upright_error * 1.25 - max_bend_error * 2.2 - mean_speed * 0.08)
            swing_window = 1.0 if tick < int(steps * 0.58) else 0.25
            late = tick / max(steps - 1, 1)
            whip_event = ((active_theta[:, -1].abs() < 0.65) & (mean_speed > 1.2)).float() * active
            reward += dense_alignment * active * (0.04 + late * 0.12)
            reward += whip_event * swing_window * 0.10
            reward += (score / 100.0).pow(2) * (0.4 + late * 1.8)
            reward += (score > 82).float() * (0.8 + late * 3.2)
            reward += (score > 92).float() * (2.0 + late * 3.0)
            reward -= (cart_x.abs() * 0.025 + (force / force_scale).square() * 0.002) * active
            held += (score > 82).float()
            whip += whip_event
            alive_steps += active
            last = torch.where(active.bool(), score, last)
        reward += (last / 100.0).pow(4) * 180.0
        reward += held * (0.45 + active_links * 0.08)
        reward += whip.clamp(max=12) * 0.08
        reward -= params[:, :knot_count].diff(dim=1).abs().mean(dim=1) * 0.15
        return reward, last, held / torch.clamp(alive_steps, min=1), whip / torch.clamp(alive_steps, min=1)

    history = []
    validation_by_stage = []
    for stage in stages:
        if stage["generations"] <= 0:
            continue
        stage_best = None
        stage_best_fitness = torch.tensor(-1e9, device=device)
        for generation in range(stage["generations"]):
            params = mean.view(1, -1) + torch.randn(population, param_count, device=device) * sigma
            reward, last, held, whip = evaluate(params, stage)
            fitness = reward + last * 2.0 + held * (90.0 + stage["links"] * 20.0) + whip * 20.0
            values, indices = torch.topk(fitness, elite_count)
            elites = params[indices]
            mean = elites.mean(dim=0)
            sigma = max(sigma * 0.972, 0.075)
            checkpoint_params = elites[: min(checkpoint_pool_count, elites.shape[0])]
            checkpoint_reward, checkpoint_last, checkpoint_held, checkpoint_whip = evaluate(
                checkpoint_params,
                stage,
                validation=True,
            )
            checkpoint_fitness = (
                checkpoint_last
                + checkpoint_held * (135.0 + stage["links"] * 30.0)
                + checkpoint_whip * 8.0
                + checkpoint_reward * 0.04
            )
            checkpoint_value, checkpoint_index = torch.max(checkpoint_fitness, dim=0)
            if checkpoint_value > stage_best_fitness:
                stage_best_fitness = checkpoint_value
                stage_best = checkpoint_params[checkpoint_index].clone()
            if values[0] > best_fitness:
                best_fitness = values[0]
            if generation % 8 == 0 or generation == stage["generations"] - 1:
                line = {
                    "phase": stage["phase"],
                    "stageLinks": stage["links"],
                    "generation": generation,
                    "bestFitness": float(values[0].detach().cpu()),
                    "eliteReward": float(reward[indices[0]].detach().cpu()),
                    "eliteScore": float(last[indices[0]].detach().cpu()),
                    "eliteHeld": float(held[indices[0]].detach().cpu()),
                    "eliteWhip": float(whip[indices[0]].detach().cpu()),
                    "checkpointFitness": float(checkpoint_value.detach().cpu()),
                    "checkpointScore": float(checkpoint_last[checkpoint_index].detach().cpu()),
                    "checkpointHeld": float(checkpoint_held[checkpoint_index].detach().cpu()),
                    "checkpointWhip": float(checkpoint_whip[checkpoint_index].detach().cpu()),
                    "randomizedEpisodeLength": bool(stage["random_horizon"]),
                    "sigma": float(sigma),
                }
                print(json.dumps(line), flush=True)
                history.append(line)
        if stage_best is not None:
            best_params = stage_best.clone()
            mean = stage_best * transition_blend + mean * (1.0 - transition_blend)
        validation_params = (stage_best if stage_best is not None else mean).view(1, -1).repeat(512 if smoke else 1024, 1)
        reward, last, held, whip = evaluate(validation_params, stage, validation=True)
        validation = {
            "phase": stage["phase"],
            "links": stage["links"],
            "reward": float(reward.mean().detach().cpu()),
            "score": float(last.mean().detach().cpu()),
            "held": float(held.mean().detach().cpu()),
            "whip": float(whip.mean().detach().cpu()),
            "scoreP10": float(torch.quantile(last, 0.1).detach().cpu()),
            "heldP10": float(torch.quantile(held, 0.1).detach().cpu()),
            "randomizedEpisodeLength": bool(stage["random_horizon"]),
        }
        print(json.dumps({"validation": validation}), flush=True)
        validation_by_stage.append(validation)

    final_stage = {
        "phase": "final-mixed",
        "links": validation_by_stage[-1]["links"] if validation_by_stage else 1,
        "pose": "mixed",
        "random_horizon": False,
    }
    validation_params = best_params.view(1, -1).repeat(1024 if not smoke else 512, 1)
    reward, last, held, whip = evaluate(validation_params, final_stage, validation=True)
    final_validation = {
        "links": final_stage["links"],
        "phase": final_stage["phase"],
        "reward": float(reward.mean().detach().cpu()),
        "score": float(last.mean().detach().cpu()),
        "held": float(held.mean().detach().cpu()),
        "whip": float(whip.mean().detach().cpu()),
        "scoreP10": float(torch.quantile(last, 0.1).detach().cpu()),
        "heldP10": float(torch.quantile(held, 0.1).detach().cpu()),
    }

    output = {
        "trainedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "algorithm": "modal-cross-entropy-curriculum",
        "environment": "six-link-browser-cartpole-curriculum-v1",
        "seed": (426120 if not smoke else 426121) + safe_max_stage_links,
        "modelType": "timeKnotFeedback",
        "knotCount": knot_count,
        "feedbackCount": feedback_count,
        "forceScale": force_scale,
        "steps": steps,
        "dt": dt,
        "training": {
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "torch": torch.__version__,
            "cudaRuntime": torch.version.cuda,
            "population": population,
            "eliteCount": elite_count,
            "checkpointPoolCount": checkpoint_pool_count,
            "checkpointSelection": "validation-aware-strict-score",
            "transitionBlend": transition_blend,
            "elapsedSeconds": round(time.time() - started, 3),
            "smoke": smoke,
            "maxStageLinks": safe_max_stage_links,
            "generationScale": safe_generation_scale,
            "curriculum": stages,
            "history": history,
            "validationByStage": validation_by_stage,
            "validation": final_validation,
            "warmStarted": bool(initial_policy_json),
            "strictScore": {
                "maxUprightAngleRad": score_max_upright_angle,
                "maxChainBendRad": score_max_chain_bend,
            },
            "sourceClues": [
                "Yacine thread: train lower link counts as an experiment, then scale to six.",
                "Yacine thread: add randomized episode length after whip behavior appears.",
                "Yacine thread: speed and many hyperparameter attempts matter more than one hand-tuned run.",
            ],
        },
        "knots": best_params[:knot_count].detach().cpu().tolist(),
        "feedback": best_params[knot_count:].detach().cpu().tolist(),
    }
    return json.dumps(output, indent=2)


@app.local_entrypoint()
def main(smoke: bool = False, max_stage_links: int = 6, generation_scale: float = 1.0):
    from pathlib import Path

    policy_path = Path("app/ailab/six-pendulum-cartpole/sixPendulumPolicy.json")
    initial = policy_path.read_text() if policy_path.exists() else ""
    return train_policy.remote(initial, smoke, max_stage_links, generation_scale)
