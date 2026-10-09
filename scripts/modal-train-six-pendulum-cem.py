import json
import time

import modal


app = modal.App("six-pendulum-cem-train")
image = modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.7.1", "numpy==2.2.6")


@app.function(image=image, gpu="L4", timeout=7200)
def train_policy(initial_policy_json: str = "") -> str:
    import torch

    started = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42612)
    torch.set_float32_matmul_precision("high")

    links = 6
    dt = 0.025
    steps = 760
    knot_count = 56
    feedback_count = 2 + links * 2
    param_count = knot_count + feedback_count
    force_scale = 32.0
    score_max_upright_angle = 0.16
    score_max_chain_bend = 0.14
    population = 8192
    elite_count = 192
    generations = 160
    sigma = 0.55 if initial_policy_json else 1.5
    mean = torch.zeros(param_count, device=device)
    best_score = torch.tensor(-1e9, device=device)
    best_params = mean.clone()
    link_weights = torch.arange(1, links + 1, device=device).float().view(1, links)
    link_index = torch.arange(links, device=device).float().view(1, links)

    if initial_policy_json:
        initial = json.loads(initial_policy_json)
        if initial.get("modelType") == "timeKnotFeedback":
            with torch.no_grad():
                knots = torch.tensor(initial.get("knots", []), device=device).float()
                feedback = torch.tensor(initial.get("feedback", []), device=device).float()
                mean[: min(knot_count, knots.numel())] = knots[:knot_count]
                mean[knot_count : knot_count + min(feedback_count, feedback.numel())] = feedback[:feedback_count]

    def make_initial(batch):
        theta = torch.pi - link_index * 0.08
        theta = theta.repeat(batch, 1) + torch.randn(batch, links, device=device) * 0.035
        omega = torch.randn(batch, links, device=device) * 0.035
        cart_x = torch.randn(batch, device=device) * 0.01
        cart_v = torch.randn(batch, device=device) * 0.03
        return cart_x, cart_v, theta, omega

    def policy_force(params, cart_x, cart_v, theta, omega, tick):
        batch = params.shape[0]
        knots = params[:, :knot_count]
        feedback = params[:, knot_count:]
        position = tick / max(steps - 1, 1) * (knot_count - 1)
        left = int(position)
        right = min(left + 1, knot_count - 1)
        mix = position - left
        base = knots[:, left] * (1 - mix) + knots[:, right] * mix
        features = [cart_x, cart_v / 5.0]
        for link in range(links):
            features.extend([torch.sin(theta[:, link]), omega[:, link] / 8.0])
        obs = torch.stack(features, dim=1)
        correction = (feedback * obs).sum(dim=1)
        return torch.tanh(base + correction) * force_scale

    def app_score(cart_x, theta, omega):
        max_upright_error = theta.abs().max(dim=1).values
        max_bend_error = (theta[:, 1:] - theta[:, :-1]).abs().max(dim=1).values
        strict_gate = (max_upright_error <= score_max_upright_angle) & (max_bend_error <= score_max_chain_bend)
        angle_error = (theta.abs() * link_weights).sum(dim=1)
        velocity_error = omega.abs().sum(dim=1)
        raw_score = torch.clamp(
            100.0 - angle_error * 12.0 - max_bend_error * 30.0 - velocity_error * 2.5 - cart_x.abs() * 8.0,
            0.0,
            100.0,
        )
        return torch.where(strict_gate, raw_score, torch.zeros_like(raw_score))

    def step(cart_x, cart_v, theta, omega, force):
        link_drag = (torch.sin(theta) * link_weights).sum(dim=1) * 0.08
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
        angular_acc = gravity + drive + coupling - damping * omega
        omega = torch.clamp(omega + angular_acc * dt, -18.0, 18.0)
        theta = torch.remainder(theta + omega * dt + torch.pi, 2 * torch.pi) - torch.pi
        return cart_x, cart_v, theta, omega

    def evaluate(params):
        batch = params.shape[0]
        cart_x, cart_v, theta, omega = make_initial(batch)
        reward = torch.zeros(batch, device=device)
        held = torch.zeros(batch, device=device)
        last = torch.zeros(batch, device=device)
        for tick in range(steps):
            force = policy_force(params, cart_x, cart_v, theta, omega, tick)
            cart_x, cart_v, theta, omega = step(cart_x, cart_v, theta, omega, force)
            score = app_score(cart_x, theta, omega)
            last = score
            late = tick / max(steps - 1, 1)
            hold_window = (tick > int(steps * 0.48)).float() if hasattr(tick, "float") else (1.0 if tick > int(steps * 0.48) else 0.0)
            max_upright_error = theta.abs().max(dim=1).values
            max_bend_error = (theta[:, 1:] - theta[:, :-1]).abs().max(dim=1).values
            dense_alignment = torch.exp(-max_upright_error * 1.6 - max_bend_error * 2.4 - omega.abs().mean(dim=1) * 0.08)
            reward += (score / 100.0).pow(2) * (0.2 + late * 1.4)
            reward += (score > 82).float() * (0.6 + late * 2.5 + hold_window * 5.0)
            reward += (score > 92).float() * hold_window * 3.0
            reward += torch.exp(-omega.abs().mean(dim=1) * 0.14) * (score > 82).float() * hold_window
            reward += dense_alignment * (0.03 + late * 0.08)
            reward -= cart_x.abs() * 0.025 + (force / force_scale).square() * 0.002
            held += (score > 82).float()
        reward += (last / 100.0).pow(4) * 180.0
        reward += held * 0.55
        reward -= params[:, :knot_count].diff(dim=1).abs().mean(dim=1) * 0.15
        return reward, last, held / steps

    history = []
    for generation in range(generations):
        params = mean.view(1, -1) + torch.randn(population, param_count, device=device) * sigma
        reward, last, held = evaluate(params)
        fitness = reward + last * 2.0 + held * 120.0
        values, indices = torch.topk(fitness, elite_count)
        elites = params[indices]
        mean = elites.mean(dim=0)
        sigma = max(sigma * 0.965, 0.08)
        if values[0] > best_score:
            best_score = values[0]
            best_params = elites[0].clone()
        if generation % 10 == 0 or generation == generations - 1:
            line = {
                "generation": generation,
                "bestFitness": float(values[0].detach().cpu()),
                "eliteReward": float(reward[indices[0]].detach().cpu()),
                "eliteScore": float(last[indices[0]].detach().cpu()),
                "eliteHeld": float(held[indices[0]].detach().cpu()),
                "sigma": float(sigma),
            }
            print(json.dumps(line), flush=True)
            history.append(line)

    validation_params = best_params.view(1, -1).repeat(1024, 1)
    reward, last, held = evaluate(validation_params)
    validation = {
        "reward": float(reward.mean().detach().cpu()),
        "score": float(last.mean().detach().cpu()),
        "held": float(held.mean().detach().cpu()),
        "scoreP10": float(torch.quantile(last, 0.1).detach().cpu()),
        "heldP10": float(torch.quantile(held, 0.1).detach().cpu()),
    }

    output = {
        "trainedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "algorithm": "modal-cross-entropy-method",
        "environment": "six-link-browser-cartpole-v3",
        "seed": 42612,
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
            "generations": generations,
            "elapsedSeconds": round(time.time() - started, 3),
            "history": history,
            "validation": validation,
            "warmStarted": bool(initial_policy_json),
            "strictScore": {
                "maxUprightAngleRad": score_max_upright_angle,
                "maxChainBendRad": score_max_chain_bend,
            },
        },
        "knots": best_params[:knot_count].detach().cpu().tolist(),
        "feedback": best_params[knot_count:].detach().cpu().tolist(),
    }
    return json.dumps(output, indent=2)


@app.local_entrypoint()
def main():
    from pathlib import Path

    policy_path = Path("app/ailab/six-pendulum-cartpole/sixPendulumPolicy.json")
    initial = policy_path.read_text() if policy_path.exists() else ""
    return train_policy.remote(initial)
