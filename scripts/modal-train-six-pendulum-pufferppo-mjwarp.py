#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path

import modal


app = modal.App("six-pendulum-pufferppo-mjwarp-train")
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "mujoco-warp==3.9.0.1",
    "numpy==2.2.6",
    "pufferlib==3.0.0",
    "torch==2.7.1",
)


DEFAULT_OUTPUT = Path(
    "local-artifacts/outputs/training-checkpoints/puffer-mjwarp-pufferppo-contract.json"
)


def one_link_sweep_rows() -> list[dict]:
    rows = []
    for nworld in (4096, 8192):
        for rollout_steps in (256, 512):
            for force_scale in (120, 240):
                rows.append(
                    {
                        "experimentId": f"pufferppo-1link-n{nworld}-r{rollout_steps}-f{force_scale}",
                        "links": 1,
                        "nworld": nworld,
                        "rolloutSteps": rollout_steps,
                        "bpttHorizon": min(256, rollout_steps),
                        "forceScale": force_scale,
                        "learningRate": 0.00025,
                        "entropyCoef": 0.01,
                        "clipCoef": 0.2,
                        "policyParamsTarget": 1_000_000,
                        "randomizedEpisodeLength": False,
                        "promotionGate": "held-out down-start learned policy holds upright >= 1.0s",
                    }
                )
    return rows


def build_puffer_train_config(row: dict, total_timesteps: int) -> dict:
    batch_size = int(row["nworld"]) * int(row["bpttHorizon"])
    return {
        "seed": 426210,
        "torch_deterministic": False,
        "device": "cuda",
        "cpu_offload": False,
        "batch_size": batch_size,
        "bptt_horizon": int(row["bpttHorizon"]),
        "minibatch_size": min(batch_size, 262144),
        "max_minibatch_size": 65536,
        "total_timesteps": int(total_timesteps),
        "learning_rate": float(row["learningRate"]),
        "anneal_lr": True,
        "gamma": 0.995,
        "gae_lambda": 0.95,
        "update_epochs": 2,
        "clip_coef": float(row["clipCoef"]),
        "vf_clip_coef": 0.2,
        "ent_coef": float(row["entropyCoef"]),
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "use_rnn": True,
        "compile": False,
        "compile_mode": "default",
        "compile_fullgraph": False,
        "optimizer": "adam",
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_eps": 1e-5,
        "precision": "float32",
        "prio_alpha": 0.7,
        "prio_beta0": 0.2,
        "vtrace_rho_clip": 1.0,
        "vtrace_c_clip": 1.0,
        "checkpoint_interval": 10,
        "env": "six-pendulum-mjwarp",
        "tag": row["experimentId"],
    }


def build_contract(total_timesteps: int = 10_000_000) -> dict:
    rows = one_link_sweep_rows()
    representative = rows[-1]
    return {
        "schema": "six-pendulum-pufferppo-mjwarp-contract-v1",
        "status": "pufferppo-contract-ready-ppo-update-smoke-gpu-puffer-integration-missing",
        "createdAtUnix": time.time(),
        "algorithm": "PufferPPO",
        "policyFamily": "PufferNet/MinGRU target, local smoke uses PufferLib Default+RNN API until PufferNet is wired",
        "sourceThreadClaims": {
            "pufferPPO": True,
            "pufferMinGRU": True,
            "policyParamsApprox": 1_000_000,
            "mujocoWarp": True,
            "apicCudaGraphCapture": True,
            "reportedExperiments": 3600,
            "reportedFastConfigSps": 18_000_000,
            "randomizedEpisodeLengthOnlyAfterWhip": True,
        },
        "hardGates": {
            "startLinks": 1,
            "learnedPolicyOnly": True,
            "strictHoldSeconds": 1.0,
            "teacherDoesNotCount": True,
            "linkTwoLockedUntilOneLinkPasses": True,
            "randomizedEpisodeLengthLockedUntilWhipExists": True,
        },
        "sweepRows": rows,
        "representativePufferTrainConfig": build_puffer_train_config(representative, total_timesteps),
        "gpuKernelProgress": {
            "firstParityGate": "scripts/six_pendulum_mjwarp_gpu_kernels.py",
            "command": "npm run train:six-pendulum:puffer-mjwarp:gpu-score-smoke",
            "artifact": "local-artifacts/outputs/training-checkpoints/puffer-mjwarp-gpu-score-kernel-smoke.json",
            "covered": "Links 1..6 observation, reward, potential, strict score, multi-link cumsum, bend penalties, catch basin, near-top-fast, whip flags, and cart terminal match the existing NumPy scorer.",
            "envIntegration": "SixPendulumMJWarpPufferEnv now uses Warp kernels for reset sampling/writes, action scaling, ctrl writes, score/reward/observation, cart terminal, truncation, held/max-held accumulation, and potential-delta reward; env-driver artifacts report resetBackend=warp-reset-kernel, scoreBackend=warp-score-kernel, and rolloutBackend=warp-post-step-kernel.",
            "deviceRollout": {
                "command": "npm run train:six-pendulum:puffer-mjwarp:device-rollout",
                "artifact": "local-artifacts/outputs/training-checkpoints/puffer-mjwarp-device-rollout.json",
                "covered": "MJWarp loop can run reset/action/score/post-step kernels without per-step CPU metric reads; only summary arrays are copied after final synchronize.",
                "caveat": "The action source is a deterministic scripted Warp kernel for substrate proof only. It is not a learned policy and does not count toward solve.",
            },
            "rolloutBuffer": {
                "command": "npm run train:six-pendulum:puffer-mjwarp:device-rollout:buffer",
                "artifact": "local-artifacts/outputs/training-checkpoints/puffer-mjwarp-device-rollout-buffer.json",
                "covered": "Fixed-shape observation, reward, terminal, truncation, and action buffers are written on device each step and copied once after final synchronize.",
                "whyItMatters": "This is the trainer bridge: PufferPPO needs fixed rollout tensors, not per-step Python info dictionaries.",
            },
            "actionBuffer": {
                "command": "npm run train:six-pendulum:puffer-mjwarp:device-rollout:action-buffer",
                "artifact": "local-artifacts/outputs/training-checkpoints/puffer-mjwarp-device-rollout-action-buffer.json",
                "covered": "A fixed-shape external action tensor is copied before rollout and consumed by a Warp ctrl kernel each step without per-step CPU action writes.",
                "caveat": "The current tensor is deterministic and precomputed. It proves the policy-action interface only; it is not a learned policy and does not count toward solve.",
            },
            "torchPolicyBridge": {
                "command": "npm run train:six-pendulum:puffer-mjwarp:device-rollout:torch-policy",
                "artifact": "local-artifacts/outputs/training-checkpoints/puffer-mjwarp-device-rollout-torch-policy.json",
                "covered": "The rollout converts the current Warp observation buffer with wp.to_torch, runs a recurrent Torch actor-critic with per-world hidden state, converts normalized policy actions/logprobs/values back with wp.from_torch, records fixed-shape PPO buffers, then writes ctrl through a Warp action-vector kernel without NumPy action writes.",
                "caveat": "The local smoke policy is tiny and untrained, and local Mac execution uses Warp CPU. The next bridge is PufferPPO ownership of these buffers, minibatch update plumbing, and GPU stream/capture discipline.",
            },
            "ppoUpdateSmoke": {
                "command": "npm run train:six-pendulum:puffer-mjwarp:device-rollout:ppo-update",
                "artifact": "local-artifacts/outputs/training-checkpoints/puffer-mjwarp-device-rollout-ppo-update.json",
                "covered": "Three PPO epochs consume the fixed recurrent rollout buffers, recompute logprobs/values over the stored sequence, backpropagate through the recurrent actor-critic, and change parameters. The rollout buffer now records pre-action observations; the first PPO epoch reconstructs ratioMean=1.0 and ratioMax=1.0 in the smoke artifact.",
                "caveat": "This is a local CPU fixed-batch smoke update, not a PufferPPO training run and not a learned policy solve.",
            },
            "devicePpoTrain": {
                "command": "npm run train:six-pendulum:puffer-mjwarp:device-ppo-train",
                "artifact": "local-artifacts/outputs/training-checkpoints/puffer-mjwarp-device-ppo-train.json",
                "covered": "Repeated stochastic recurrent-policy collection, persistent PPO optimizer updates, and deterministic held-out down-start/hold-start evaluation after each update on the MJWarp rollout-buffer path.",
                "caveat": "Local Mac execution still uses Warp/MJWarp CPU and is a correctness path, not the final PufferPPO/MinGRU GPU sweep.",
            },
            "holdProbe": {
                "preferredCommand": "npm run train:six-pendulum:puffer-mjwarp:device-ppo-hold-probe",
                "preferredArtifact": "local-artifacts/outputs/training-checkpoints/puffer-mjwarp-device-ppo-hold-probe-f32.json",
                "comparisonCommand": "npm run train:six-pendulum:puffer-mjwarp:device-ppo-hold-probe-f64",
                "comparisonArtifact": "local-artifacts/outputs/training-checkpoints/puffer-mjwarp-device-ppo-hold-probe.json",
                "bcWarmupCommand": "npm run train:six-pendulum:puffer-mjwarp:device-ppo-hold-bc-probe",
                "bcWarmupArtifact": "local-artifacts/outputs/training-checkpoints/puffer-mjwarp-device-ppo-hold-bc-probe.json",
                "covered": "A three-update 2.56s-horizon hold-start probe at forceScale 32 reached held-out hold-start 0.435s with no cart terminal; the same probe at forceScale 64 reached only 0.3175s and all hold eval worlds hit cart terminal. After fixing PPO buffer observation timing and strengthening swing-up reward, a learned stabilizer BC warmup on the same recurrent device-buffer policy reached held-out hold-start 1.4425s.",
                "caveat": "BC warmup proves the policy can learn the top stabilizer, but it is curriculum progress only. Held-out down-start remains 0.0s and does not promote to link two.",
            },
            "policyReflection": {
                "consults": [
                    "local-artifacts/outputs/model-consults/hermes-pendulum-policy-reflection.md",
                    "local-artifacts/outputs/model-consults/gemini-pendulum-policy-reflection.md",
                    "local-artifacts/outputs/model-consults/oracle-pendulum-policy-reflection.md",
                ],
                "diagnosis": "PPO tuple plumbing is now coherent: pre-action observations are stored with their sampled actions/logprobs, and ratio reconstruction is 1.0 on the first smoke epoch. The remaining blocker is pure down-start transfer: forceScale 120 mixed-start probes reach near-catch flashes up to 0.8825s inside curriculum rollouts, but held-out down-start remains 0.0s.",
                "nextCommand": "npm run train:six-pendulum:puffer-mjwarp:device-ppo-down-heavy-conservative",
            },
            "actionScaleDiagnostic": {
                "command": "npm run train:six-pendulum:puffer-mjwarp:action-scale-diagnostic",
                "artifact": "local-artifacts/outputs/training-checkpoints/puffer-mjwarp-action-scale-diagnostic.json",
                "covered": "Open-loop max-force probes from one-link down-start show forceScale 32 and 64 do not reach near-vertical in 1s; forceScale 120 reaches near-vertical at about 0.60s and 240 at about 0.335s, usually with cart-terminal risk.",
                "caveat": "This is an action-authority diagnostic, not a policy and not a solve.",
            },
            "randomHorizonSupport": {
                "command": "npm run train:six-pendulum:puffer-mjwarp:device-rollout:random-horizon",
                "artifact": "local-artifacts/outputs/training-checkpoints/puffer-mjwarp-device-rollout-random-horizon.json",
                "covered": "Per-world truncation horizons are sampled in the reset Warp kernel and consumed by the post-step Warp kernel without per-step host reads.",
                "gate": "Keep disabled until a learned policy shows whip/near-catch behavior; this matches the source thread warning that random horizons helped only after the model was already scoring.",
            },
            "notCovered": "Puffer rollout integration, PufferPPO/MinGRU training attachment, Modal GPU execution, and CUDA graph/APIC capture.",
        },
        "gpuKernelBlocker": {
            "currentEnv": "scripts/six_pendulum_mjwarp_env.py",
            "problem": "The standalone device-rollout smoke removes per-step CPU metric reads, but the current PufferEnv interface still returns NumPy observations/rewards/dones each step.",
            "whyItBlocksYacineSpeed": "Those remaining CPU transfers and Python-side bookkeeping still defeat MJWarp batched GPU throughput and cannot reach wallclock-first PufferPPO speed.",
            "requiredBeforeRealSweep": [
                "Replace the Python/PufferEnv CPU return interface with a Puffer-compatible rollout path that can keep tensors device-side.",
                "Keep fixed shapes for nworld, obs_dim, action_dim, and rollout horizon so CUDA graph capture is possible.",
                "Only then attach pufferlib.pufferl.PuffeRL to the vector env and run 4096+ worlds.",
            ],
        },
        "nextCommands": [
            "npm run train:six-pendulum:puffer-mjwarp:pufferppo-contract",
            "npm run train:six-pendulum:puffer-mjwarp:device-ppo-hold-bc-probe",
            "npm run train:six-pendulum:puffer-mjwarp:device-ppo-down-swingup-probe",
            "npm run train:six-pendulum:puffer-mjwarp:device-ppo-down-swingup-conservative",
            "npm run train:six-pendulum:puffer-mjwarp:device-ppo-down-heavy-conservative",
            "npm run train:six-pendulum:puffer-mjwarp:device-ppo-link2-diagnostic",
            "npm run train:six-pendulum:puffer-mjwarp:device-ppo-link3-diagnostic",
            "npm run train:six-pendulum:puffer-mjwarp:device-ppo-link4-diagnostic",
            "modal run --write-result local-artifacts/outputs/training-checkpoints/puffer-mjwarp-pufferppo-runtime.json scripts/modal-train-six-pendulum-pufferppo-mjwarp.py::inspect_pufferppo_runtime",
        ],
    }


@app.function(image=image, gpu="L4", timeout=1800)
def inspect_pufferppo_runtime(total_timesteps: int = 10_000_000) -> str:
    import pufferlib
    import pufferlib.models
    import pufferlib.pufferl
    import pufferlib.sweep
    import torch
    import warp as wp
    import mujoco_warp as mjw

    contract = build_contract(total_timesteps)
    contract["status"] = "pufferppo-runtime-inspected-score-kernel-parity-env-integration-missing"
    contract["runtime"] = {
        "torch": getattr(torch, "__version__", "unknown"),
        "torchCudaAvailable": bool(torch.cuda.is_available()),
        "torchCudaDeviceCount": int(torch.cuda.device_count()),
        "pufferlib": getattr(pufferlib, "__version__", "unknown"),
        "mujocoWarp": getattr(mjw, "__version__", "unknown"),
        "warp": getattr(wp, "__version__", "unknown"),
        "devices": [str(device) for device in wp.get_devices()],
        "pufferPpoClass": f"{pufferlib.pufferl.PuffeRL.__module__}.{pufferlib.pufferl.PuffeRL.__name__}",
        "defaultPolicyClass": f"{pufferlib.models.Default.__module__}.{pufferlib.models.Default.__name__}",
        "sweepClass": f"{pufferlib.sweep.Protein.__module__}.{pufferlib.sweep.Protein.__name__}",
    }
    return json.dumps(contract, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Create or inspect the real six-pendulum PufferPPO/MJWarp training contract.")
    parser.add_argument("--total-timesteps", type=int, default=10_000_000)
    parser.add_argument("--write-result", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--modal-runtime", action="store_true")
    args = parser.parse_args()

    if args.modal_runtime:
        result = json.loads(inspect_pufferppo_runtime.remote(args.total_timesteps))
    else:
        result = build_contract(args.total_timesteps)

    args.write_result.parent.mkdir(parents=True, exist_ok=True)
    args.write_result.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
