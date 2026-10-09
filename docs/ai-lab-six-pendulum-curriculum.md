---
read_when:
  - Working on the six pendulum cartpole AI Lab route
  - Changing six pendulum training scripts
  - Running Modal GPU training for cartpole policies
---

# Six Pendulum Curriculum Plan

## Current Status

Live route: https://www.maxpetrusenko.com/ailab/six-pendulum-cartpole

One-link down-start is now solved in the local browser runtime using the Modal-trained Pezzza-style policy. Six links are not solved. Strict score is zero until the active chain is near upright, nearly straight, and held for at least one consecutive second.

Separate model-based progress: `m1el/inverted-pendulum` now gives this project a locally verified six-link control solve. It is not the browser neural checkpoint. It is seed-free controllability-aware direct collocation plus full-state TVLQR on verified N-link dynamics. Local repro generated controls, passed nominal verification, and passed the perturbed-start challenge.

## Source Evidence

- X thread: https://x.com/yacineMTB/status/2064148140899348779
- Raw Bird thread read saved to `local-artifacts/outputs/research/yacine-thread-bird-2064148140899348779.txt`.
- Current raw Bird JSON saved to `local-artifacts/assets/x-yacine/2064148140899348779.thread.json`.
- Bird thread read confirmed the core claims: PufferPPO, Puffer MinGRU, MuJoCo Warp, APIC/CUDA graph capture, 18M SPS on some configs, 3.6k experiments, GP-picked top hyperparameters, and randomized episode length after whipping appeared.
- Bird also confirmed gravity `9.8`, no hinge friction, velocity reduction to keep the chain straight, and the cart attempting to track `0`.
- Hero solve video downloaded to `outputs/yacine-thread-media/2064145781477580800-*.mp4`.
- Hyperparameter scatter video downloaded from `2064150381408485769`.
- Reward/policy video downloaded from `2064152523095560528`.
- Phase-space / simulator-speed video downloaded from `2064155513244246028`.
- Current downloaded media and sampled frames are under `local-artifacts/assets/x-yacine/`.
- Main clues: PufferPPO, Puffer MinGRU, MuJoCo Warp, APIC/CUDA graph capture, 3.6k experiments, top hyperparameters selected from high-compute runs, randomized episode length after whip behavior appeared.
- Pezzza video downloaded with `yt-dlp` to `local-artifacts/outputs/youtube/pezzza-double-pendulum.mp4`.
- Pezzza transcript downloaded to `local-artifacts/outputs/youtube/pezzza-double-pendulum.en.vtt`; frames sampled under `outputs/youtube/frames/`.
- XPBD paper downloaded to `local-artifacts/outputs/papers/XPBD.pdf`.
- `johnBuffer/Pendulum-NEAT` cloned under `oss/Pendulum-NEAT`.
- `m1el/inverted-pendulum` cloned under `oss/inverted-pendulum`. This is the strongest public N=6 source found: verified dynamics, seed-free controllability-aware trajectory optimization, TVLQR, official verifier, and media.
- `jgerstmayr/EXUDYN` `openAIgymNLinkAdvanced.py` is the strongest public N=5 RL source found. It documents SAC for multi-link inverted pendulums, says four and five links work, uses smaller random initialization/disturbance for `>=5`, and includes N=6 actuation/threshold parameters but no public N=6 checkpoint.
- Deep-search artifacts are saved under `local-artifacts/outputs/research/`. Perplexity API returned `401 insufficient_quota`; Gemini CLI hit capacity/503 retries; DeepSeek completed but was treated as advisory because it hallucinated some commands.

### Canonical Yacine Requirements From Bird Read

Treat these as hard constraints for the RL reproduction:

1. Method is RL, not model-based control: train an environment with `pufferPPO`.
2. Infrastructure matters: PufferLib wallclock speed, MuJoCo Warp, APIC/CUDA graph capture callable from C.
3. Scale is many experiments: Yacine reported `3.6k` experiments, with wallclock on x-axis and score on y-axis.
4. Policy is small recurrent: puffer MinGRU, about `1m` parameters.
5. Hyperparameters come from broad sweeps: use GP-picked/top-scoring higher-compute runs, then tweak the task.
6. Reward shaping is active: help the policy find whipping behavior, then stabilize.
7. Randomized episode length is a late-stage trick: add it after the model learns whipping but gets lazy around fixed episode timing.
8. Physics constraints: gravity `9.8`, no hinge friction.
9. Task constraint: cart is trying to track `0`.
10. No subsecond score: public score still requires at least one continuous upright hold second.

Implication for current code:

The m1el route proves a six-link control trajectory and gives the bend-order lesson. The Yacine route requires a separate high-throughput RL path, likely PufferPPO/MuJoCo Warp/MinGRU, not more small local CEM.

## Verified Model-Based N=6 Reproduction

Source:
`oss/inverted-pendulum`

The decisive source insight is bend order. A six-link chain that goes near-straight mid-swing is near-uncontrollable from one pivot. The working trajectory keeps a coordinated bend shimmy and adds a one-sided soft floor on bend-mode excitation instead of globally fighting the swing.

Commands:

```bash
uv run python tests/test_dynamics.py
uv run python repro/generate_n6.py
uv run python repro/simulate_n6.py local-artifacts/outputs/m1el-n6-seedfree-proof.mp4 repro/n6_controls.npz
uv run python repro/perturbed_n6.py repro/n6_controls.npz --render local-artifacts/outputs/m1el-n6-seedfree-perturbed-proof.mp4
```

Results:

- Dynamics tests passed.
- Seed-free generator selected `soft_w100_fl0.8`.
- Generator metrics: final `0.0184deg`, maxK `68110`, accel max `7.3`, saved `repro/n6_controls.npz`.
- Nominal verifier: final angle error `0.364deg` over all six links, `VERIFICATION: PASS`.
- Perturbed challenge: `16/16 succeed`, `CHALLENGE: PASS`, robust to `>= +/-0.80 rad` initial angle error.
- Hosted proof videos copied to:
  - `./public/ailab/six-pendulum/m1el-n6-seedfree-proof.mp4`
  - `./public/ailab/six-pendulum/m1el-n6-seedfree-perturbed-proof.mp4`

Trainer implications:

1. Do not reward straightness during pump.
2. Add bend-excitation/controllability shaping that penalizes bend collapse only below a threshold.
3. Keep phases explicit: energy pump, catch, hold.
4. Keep strict score separate: no score for subsecond holds.
5. Use Exudyn's N=5 SAC setup as a reward/curriculum reference only. The Yacine reproduction lane is PufferPPO or close recurrent PPO on MuJoCo Warp, not more browser CEM or plain SAC.

## Multi-Agent Review

Agents were split by approach:

- Aristotle: Pendulum-NEAT, Pezzza transcript, XPBD source table.
- Ampere: CEM/evolution recommendation and policy shape.
- Pascal: PPO/SAC/TD3 baseline comparison.
- Goodall: XPBD/MuJoCo physics recommendation.
- Pauli: PufferPPO/PufferLib wallclock-first sweep model.
- Hegel: Puffer MinGRU policy ablation.
- Boole: MuJoCo Warp, MJX, APIC/CUDA graph speed path.
- Tesla: reward shaping, randomized horizon trigger, GP/hyperparameter sweep schema.

Consensus:

1. Do not count holds shorter than one second.
2. Keep strict validation separate from shaped reward.
3. Use CEM/evolution for fast one-link discovery.
4. Use many experiments and wallclock-vs-score reporting before trying to scale.
5. Treat MinGRU/PufferPPO as the next controlled RL ablation.
6. Keep MuJoCo/MJX/MJWarp for later physics-speed work; XPBD is useful for browser visual fidelity, not the current training bottleneck.

## Plan

1. Train lower link counts first: 1 link, then 2, then 3, then 4, then 5, then 6.
2. Use the same policy shape across stages so later stages inherit useful timing and feedback from earlier stages.
3. Keep dense reward shaping for learning, but keep public score strict.
4. Track two different milestones:
   - Whip: the policy learns to swing the chain toward upright.
   - Hold: the policy keeps the active chain upright and straight under strict score.
5. Add randomized episode lengths only after early whip/hold appears, matching Yacine's thread.
6. Save every run as JSON with stage validation, score, held fraction, whip fraction, seed, GPU, and elapsed time.
7. Move from the browser physics approximation to MuJoCo Playground or MJWarp once the curriculum path is proven.

## Yacine RL Reproduction Workstream

This is now the target lane. The existing Pezzza/CEM and Python MuJoCo PPO scripts are useful lower-link research, but they are not the reproduced solve because they do not use PufferPPO, Puffer MinGRU, MuJoCo Warp batching, or CUDA graph capture.

Phase 0, proof substrate:

```bash
npm run six-pendulum:mjcf
npm run train:six-pendulum:puffer-mjwarp:local-smoke
npm run train:six-pendulum:puffer-mjwarp:smoke
```

The local smoke writes `outputs/training-checkpoints/puffer-mjwarp-local-substrate-smoke.json`. The Modal smoke writes `outputs/training-checkpoints/puffer-mjwarp-substrate-smoke.json`. These are not solve metrics. They prove the substrate before spending on PPO: PufferLib and MuJoCo Warp import, the same MJCF loads, `nworld` batched worlds step, and Warp graph capture/replay is attempted.

Local substrate result:

- Command: `npm run train:six-pendulum:puffer-mjwarp:local-smoke`
- Artifact: `local-artifacts/outputs/training-checkpoints/puffer-mjwarp-local-substrate-smoke.json`
- Result: `substrate-smoke-passed`, `nworld=4`, `steps=16`, `64` simulated world-steps.
- Device: local Mac Warp CPU only, so throughput is not meaningful for the final target.
- Model invariants: `nq=2`, `nv=2`, `nu=1`, timestep `0.0025`, gravity `[0, 0, -9.8]`, DOF damping `0`, DOF frictionloss `0`.
- MJCF fix: visual geoms are now contact-disabled with `contype=0` and `conaffinity=0`; the six-pendulum task has no contacts, and this avoids MJWarp friction/contact warnings from decorative rail/cart/link geoms.

Local environment-contract result:

- Command: `npm run train:six-pendulum:puffer-mjwarp:local-env`
- Artifact: `local-artifacts/outputs/training-checkpoints/puffer-mjwarp-local-env-contract.json`
- Result: `env-contract-smoke-passed`, `links=1`, `nworld=4`, `steps=64`.
- Reset contract now covers down-start and near-upright hold-start batches.
- Observation contract is `[nworld, 33]`: cart position, cart velocity, previous action, and padded six-link sin/cos/velocity features.
- Reward/score contract now separates shaped reward from strict score.
- Down-start strict score stayed `0`; hold-start reached max strict score `99.73`, but the one-second gate still stayed false because the smoke ran only `0.16s`. This proves subsecond upright flashes are not counted as solved.

PufferEnv driver result:

- Command: `npm run train:six-pendulum:puffer-mjwarp:env-driver`
- Artifact: `local-artifacts/outputs/training-checkpoints/puffer-mjwarp-env-driver.json`
- Script: `scripts/six_pendulum_mjwarp_env.py`
- Result: `puffer-env-driver-smoke-passed`, `links=1`, `nworld=4`, `steps=128`.
- Adapter: `SixPendulumMJWarpPufferEnv` exposes Puffer-compatible `single_observation_space` and `single_action_space`, internal reset, `step(actions)`, reward, terminals, truncations, and per-agent info.
- Action contract: normalized action `[-1, 1]` maps to cart force `[-32, 32]`.
- Down-start baseline still scored `0`, which is expected. This artifact proves the trainer-facing environment loop, not learning.

Local recurrent training smoke:

- Command: `npm run train:six-pendulum:puffer-mjwarp:local-ppo`
- Artifact: `local-artifacts/outputs/training-checkpoints/puffer-mjwarp-local-ppo-smoke.json`
- Script: `scripts/train_six_pendulum_mjwarp_local_ppo.py`
- Result: `training-smoke-passed`, `links=1`, `nworld=4`, `rolloutSteps=96`, `evalSteps=480`, `updates=3`.
- Policy: tiny GRU policy-gradient smoke, `27,267` parameters. This is a local CPU wiring test, not the final Puffer MinGRU scale.
- Hold-start eval: max strict score `99.73`, max held `0.46s` over a `1.2s` validation horizon, one-second gate false.
- Down-start eval: max strict score `0`, max held `0s`, promotion false.
- Interpretation: recurrent training now runs through MJWarp and backpropagates, but it does not solve one-link down-start. Next useful work is GPU scale plus proper PPO minibatches, then Puffer MinGRU/PufferNet.

Stronger local hold-search:

- Command: `npm run train:six-pendulum:puffer-mjwarp:hold-search`
- Artifact: `local-artifacts/outputs/training-checkpoints/puffer-mjwarp-local-ppo-hold-search.json`
- Result: `links=1`, `nworld=8`, `rolloutSteps=160`, `evalSteps=640`, `updates=16`.
- Hold-start eval improved to max held `0.5875s` over a `1.6s` validation horizon, still below the one-second gate.
- Down-start eval remained strict score `0`, max held `0s`, so no link-two promotion.
- Interpretation: more local CPU updates improve near-upright stabilization but plateau below the strict hold gate. The next change should improve the PPO update or use GPU scale, not unlock more links.

Learned hold-start stabilizer:

- Command: `npm run train:six-pendulum:puffer-mjwarp:stabilizer-bc`
- Artifact: `local-artifacts/outputs/training-checkpoints/puffer-mjwarp-stabilizer-bc.json`
- Result: `behavior-clone-smoke-passed`, `links=1`, `nworld=8`, `rolloutSteps=480`, `evalSteps=640`, `epochs=600`.
- Expert target: near-upright stabilizer `force = -(kx*x + kv*v + kt*theta + kw*omega)` with gains `kx=8`, `kv=4`, `kt=-60`, `kw=-16`. This expert held `1.2s` in MJWarp.
- Learned model: same tiny GRU policy, `27,267` parameters, trained by behavior cloning from the stabilizer target.
- Learned hold-start eval: max strict score `99.89`, max held `1.285s`, one-second gate passed.
- Learned down-start eval: strict score `0`, max held `0s`, so there is still no link-two promotion.
- Interpretation: one-link near-upright catch/hold is now learned in the MJWarp lane. This is curriculum pretraining, not the final down-start swing-up solve.

Checkpoint warmstart and mixed curriculum:

- Commands:
  - `npm run train:six-pendulum:puffer-mjwarp:down-warmstart`
  - `npm run train:six-pendulum:puffer-mjwarp:mixed-warmstart`
- Artifacts:
  - `local-artifacts/outputs/training-checkpoints/puffer-mjwarp-down-warmstart.json`
  - `local-artifacts/outputs/training-checkpoints/puffer-mjwarp-mixed-warmstart.json`
- Code changes: the MJWarp env now uses seeded random perturbations instead of deterministic linspace resets, adds energy/potential/catch-basin shaping for down-start, and supports a `mixed` reverse-curriculum reset.
- Horizon fix: warmstart training now uses `480` rollout steps (`1.2s`) and `800` eval steps (`2.0s`), so subsecond flashes are not counted as training progress.
- Down warmstart result: randomized hold eval stayed solved at `1.2325s`; pure down-start eval stayed `0s`.
- Mixed warmstart result: mixed training produced one-second-plus catch/hold inside rollouts on `6/8` updates, best `1.2s`; held-out pure down-start eval stayed `0s`.
- Interpretation: the learned catch policy is real, and the reverse curriculum can exercise one-second holds. The missing skill is still swing-up from pure down-start. Do not unlock link two.

Energy teacher scaffold for PufferPPO:

- Commands:
  - `npm run train:six-pendulum:puffer-mjwarp:energy-teacher`
  - `npm run train:six-pendulum:puffer-mjwarp:energy-teacher-bc`
- Artifacts:
  - `local-artifacts/outputs/training-checkpoints/puffer-mjwarp-energy-teacher.json`
  - `local-artifacts/outputs/training-checkpoints/puffer-mjwarp-energy-teacher-bc.json`
- MJCF motor range is now `[-240, 240]`; existing envs still default to normalized force scale `32`, while the one-link teacher lane uses force scale `240`.
- Teacher: Astrom/Furuta-style energy pumping with symmetry-breaking kick plus stabilizer catch. Tuned local gains: `kE=12`, `kv=0.6`, `kx=0.05`, `aMax=30`, `forcePerAcceleration=6.75`, catch angle `0.36`, catch rate `3.0`.
- Teacher result: pure one-link down-start in MJWarp reached max strict score `99.04` and max held `1.3875s` over `10s`, so the swing-up/catch signal exists in the MJWarp environment.
- Learned tiny-GRU BC result: teacher rollout loss fell to `0.00028`, but closed-loop held-out down-start stayed `0s`; no link-two promotion.
- Sequence BC result: GRU sequence training reached held-out down-start max strict score `77.13`, but max held stayed `0s`.
- Sequence-BC warmstart RL result: in-training max held improved to `0.2275s`; held-out down-start stayed `0s`.
- DAgger result: online learner-state labeling produced learner rollouts with catch/near-top events, but held-out down-start stayed `0s`.
- Anchored warmstart result: teacher-anchor PPO improved in-training max held to `0.37s`; held-out down-start stayed `0s`.
- Interpretation: this is useful reward/curriculum scaffolding for the Yacine-style PufferPPO sweep, not a final learned policy. The next real reproduction step is many PufferPPO/MJWarp experiments with MinGRU-sized policies and wallclock-vs-score logging.

Phase 1, one-link PufferPPO:

- Environment: same MJCF constraints, gravity `9.8`, hinge friction `0`, cart track centered at `0`.
- Observation: cart position, cart velocity, link absolute/relative angle encodings, angular velocities, previous action. Do not leak score-only terms into observation.
- Reward: dense height/energy/whip shaping plus strict hold bonus. Strict score remains zero until at least one continuous upright second.
- Policy: start small recurrent, then grow toward puffer MinGRU/PufferNet and the reported `~1m` parameter policy.
- Sweep: run many PufferPPO/MJWarp experiments, report wallclock on x-axis and strict score on y-axis, and promote only held-out down-start checkpoints.
- PufferPPO contract command: `npm run train:six-pendulum:puffer-mjwarp:pufferppo-contract`
- PufferPPO contract artifact: `local-artifacts/outputs/training-checkpoints/puffer-mjwarp-pufferppo-contract.json`
- Runtime inspect command: `npm run train:six-pendulum:puffer-mjwarp:pufferppo-runtime`
- Runtime inspect result on 2026-06-10: `App creation failed: workspace billing cycle spend limit reached`.
- Contract result: first sweep rows are one-link only, `nworld={4096,8192}`, `rollout={256,512}`, `forceScale={120,240}`, fixed horizon first, and randomized episode length locked until learned whip behavior appears.
- GPU score-kernel smoke command: `npm run train:six-pendulum:puffer-mjwarp:gpu-score-smoke`
- GPU score-kernel smoke artifact: `local-artifacts/outputs/training-checkpoints/puffer-mjwarp-gpu-score-kernel-smoke.json`
- GPU score-kernel smoke result on 2026-06-10: passed on local Warp CPU for `512` worlds at each link count `1..6`. Max parity error vs the existing NumPy scorer was `7.62939453125e-06` across observation, reward, potential, strict score, multi-link cumsum, bend penalties, catch basin, near-top-fast, whip, and cart-terminal terms.
- Env-driver integration result on 2026-06-10: `npm run train:six-pendulum:puffer-mjwarp:env-driver` now reports `resetBackend: warp-reset-kernel`, `scoreBackend: warp-score-kernel`, and `rolloutBackend: warp-post-step-kernel`; a separate six-link driver artifact also passed at `local-artifacts/outputs/training-checkpoints/puffer-mjwarp-env-driver-link6-kernel.json`.
- Device-rollout smoke command: `npm run train:six-pendulum:puffer-mjwarp:device-rollout`
- Device-rollout smoke artifact: `local-artifacts/outputs/training-checkpoints/puffer-mjwarp-device-rollout.json`
- Six-link device-rollout smoke artifact: `local-artifacts/outputs/training-checkpoints/puffer-mjwarp-device-rollout-link6.json`
- Device-rollout result on 2026-06-10: one-link ran `128` worlds for `256` steps and six-link ran `16` worlds for `32` steps with `cpuMetricReadsPerStep=0` and `cpuStateWritesPerStep=0`; summary arrays are copied only after final synchronize. Both use local Mac Warp CPU, so SPS is not the final GPU number.
- Rollout-buffer device smoke command: `npm run train:six-pendulum:puffer-mjwarp:device-rollout:buffer`
- Rollout-buffer device smoke artifact: `local-artifacts/outputs/training-checkpoints/puffer-mjwarp-device-rollout-buffer.json`
- Rollout-buffer result on 2026-06-10: one-link wrote fixed-shape device buffers for observations `[32, 8, 33]`, rewards `[32, 8]`, terminals `[32, 8]`, truncations `[32, 8]`, and actions `[32, 8]`; all checked finite after one final read.
- Action-buffer device smoke command: `npm run train:six-pendulum:puffer-mjwarp:device-rollout:action-buffer`
- Action-buffer device smoke artifact: `local-artifacts/outputs/training-checkpoints/puffer-mjwarp-device-rollout-action-buffer.json`
- Action-buffer result on 2026-06-10: one-link copied a fixed time-major `[32, 8]` normalized action tensor before rollout, consumed it through `warp-action-buffer-kernel`, wrote ctrl/last-action on device, and kept `actionPlanCpuWritesPerStep=0`. The driver now rejects wrong-shaped or non-finite action plans. The recorded rollout action buffer is scaled cart force; PPO/logprob plumbing still needs the normalized sampled action too. This is precomputed deterministic policy-interface plumbing only, not a learned policy.
- Torch-policy bridge smoke command: `npm run train:six-pendulum:puffer-mjwarp:device-rollout:torch-policy`
- Torch-policy bridge artifact: `local-artifacts/outputs/training-checkpoints/puffer-mjwarp-device-rollout-torch-policy.json`
- Torch-policy bridge result on 2026-06-10: the rollout converts current Warp observations with `wp.to_torch(runner.obs_wp)`, runs a tiny recurrent Torch actor-critic, converts normalized actions/logprobs/values back with `wp.from_torch`, records fixed PPO-style buffers, and writes ctrl through `warp-policy-action-vector-kernel` with `torchPolicyCpuActionWritesPerStep=0`. The policy has `27267` parameters, hidden state shape `[8, 64]`, and is untrained, so it proves recurrent rollout plumbing only and does not count toward solve.
- PPO-update smoke command: `npm run train:six-pendulum:puffer-mjwarp:device-rollout:ppo-update`
- PPO-update smoke artifact: `local-artifacts/outputs/training-checkpoints/puffer-mjwarp-device-rollout-ppo-update.json`
- PPO-update result on 2026-06-10: three local CPU PPO epochs consumed the fixed recurrent buffers, recomputed logprobs/values over the stored sequence, backpropagated, and changed parameters (`parameterDeltaL2=0.11130`, final `gradNorm=0.01641`). This proves fixed-batch buffer-to-update plumbing only; it is not a PufferPPO sweep and not a learned solve.
- PPO-buffer timing fix on 2026-06-10: the rollout buffer now stores the pre-action observation with the action/logprob/value sampled from it. Before this fix, PPO was reconstructing logprobs against post-step observations. The rerun `device-rollout:ppo-update` smoke now has first-epoch `ratioMean=1.0` and `ratioMax=1.0`, so the PPO tuples are coherent.
- Repeated device-buffer PPO command: `npm run train:six-pendulum:puffer-mjwarp:device-ppo-train`
- Repeated device-buffer PPO artifact: `local-artifacts/outputs/training-checkpoints/puffer-mjwarp-device-ppo-train.json`
- Repeated device-buffer PPO status: this is the first local path that repeatedly collects stochastic recurrent policy rollouts, updates the same policy with a persistent PPO optimizer, and runs deterministic held-out down-start/hold-start evaluation after every update. It still runs on local Mac Warp/MJWarp CPU, not Modal GPU/PufferPPO/MinGRU scale.
- Reflection checkpoint on 2026-06-10: Hermes, Gemini, and Oracle independently flagged the same policy failure. The `96` step rollout is only `0.24s`, `nworld=8` is too small for PPO, down-start reward is effectively flat, and hold-start has not passed one second. Next proof should be a one-link hold-start long-horizon run plus action-scale reachability check before more down-start or GPU work.
- Action-scale diagnostic command: `npm run train:six-pendulum:puffer-mjwarp:action-scale-diagnostic`
- Action-scale diagnostic artifact: `local-artifacts/outputs/training-checkpoints/puffer-mjwarp-action-scale-diagnostic.json`
- Action-scale diagnostic result on 2026-06-10: one-link open-loop probes from down-start show `forceScale=32` and `64` do not reach near-vertical in `1.0s`; `120` reaches near-vertical in about `0.60s`; `240` reaches it in about `0.335s` but usually with cart-terminal risk. This is not a policy and does not count.
- Long-horizon hold-start probe command: `npm run train:six-pendulum:puffer-mjwarp:device-ppo-hold-probe`
- Hold-start probe artifacts:
  - Preferred `forceScale=32`: `local-artifacts/outputs/training-checkpoints/puffer-mjwarp-device-ppo-hold-probe-f32.json`
  - Comparison `forceScale=64`: `local-artifacts/outputs/training-checkpoints/puffer-mjwarp-device-ppo-hold-probe.json`
- Hold-start probe result on 2026-06-10: `forceScale=32` is safer for hold-start, reaching held-out hold `0.435s` over `2.56s` eval with zero terminal worlds; stochastic rollouts reached up to `0.6825s`. `forceScale=64` reached only `0.3175s` and all hold eval worlds hit cart terminal. Neither passes the one-second gate.
- Device hold-start BC warmup command: `npm run train:six-pendulum:puffer-mjwarp:device-ppo-hold-bc-probe`
- Device hold-start BC warmup artifact: `local-artifacts/outputs/training-checkpoints/puffer-mjwarp-device-ppo-hold-bc-probe.json`
- Device hold-start BC warmup result on 2026-06-10: after the buffer-timing fix and stronger swing-up reward, `300` BC epochs on stabilizer rollouts warmed the same recurrent device-buffer policy, then one PPO update reached held-out hold-start `1.4425s` over `2.56s`. Held-out down-start stayed `0.0s`, so this is a learned top-stabilizer checkpoint, not a down-start solve and not a link-two unlock.
- One-link mixed swing-up command: `npm run train:six-pendulum:puffer-mjwarp:device-ppo-down-swingup-probe`
- One-link conservative mixed swing-up command: `npm run train:six-pendulum:puffer-mjwarp:device-ppo-down-swingup-conservative`
- One-link mixed swing-up result on 2026-06-10: both forceScale `120` probes use stronger potential/catch reward and coherent PPO buffers. Regular PPO still held pure down-start `0.0s`; conservative PPO also held pure down-start `0.0s`. The conservative run did produce a mixed-rollout near-catch flash of `0.8825s`, but held-out pure down-start remains unsolved and link two stays locked.
- Down-heavy one-link command: `npm run train:six-pendulum:puffer-mjwarp:device-ppo-down-heavy-conservative`
- Down-heavy one-link result on 2026-06-10: added mostly-down reset sampling, low-height pump reward, cart-boundary penalty, terminal penalty, sustained-hold shaping, action/std diagnostics, and cart drift metrics. Held-out pure down-start still held `0.0s`, so it does not count. Best stochastic down-heavy rollout reached `0.73s`; this is still subsecond. The new `cartAbsMax ~= 2.35` metric proves the learned policy is driving into the rail before a valid catch.
- Parallel lower-link diagnostics:
  - `npm run train:six-pendulum:puffer-mjwarp:device-ppo-link2-diagnostic`
  - `npm run train:six-pendulum:puffer-mjwarp:device-ppo-link3-diagnostic`
  - `npm run train:six-pendulum:puffer-mjwarp:device-ppo-link4-diagnostic`
- Parallel lower-link diagnostic result on 2026-06-10: links `2`, `3`, and `4` all stayed at held-out pure down-start `0.0s`. Link 2 reached hold-start `0.58s`; link 3 reached `0.1175s`; link 4 reached `0.0975s`. Down-start action usage saturated upward with link count: link 2 mean force about `97/120`, link 3 about `112/120`, link 4 about `108/120`. These runs are diagnostic dots only, not promotion.
- Long-horizon hold-start full command queued after probe: `npm run train:six-pendulum:puffer-mjwarp:device-ppo-hold-long` (`forceScale=32`, `nworld=128`, `rolloutSteps=1024`, `updates=50`, `evalInterval=5`)
- Random-horizon device smoke command: `npm run train:six-pendulum:puffer-mjwarp:device-rollout:random-horizon`
- Random-horizon device smoke artifact: `local-artifacts/outputs/training-checkpoints/puffer-mjwarp-device-rollout-random-horizon.json`
- Random-horizon result on 2026-06-10: one-link ran `32` worlds for `96` steps with per-world horizons sampled on-device between `16` and `32` steps; reset counts averaged above `4`, proving truncation/resets occurred without per-step CPU metric reads.
- Interpretation: reset sampling/writes, horizon sampling, action scaling, action-tensor ctrl writes, Torch/Warp recurrent actor-critic interop, reward, observation, strict score, cart terminal, truncation, held/max-held accumulation, potential-delta reward, fixed-shape PPO rollout-buffer writes, coherent pre-action PPO observations, fixed-buffer PPO epochs, and repeated collect-update-eval training are now reusable. A standalone MJWarp rollout can keep metrics device-side per step, and a recurrent Torch policy can consume current Warp observations without NumPy action writes. The remaining Yacine-speed blocker is replacing the local CPU trainer with real PufferPPO/MinGRU ownership of this path, high-volume sweeps, Modal/GPU execution, and GPU stream/capture discipline.
- Proof boundary: the device-rollout action source is a deterministic Warp scripted-action kernel for plumbing only. It is not a learned policy, not a score row, and not a solve.

Lower-link execution plan from current evidence:

1. Keep training on one link until held-out pure down-start reaches one continuous second. Do not unlock link two from hold-start, teacher, scripted rollout, or mixed-start wins.
2. Use the learned top stabilizer as initialization only. The next one-link task must bias harder toward pure down-start at `forceScale=120`; mixed curriculum already shows near-catch flashes but does not transfer to held-out down-start. A BC or teacher scaffold can warm the policy, but the score row remains zero until held-out pure down-start passes.
3. Use fixed episode lengths first. Turn on randomized per-world horizons only after learned whip or near-catch appears, because the source thread says it helped after the model was already scoring but getting lazy on fixed endings.
4. Train by sweeps, not single hand-tuning: many one-link PufferPPO/PufferNet-MinGRU runs, wallclock-vs-score dots, strict promotion gate.
5. Keep observations policy-usable only: cart state, previous action, relative/absolute angle encodings, velocities. Do not leak score-only terms.
6. Port the device rollout into the trainer path before GPU spend: the Puffer-facing step still returns NumPy arrays, while the device smoke proves the no-per-step-host-read path.
7. On GPU, capture only fixed-topology work. CPU Python and synchronization do not belong inside `wp.ScopedCapture`; graph capture comes after fixed-shape rollout and policy plumbing.

Source anchors:

- Yacine/kache thread: PufferPPO, Puffer MinGRU, MuJoCo Warp, random horizon only after whip, `3.6k` experiments, wallclock-vs-score.
- PufferLib docs: PuffeRL/PufferNet/Protein are the native path for fast training and sweeps.
- MuJoCo Warp docs: MJWarp is throughput-first for large parallel RL batches, not single-env latency.
- Warp APIC issue and CUDA graph docs: capture/replay requires device-side fixed work and no synchronization/query inside capture.
- Warp Torch interop: use `wp.to_torch`, `wp.from_torch`, or DLPack-style interop for policy tensors; do not route CUDA policy actions through NumPy.

Puffer-style sweep ledger:

- Command: `npm run six-pendulum:puffer-ledger`
- Markdown: `local-artifacts/outputs/sweeps/puffer-mjwarp-one-link-sweep-ledger.md`
- JSONL dots: `local-artifacts/outputs/sweeps/puffer-mjwarp-one-link-sweep-ledger.jsonl`
- Result: learned policy rows solved held-out one-link down-start `0/16`. The energy teacher scaffold reaches one-link down-start hold `1.387s` with strict score `99.04`, but it is explicitly not counted as a learned policy solve.
- Queued rows now match the Yacine experiment shape: PufferPPO, Puffer MinGRU/PufferNet, about `1m` params, MJWarp GPU batching, fixed horizon first, randomized episode length only after whip behavior appears, and link-two promotion only after held-out one-link down-start passes the one-second gate.
- Current blockers: Modal GPU execution is blocked by the workspace spend limit, and the current Puffer-facing env still returns NumPy observations/rewards/dones each step. The standalone device-rollout smoke proves the no-per-step-CPU-metric path; the action-buffer smoke proves a no-per-step-host-action-write interface; the recurrent Torch-policy smoke proves `wp.to_torch`/`wp.from_torch` observation/action/logprob/value plumbing; the PPO-update smoke proves those buffers can drive an optimizer step; the repeated device-buffer PPO trainer proves the same policy can collect, update, and be strictly evaluated over multiple iterations. The real PufferPPO/MJWarp path must move this into Puffer/MinGRU, run many updates/sweeps on GPU, and capture fixed-shape loops before claiming Yacine-like speed.

Phase 2, link scaling:

- Unlock link 2 only after 1-link held-out down-start holds for at least one second.
- Use the m1el lesson during pump: preserve bend order and penalize bend collapse only below a floor; do not reward straightness too early.
- Unlock links 3 through 6 only after the previous count passes the same held-out one-second gate.
- Add randomized episode length only after whip behavior exists and the policy is plateauing on short/fixed holds.
- Treat the eventual target as a sweep problem, not one long run: Yacine reported `3.6k` experiments and GP-selected high-compute hyperparameters.

Phase 3, acceptance:

- Public score counts only continuous strict hold time over one second.
- Final proof must include checkpoint JSON, training scatter/ledger, held-out validation replay, and hosted video.
- A model-based TVLQR trajectory can stay as external proof, but it cannot close the RL reproduction.

## Pezzza-Style One-Link Evolution Result

New trainer:

```bash
npm run train:six-pendulum:pezzza:smoke
```

Modal run:
(Modal run link omitted)

Artifact:
`local-artifacts/outputs/training-checkpoints/pezzza-evolution-1link-smoke.json`

Result:

- Algorithm: `modal-pezzza-style-evolution`.
- Policy: 24 open-loop force knots plus tiny MLP feedback.
- Inputs: cart position, cart velocity, sin/cos angle, angular velocity, previous action, normalized time.
- Movement terms: compensation, loss-of-balance penalty, gravity swing-up, whiplash, center staying.
- Curriculum: low gravity/high friction, medium gravity, near-normal, normal gravity.
- Final comparison score: strict score 187.95, mean hold 6.899 seconds, solved-one-second rate 1.0, about 721k simulated steps per second.

Interpretation:

This was the first one-link down-start solve in the vectorized analytic trainer. The checkpoint is now exported into the browser runtime and verified locally, so the current one-link gate passes. It proves the source direction: curriculum plus whiplash/recovery shaping solves the discovery problem that PPO/SAC/TD3 did not.

### Browser Runtime Proof

The browser policy was replaced with the Pezzza one-link checkpoint and the runtime now supports that checkpoint schema directly as `pezzzaKnotMlp`: time knots plus MLP feedback with inputs `cart x`, `cart velocity`, `sin(theta)`, `cos(theta)`, `angular velocity`, `previous action`, and normalized time.

Verification:

- Local deterministic replay over 16 down-start seeds: mean strict hold `5.681s`, every seed over `5.5s`.
- Playwright page proof on `http://localhost:3016/ailab/six-pendulum-cartpole`: held `8.043s`, score `95`.
- Screenshot: `local-artifacts/outputs/verification/pezzza-one-link-browser-schema-v1-clean/one-link-down-start-proof.png`
- Video: `local-artifacts/outputs/verification/pezzza-one-link-browser-schema-v1-clean/8197ecb6de75b2ce231a2bb617c2d6c6.webm`

The browser score stays zero until the one-second gate is crossed. Subsecond upright moments do not count.

### Ablations

Full-gravity, no curriculum:

```bash
npm run train:six-pendulum:pezzza:full-gravity-smoke
```

- Modal run: (Modal run link omitted)
- Result: strict score 0.0, mean hold 0.0 seconds.
- Interpretation: full difficulty from step zero does not find the behavior.

480 Hz curriculum:

```bash
npm run train:six-pendulum:pezzza:480hz-smoke
```

- Modal run: (Modal run link omitted)
- Result: strict score 164.60, mean hold 5.373 seconds, solved-one-second rate 0.949, about 784k simulated steps per second.
- Interpretation: 480 Hz works and is strong, but not clearly better enough to replace 240 Hz for smoke work.

Comparison report:

```bash
npm run six-pendulum:compare
```

- Markdown: `local-artifacts/outputs/sweeps/latest-six-pendulum-comparison.md`
- JSONL dots: `local-artifacts/outputs/sweeps/latest-six-pendulum-comparison.jsonl`

The report uses strict validation score for the Y axis. Subsecond holds remain diagnostics and do not count.

## Smoke Result

Command:

```bash
npm run train:six-pendulum:curriculum:smoke
```

Modal run:
(Modal run link omitted)

Artifact:
`local-artifacts/outputs/training-checkpoints/curriculum-smoke-lower-links.json`

Result:

- Link 1 validation: score 9.07, held 0.0533, whip 0.2369.
- Link 2 validation: score 0.0, held 0.00547, whip 0.2048.
- Final lower-link smoke validation: link 2, score 0.11, held 0.00530, whip 0.2054.

Interpretation:

The curriculum path is wired and produces real lower-link learning signal. It does not yet solve link 2 or six links. Next run should increase link 1 and link 2 generations before advancing.

## Hold-First Lower-Link Result

Command:

```bash
npm run train:six-pendulum:curriculum:lower
```

Modal run:
(Modal run link omitted)

Artifact:
`local-artifacts/outputs/training-checkpoints/curriculum-lower-links-v2.json`

Result:

- Link 1 hold validation: score 9.57, held 0.0909, whip 0.2961.
- Link 1 swing validation: score 72.73, held 0.1143, whip 0.3000.
- Link 2 hold validation: score 0.50, held 0.0106, whip 0.2155.
- Link 2 swing validation: score 0.17, held 0.00392, whip 0.2011.
- Final lower-link validation: link 2, score 0.14, held 0.00438, whip 0.2040.

Interpretation:

Hold-first training is the right direction for one link: it materially improves strict one-link hold and swing validation. The same time-knot feedback policy still does not carry to link 2. Do not deploy this as the public policy checkpoint. Validation-aware elite selection and distribution-mean handoff were implemented later, but link 2 still does not pass the strict down-start gate. The target lane is now recurrent PPO/Puffer-style training on MuJoCo/MJWarp.

Follow-up implementation:

- The trainer now re-scores top elites with strict validation before choosing the checkpoint to carry into the next stage.
- Stage handoff blends the validation checkpoint with the elite distribution mean instead of collapsing entirely to one best sampled policy.

Validation-aware smoke check:

- Modal run: (Modal run link omitted)
- Artifact: `local-artifacts/outputs/training-checkpoints/curriculum-validation-aware-check.json`
- Link 1 swing validation: score 38.98, held 0.0759, whip 0.2740.
- Link 2 hold validation: score 0.0, held 0.0181, whip 0.2454.
- Link 2 swing validation: score 0.60, held 0.00812, whip 0.2165.
- Final lower-link validation: link 2, score 0.22, held 0.00716, whip 0.2177.

## Validation-Aware Link-2 Search

Command:

```bash
modal run --write-result local-artifacts/outputs/training-checkpoints/curriculum-link2-search.json scripts/modal-train-six-pendulum-curriculum.py --smoke --max-stage-links 2 --generation-scale 8
```

Modal run:
(Modal run link omitted)

Artifact:
`local-artifacts/outputs/training-checkpoints/curriculum-link2-search.json`

Result:

- Elapsed: 453.119 seconds.
- Link 1 hold validation: score 34.14, held 0.1156, whip 0.3194.
- Link 1 swing validation: score 39.84, held 0.1162, whip 0.3067.
- Link 2 hold validation: score 0.13, held 0.0168, whip 0.2265.
- Link 2 swing validation: score 0.24, held 0.00424, whip 0.1819.
- Final lower-link validation: link 2, score 0.82, held 0.00476, whip 0.1840.
- Best transient link-2 hold checkpoint: generation 8, strict score 67.32, held 0.0192.
- Best transient link-2 swing checkpoint by held fraction: generation 56, held 0.0577, strict score 0.0.

Interpretation:

Validation-aware CEM can find brief two-link upright moments, but it still cannot make a stable two-link controller. More spend on the current time-knot feedback policy is likely a poor use of Modal credits unless the policy class changes.

## Next Run

Next target is two links, not six. Run order:

1. Extend the Pezzza vectorized trainer from one link to two links, keeping the same strict one-second gate.
2. Keep curriculum stages and whiplash/recovery terms; add randomized horizon only after two-link whip appears.
3. Add a recurrent Puffer-style PPO/MinGRU run only after the two-link environment and strict validator are fast enough to sweep.
4. Unlock link three only after two-link browser or MuJoCo validation holds for at least one second on held-out seeds.

MuJoCo path remains ready for the recurrent PPO branch:

```bash
npm run six-pendulum:mjcf
```

### Two-Link Trainer Scaffold

New vectorized chain trainer:

```bash
npm run train:six-pendulum:pezzza:chain2-smoke
```

Implementation:

- Script: `scripts/modal-train-six-pendulum-pezzza-chain.py`
- Policy: `pezzzaChainKnotMlp`, 32 time knots plus MLP feedback.
- Curriculum: one-link schema pretrain, two-link hold, two-link low gravity, two-link normal gravity.
- Gate: strict score stays zero unless mean max hold is at least one second.
- Promotion rule: do not unlock link three until two-link down-start validation has nonzero strict score, solved rate, and P10 hold.

First Modal attempt:

- Run: (Modal run link omitted)
- Result: blocked before function execution with `ResourceExhaustedError: Function call failed: workspace billing cycle spend limit reached`.
- Modal billing report command works: `modal billing report --for "this month" --json`.

Local MPS mini-run:

```bash
npm run train:six-pendulum:pezzza:chain2-local-mini
```

- Cold artifact: `local-artifacts/outputs/training-checkpoints/pezzza-chain-2link-local-mini.json`
- Warm-start artifact: `local-artifacts/outputs/training-checkpoints/pezzza-chain-2link-local-warm-mini.json`
- Branch-cut fix artifact: `local-artifacts/outputs/training-checkpoints/pezzza-chain-2link-local-angle-mini.json`
- Device: Apple MPS through local Torch.
- Cold elapsed: `83.892s`.
- Warm-start elapsed: `82.444s`.
- Branch-cut fix elapsed: `172.697s`.
- Warm start maps the proven one-link checkpoint into the two-link policy: interpolated time knots, first-link observation weights, softened second-link and relative-angle weights, and lower initial sigma.
- Stage selection is local to each curriculum stage so a high one-link pretrain score cannot block the two-link checkpoint from being carried forward.
- Two-link relative angles are now wrapped with `atan2(sin(delta), cos(delta))` for policy features, coupling, bend reward, and strict bend checks. Before this fix, a tiny bend near the `pi` branch cut could look like a false `~6.2 rad` bend.
- The trainer now re-validates elite candidates and carries the validation-best checkpoint between stages instead of raw shaped-selection best.
- Cold intermediate two-link hold signal: `0.512s` mean max hold, `0.135` solved-one-second rate from near-upright.
- Warm intermediate two-link hold signal after stage-local selection: `0.555s` mean max hold, `0.156` solved-one-second rate from near-upright.
- Warm best normal-gravity down-start flash: `0.0104s`.
- Warm final two-link down-start validation: strict score `0`, mean max hold `0.00241s`, solved-one-second rate `0`.
- Branch-cut fix final two-link down-start validation: strict score `0`, mean max hold `0.0768s`, solved-one-second rate `0`, whiplash `0.547s`.
- Branch-cut fix diagnostics: best two-link hold-start `0.643s`, best angle-curriculum hold `0.129s`.
- Comparison report now separates final down-start hold from best intermediate stage hold so subsecond stage progress cannot be mistaken for a solved run.
- Browser runtime now has a matching `pezzzaChainKnotMlp` path for future chain checkpoints: exact trainer feature order, wrapped relative angles, and chain physics matching the trainer. The public policy remains the solved one-link checkpoint until two-link passes strict down-start validation.
- Interpretation: local smoke proves the new two-link trainer runs without Modal. The one-link warm start plus wrapped-angle math improves actual down-start transfer from tiny flashes to measurable whiplash and `0.0768s` holds, but it still does not solve two-link down-start.

Planned implementation:

1. Generate a one-through-six-link MJCF chain with gravity 9.8 and zero hinge friction.
2. Start with one and two links only.
3. Train PPO with a small recurrent policy so the controller has memory for whip timing.
4. Add randomized episode length only after link-two whip appears.
5. Require strict held-time validation before unlocking link 3.
6. Keep the browser page as public proof and run ledger, not the source simulator.

MJCF scaffold:

- Command: `npm run six-pendulum:mjcf`
- Generated files: `app/ailab/six-pendulum-cartpole/mjcf/cartpole_1_link.xml` through `cartpole_6_link.xml`.
- Verified: each file parses as XML, has gravity `0 0 -9.8`, one cart motor, and the expected number of hinge joints.

## MuJoCo Recurrent PPO Smoke

Command:

```bash
npm run train:six-pendulum:mujoco-ppo:smoke
```

Modal run:
(Modal run link omitted)

Artifact:
`local-artifacts/outputs/training-checkpoints/mujoco-ppo-1link-smoke.json`

Result:

- Algorithm: `modal-mujoco-recurrent-ppo`.
- Simulator: MuJoCo 3.3.7.
- GPU: NVIDIA L4.
- Elapsed: 14.487 seconds after image build.
- One-link hold validation: score 0.0, held 0.0979, held P10 0.0604, return 108.20.
- One-link mixed validation: score 0.0, held 0.0646, held P10 0.0, return 71.79.

Interpretation:

The MuJoCo/PPO path is now wired and produces measurable hold signal, but the first recurrent PPO smoke does not keep the one-link pendulum upright through the final strict score gate. The next run should overfit one-link hold before enabling mixed starts or link two.

## One-Link Down-Start Gate

The acceptance rule changed after reviewing the failure mode: less than one consecutive second does not count, and no run starting near the top counts as solved. Validation must start from the hanging position and use only model-produced cart force.

UI changes:

- Default active count is one pendulum.
- Links two through six are locked until one-link down-start validation passes.
- Manual force, seed, kick, and top-start controls were removed.
- Canvas hold time resets to zero whenever strict score drops below 82.

### PPO Down-Start Smoke

Command:

```bash
npm run train:six-pendulum:mujoco-ppo:down-smoke
```

Modal run:
(Modal run link omitted)

Artifact:
`local-artifacts/outputs/training-checkpoints/mujoco-ppo-1link-down-smoke.json`

Result:

- Down-start final validation: score 0.0, held 0.00208, max held 0.0125 seconds, solved-one-second rate 0.0.
- Hold-start validation briefly reached max held 0.4766 seconds, but top-start does not count.

### SAC Down-Start Smoke

Command:

```bash
npm run train:six-pendulum:mujoco-sac:down-smoke
```

Modal run:
(Modal run link omitted)

Artifact:
`local-artifacts/outputs/training-checkpoints/mujoco-sac-1link-down-smoke.json`

Result:

- Final validation: score 0.0, held 0.000521, max held 0.00469 seconds, solved-one-second rate 0.0.
- Best transient checkpoint: 30k steps, max held 0.0125 seconds, solved-one-second rate 0.0.

### SAC Down-Start Full Run

Modal run:
(Modal run link omitted)

Artifact:
`local-artifacts/outputs/training-checkpoints/mujoco-sac-1link-down-full.json`

Result:

- Final validation: score 0.0, held 0.00100, max held 0.0141 seconds, solved-one-second rate 0.0.
- Best transient checkpoint: 245k steps, max held 0.0229 seconds, solved-one-second rate 0.0.

### SAC Stabilize-Then-Down Smoke

Command:

```bash
npm run train:six-pendulum:mujoco-sac:stabilize-smoke
```

Modal run:
(Modal run link omitted)

Artifact:
`local-artifacts/outputs/training-checkpoints/mujoco-sac-1link-stabilize-smoke.json`

Result:

- Final validation: score 0.0, held 0.00122, max held 0.0109 seconds, solved-one-second rate 0.0.
- Best transient checkpoint: hold phase at 30k steps, max held 0.0229 seconds, solved-one-second rate 0.0.

Interpretation:

The one-link Pezzza/browser policy is solved, but this MuJoCo SAC/TD3 lane did not solve one-link down-start. Plain down-start SAC learns some energy/height signal, but not the stabilizer. Stabilizer pretraining helps reach the same best transient much faster, but it still does not transfer into a one-second down-start hold. The next training change should use a real recurrent/off-policy sequence policy or explicit phase-conditioned curriculum while keeping the down-start one-second validation gate unchanged.
