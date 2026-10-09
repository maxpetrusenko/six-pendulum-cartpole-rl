#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path

import numpy as np
import warp as wp


DEFAULT_ACTION_SCALE = 32.0
MAX_LINKS = 6
OBS_DIM = 3 + MAX_LINKS * 5
DEFAULT_OUTPUT = Path(
    "local-artifacts/outputs/training-checkpoints/puffer-mjwarp-gpu-score-kernel-smoke.json"
)


@wp.func
def wrap_angle(angle: float):
    two_pi = 6.283185307179586
    pi = 3.141592653589793
    shifted = angle + pi
    shifted = shifted - two_pi * wp.floor(shifted / two_pi)
    return shifted - pi


@wp.func
def rand_unit(seed: int, world: int, channel: int, reset_count: int):
    value = wp.sin(float(seed) * 12.9898 + float(world) * 78.233 + float(channel) * 37.719 + float(reset_count) * 19.19) * 43758.5453
    return value - wp.floor(value)


@wp.func
def rand_range(seed: int, world: int, channel: int, reset_count: int, low: float, high: float):
    return low + (high - low) * rand_unit(seed, world, channel, reset_count)


@wp.kernel
def reset_state_kernel(
    qpos: wp.array2d(dtype=wp.float32),
    qvel: wp.array2d(dtype=wp.float32),
    ctrl: wp.array2d(dtype=wp.float32),
    terminal: wp.array(dtype=wp.float32),
    truncation: wp.array(dtype=wp.float32),
    last_action: wp.array(dtype=wp.float32),
    elapsed: wp.array(dtype=wp.int32),
    held_steps: wp.array(dtype=wp.int32),
    max_held_steps: wp.array(dtype=wp.int32),
    horizon_steps: wp.array(dtype=wp.int32),
    reset_count: wp.array(dtype=wp.int32),
    links: int,
    pose_mode: int,
    seed: int,
    reset_all: int,
    random_horizon_enabled: int,
    min_horizon: int,
    max_horizon: int,
):
    i = wp.tid()
    should_reset = reset_all == 1 or terminal[i] > 0.5 or truncation[i] > 0.5
    if should_reset:
        count = reset_count[i] + 1
        reset_count[i] = count
        qpos[i, 0] = rand_range(seed, i, 0, count, -0.045, 0.045)
        qvel[i, 0] = rand_range(seed, i, 1, count, -0.05, 0.05)
        ctrl[i, 0] = 0.0
        last_action[i] = 0.0
        elapsed[i] = 0
        held_steps[i] = 0
        max_held_steps[i] = 0
        horizon_steps[i] = 0
        if random_horizon_enabled == 1:
            span = wp.max(0, max_horizon - min_horizon)
            horizon_steps[i] = min_horizon + int(rand_unit(seed, i, 70, count) * float(span + 1))

        selector = (i + count) - 4 * ((i + count) / 4)
        selector8 = (i + count) - 8 * ((i + count) / 8)
        down_noise = rand_range(seed, i, 2, count, -0.08, 0.08)
        for link in range(MAX_LINKS):
            if link < links:
                angle = 0.0
                velocity = 0.0
                if pose_mode == 1:
                    angle = rand_range(seed, i, 10 + link, count, -0.035, 0.035) * float(link + 1)
                    velocity = rand_range(seed, i, 30 + link, count, -0.06, 0.06)
                elif pose_mode == 2:
                    if selector == 0:
                        angle = 3.141592653589793 - float(link) * 0.05 + rand_range(seed, i, 10 + link, count, -0.08, 0.08)
                        velocity = rand_range(seed, i, 30 + link, count, -0.12, 0.12)
                    elif selector == 1:
                        angle = 0.75 * 3.141592653589793 - float(link) * 0.04 + rand_range(seed, i, 10 + link, count, -0.08, 0.08)
                        velocity = rand_range(seed, i, 30 + link, count, -1.45, -0.8)
                    elif selector == 2:
                        angle = 0.42 * 3.141592653589793 - float(link) * 0.03 + rand_range(seed, i, 10 + link, count, -0.07, 0.07)
                        velocity = rand_range(seed, i, 30 + link, count, -0.85, -0.25)
                    else:
                        angle = rand_range(seed, i, 10 + link, count, -0.035, 0.035) * float(link + 1)
                        velocity = rand_range(seed, i, 30 + link, count, -0.06, 0.06)
                elif pose_mode == 3:
                    if selector8 < 5:
                        angle = 3.141592653589793 - float(link) * 0.05 + down_noise
                        velocity = rand_range(seed, i, 30 + link, count, -0.18, 0.18)
                    elif selector8 == 5:
                        angle = 3.141592653589793 - float(link) * 0.05 + rand_range(seed, i, 10 + link, count, -0.18, 0.18)
                        velocity = rand_range(seed, i, 30 + link, count, 0.8, 1.7)
                    elif selector8 == 6:
                        angle = 0.78 * 3.141592653589793 - float(link) * 0.04 + rand_range(seed, i, 10 + link, count, -0.12, 0.12)
                        velocity = rand_range(seed, i, 30 + link, count, -1.55, -0.7)
                    else:
                        angle = rand_range(seed, i, 10 + link, count, -0.045, 0.045) * float(link + 1)
                        velocity = rand_range(seed, i, 30 + link, count, -0.08, 0.08)
                else:
                    angle = 3.141592653589793 - float(link) * 0.05 + down_noise
                    velocity = rand_range(seed, i, 30 + link, count, -0.08, 0.08)
                qpos[i, 1 + link] = angle
                qvel[i, 1 + link] = velocity


@wp.kernel
def sync_reset_potential_kernel(
    potential: wp.array(dtype=wp.float32),
    terminal: wp.array(dtype=wp.float32),
    truncation: wp.array(dtype=wp.float32),
    prev_potential: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    if terminal[i] > 0.5 or truncation[i] > 0.5:
        prev_potential[i] = potential[i]


@wp.kernel
def copy_float_kernel(
    source: wp.array(dtype=wp.float32),
    target: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    target[i] = source[i]


@wp.kernel
def action_to_ctrl_kernel(
    actions: wp.array2d(dtype=wp.float32),
    ctrl: wp.array2d(dtype=wp.float32),
    last_action: wp.array(dtype=wp.float32),
    action_scale: float,
):
    i = wp.tid()
    normalized = wp.min(wp.max(actions[i, 0], -1.0), 1.0)
    force = normalized * action_scale
    ctrl[i, 0] = force
    last_action[i] = force


@wp.kernel
def scripted_action_to_ctrl_kernel(
    ctrl: wp.array2d(dtype=wp.float32),
    last_action: wp.array(dtype=wp.float32),
    step_index: int,
    total_steps: int,
    action_scale: float,
):
    i = wp.tid()
    phase = float(step_index) / wp.max(1.0, float(total_steps - 1))
    force = wp.sin(phase * 12.566370614359172) * 18.0
    force = wp.min(wp.max(force, -action_scale), action_scale)
    ctrl[i, 0] = force
    last_action[i] = force


@wp.kernel
def action_buffer_to_ctrl_kernel(
    action_plan: wp.array(dtype=wp.float32),
    ctrl: wp.array2d(dtype=wp.float32),
    last_action: wp.array(dtype=wp.float32),
    step_index: int,
    nworld: int,
    action_scale: float,
):
    i = wp.tid()
    normalized = wp.min(wp.max(action_plan[step_index * nworld + i], -1.0), 1.0)
    force = normalized * action_scale
    ctrl[i, 0] = force
    last_action[i] = force


@wp.kernel
def action_vector_to_ctrl_kernel(
    action_vector: wp.array(dtype=wp.float32),
    ctrl: wp.array2d(dtype=wp.float32),
    last_action: wp.array(dtype=wp.float32),
    action_scale: float,
):
    i = wp.tid()
    normalized = wp.min(wp.max(action_vector[i], -1.0), 1.0)
    force = normalized * action_scale
    ctrl[i, 0] = force
    last_action[i] = force


@wp.kernel
def score_obs_kernel(
    qpos: wp.array2d(dtype=wp.float32),
    qvel: wp.array2d(dtype=wp.float32),
    last_action: wp.array(dtype=wp.float32),
    links: int,
    action_scale: float,
    terminal_boundary: float,
    obs: wp.array(dtype=wp.float32),
    reward: wp.array(dtype=wp.float32),
    strict_score: wp.array(dtype=wp.float32),
    potential: wp.array(dtype=wp.float32),
    mean_tip_height: wp.array(dtype=wp.float32),
    energy_error: wp.array(dtype=wp.float32),
    catch_basin: wp.array(dtype=wp.float32),
    near_top_fast: wp.array(dtype=wp.float32),
    whip: wp.array(dtype=wp.float32),
    terminal: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    x = qpos[i, 0]
    xvel = qvel[i, 0]
    running_angle = 0.0
    running_velocity = 0.0
    max_upright_error = 0.0
    max_bend_error = 0.0
    sum_upright_error = 0.0
    sum_speed = 0.0
    sum_tip_height = 0.0
    sum_energy = 0.0
    weighted_upright_error = 0.0
    sum_abs_velocity = 0.0
    terminal_upright_error = 0.0

    for link in range(MAX_LINKS):
        obs_base = i * OBS_DIM + 3 + link * 5
        if link < links:
            relative = wrap_angle(qpos[i, 1 + link])
            running_angle = wrap_angle(running_angle + relative)
            running_velocity += qvel[i, 1 + link]
            abs_angle = wp.abs(running_angle)
            abs_velocity = wp.abs(running_velocity)
            height = (wp.cos(running_angle) + 1.0) * 0.5
            energy = 0.5 * (running_velocity * 0.28) * (running_velocity * 0.28) + wp.cos(running_angle)

            max_upright_error = wp.max(max_upright_error, abs_angle)
            if link > 0:
                max_bend_error = wp.max(max_bend_error, wp.abs(relative))
            sum_upright_error += abs_angle
            sum_speed += abs_velocity
            sum_tip_height += height
            sum_energy += energy
            weighted_upright_error += abs_angle * float(link + 1)
            sum_abs_velocity += abs_velocity
            terminal_upright_error = abs_angle

            obs[obs_base] = wp.sin(running_angle)
            obs[obs_base + 1] = wp.cos(running_angle)
            obs[obs_base + 2] = wp.sin(relative)
            obs[obs_base + 3] = wp.cos(relative)
            obs[obs_base + 4] = running_velocity / 8.0

    link_count = float(links)
    mean_upright_error = sum_upright_error / link_count
    mean_speed = sum_speed / link_count
    height = sum_tip_height / link_count
    energy_gap = wp.abs(sum_energy / link_count - 1.0)

    near_top = max_upright_error < 0.16 and max_bend_error < 0.14
    in_catch_basin = max_upright_error < 0.55 and max_bend_error < 0.34 and mean_speed < 3.2
    is_near_top_fast = max_upright_error < 0.72 and mean_speed > 1.2
    is_whip = terminal_upright_error < 0.65 and mean_speed > 1.0

    score = 0.0
    if near_top:
        score = 100.0 - weighted_upright_error * 12.0 - max_bend_error * 30.0 - sum_abs_velocity * 2.5 - wp.abs(x) * 8.0
        score = wp.min(wp.max(score, 0.0), 100.0)

    dense_alignment = wp.exp(-mean_upright_error * 1.15 - max_bend_error * 2.0 - mean_speed * 0.08)
    action_fraction = last_action[i] / action_scale
    low_height = 1.0 - height
    pump_reward = wp.min(wp.abs(xvel) * low_height, 3.0) * 0.08
    pump_reward += wp.min(mean_speed * low_height, 3.0) * 0.04
    shaped_reward = height * 0.3 + dense_alignment * 0.1 + (score / 100.0) * (score / 100.0) * 2.5
    shaped_reward += pump_reward
    if is_whip:
        shaped_reward += 0.2
    if is_near_top_fast:
        shaped_reward += 0.35
    if in_catch_basin:
        shaped_reward += 1.0
    cart_fraction = wp.abs(x) / terminal_boundary
    shaped_reward -= cart_fraction * cart_fraction * 0.25
    if cart_fraction > 0.8:
        shaped_reward -= (cart_fraction - 0.8) * 1.5
    shaped_reward -= action_fraction * action_fraction * 0.015

    base = i * OBS_DIM
    obs[base] = x
    obs[base + 1] = xvel / 5.0
    obs[base + 2] = action_fraction

    reward[i] = shaped_reward
    strict_score[i] = score
    potential[i] = height - 0.02 * energy_gap - 0.01 * wp.abs(x)
    mean_tip_height[i] = height
    energy_error[i] = energy_gap
    catch_basin[i] = 1.0 if in_catch_basin else 0.0
    near_top_fast[i] = 1.0 if is_near_top_fast else 0.0
    whip[i] = 1.0 if is_whip else 0.0
    terminal[i] = 1.0 if wp.abs(x) > terminal_boundary else 0.0


@wp.kernel
def post_step_kernel(
    shaped_reward: wp.array(dtype=wp.float32),
    potential: wp.array(dtype=wp.float32),
    strict_score: wp.array(dtype=wp.float32),
    terminal: wp.array(dtype=wp.float32),
    pose_hold: int,
    horizon: int,
    prev_potential: wp.array(dtype=wp.float32),
    elapsed: wp.array(dtype=wp.int32),
    held_steps: wp.array(dtype=wp.int32),
    max_held_steps: wp.array(dtype=wp.int32),
    horizon_steps: wp.array(dtype=wp.int32),
    rollout_max_held_steps: wp.array(dtype=wp.int32),
    rollout_max_strict_score: wp.array(dtype=wp.float32),
    final_reward: wp.array(dtype=wp.float32),
    truncation: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    step_count = elapsed[i] + 1
    elapsed[i] = step_count

    reward_value = shaped_reward[i]
    hit_terminal = terminal[i] > 0.5
    if hit_terminal:
        reward_value -= 3.0
    elif pose_hold == 0:
        delta = wp.min(wp.max(potential[i] - prev_potential[i], -0.25), 0.35)
        reward_value = reward_value + delta * 2.0
    prev_potential[i] = potential[i]

    current_held = 0
    if strict_score[i] > 82.0:
        current_held = held_steps[i] + 1
    if current_held > 0:
        reward_value += wp.min(float(current_held), 400.0) * 0.001
    if current_held >= 40:
        reward_value += 0.2
    if current_held >= 160:
        reward_value += 0.4
    final_reward[i] = reward_value
    held_steps[i] = current_held
    max_held_steps[i] = wp.max(max_held_steps[i], current_held)
    rollout_max_held_steps[i] = wp.max(rollout_max_held_steps[i], max_held_steps[i])
    rollout_max_strict_score[i] = wp.max(rollout_max_strict_score[i], strict_score[i])

    world_horizon = horizon
    if horizon_steps[i] > 0:
        world_horizon = horizon_steps[i]
    done = terminal[i] > 0.5 or step_count >= world_horizon
    truncation[i] = 1.0 if step_count >= world_horizon and terminal[i] <= 0.5 else 0.0
    if done:
        elapsed[i] = 0
        held_steps[i] = 0
        max_held_steps[i] = 0


@wp.kernel
def record_rollout_obs_kernel(
    obs: wp.array(dtype=wp.float32),
    step_index: int,
    nworld: int,
    obs_buffer: wp.array(dtype=wp.float32),
):
    world, feature = wp.tid()
    src = world * OBS_DIM + feature
    dst = (step_index * nworld + world) * OBS_DIM + feature
    obs_buffer[dst] = obs[src]


@wp.kernel
def record_rollout_scalars_kernel(
    final_reward: wp.array(dtype=wp.float32),
    terminal: wp.array(dtype=wp.float32),
    truncation: wp.array(dtype=wp.float32),
    last_action: wp.array(dtype=wp.float32),
    step_index: int,
    nworld: int,
    reward_buffer: wp.array(dtype=wp.float32),
    terminal_buffer: wp.array(dtype=wp.float32),
    truncation_buffer: wp.array(dtype=wp.float32),
    action_buffer: wp.array(dtype=wp.float32),
):
    world = wp.tid()
    dst = step_index * nworld + world
    reward_buffer[dst] = final_reward[world]
    terminal_buffer[dst] = terminal[world]
    truncation_buffer[dst] = truncation[world]
    action_buffer[dst] = last_action[world]


@wp.kernel
def record_policy_scalars_kernel(
    normalized_action: wp.array(dtype=wp.float32),
    logprob: wp.array(dtype=wp.float32),
    value: wp.array(dtype=wp.float32),
    step_index: int,
    nworld: int,
    normalized_action_buffer: wp.array(dtype=wp.float32),
    logprob_buffer: wp.array(dtype=wp.float32),
    value_buffer: wp.array(dtype=wp.float32),
):
    world = wp.tid()
    dst = step_index * nworld + world
    normalized_action_buffer[dst] = normalized_action[world]
    logprob_buffer[dst] = logprob[world]
    value_buffer[dst] = value[world]


def deterministic_batch(nworld: int, action_scale: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    qpos = np.zeros((nworld, 1 + MAX_LINKS), dtype=np.float32)
    qvel = np.zeros((nworld, 1 + MAX_LINKS), dtype=np.float32)
    qpos[:, 0] = rng.uniform(-0.35, 0.35, nworld).astype(np.float32)
    qpos[:, 1:] = rng.uniform(-3.8, 3.8, (nworld, MAX_LINKS)).astype(np.float32)
    qvel[:, 0] = rng.uniform(-1.4, 1.4, nworld).astype(np.float32)
    qvel[:, 1:] = rng.uniform(-7.5, 7.5, (nworld, MAX_LINKS)).astype(np.float32)
    last_action = rng.uniform(-action_scale, action_scale, nworld).astype(np.float32)

    anchors = np.asarray([-0.15, -0.01, 0.0, 0.01, 0.15, 0.54, 0.64, 0.71, np.pi], dtype=np.float32)
    for index, angle in enumerate(anchors[:nworld]):
        qpos[index, 0] = np.float32((index - 4) * 0.035)
        qpos[index, 1] = angle
        qvel[index, 1] = np.float32((index - 4) * 0.55)
        qpos[index, 2:] = np.float32(angle / 8.0)
        qvel[index, 2:] = np.float32((index - 4) * 0.07)
        last_action[index] = np.float32((index - 4) * action_scale / 8.0)
    if nworld > 2:
        qpos[-2, 0] = np.float32(-2.7)
        qpos[-1, 0] = np.float32(2.6)

    return qpos, qvel, last_action


class WarpScoreKernel:
    def __init__(
        self,
        nworld: int,
        links: int,
        action_scale: float = DEFAULT_ACTION_SCALE,
        device: str = "cpu",
        terminal_boundary: float = 2.35,
    ):
        self.nworld = int(nworld)
        self.links = max(1, min(MAX_LINKS, int(links)))
        self.action_scale = float(action_scale)
        self.device = device
        self.terminal_boundary = float(terminal_boundary)
        self.last_action_wp = wp.zeros(self.nworld, dtype=wp.float32, device=self.device)
        self.obs_wp = wp.zeros(self.nworld * OBS_DIM, dtype=wp.float32, device=self.device)
        self.reward_wp = wp.zeros(self.nworld, dtype=wp.float32, device=self.device)
        self.strict_score_wp = wp.zeros(self.nworld, dtype=wp.float32, device=self.device)
        self.potential_wp = wp.zeros(self.nworld, dtype=wp.float32, device=self.device)
        self.mean_tip_height_wp = wp.zeros(self.nworld, dtype=wp.float32, device=self.device)
        self.energy_error_wp = wp.zeros(self.nworld, dtype=wp.float32, device=self.device)
        self.catch_basin_wp = wp.zeros(self.nworld, dtype=wp.float32, device=self.device)
        self.near_top_fast_wp = wp.zeros(self.nworld, dtype=wp.float32, device=self.device)
        self.whip_wp = wp.zeros(self.nworld, dtype=wp.float32, device=self.device)
        self.terminal_wp = wp.zeros(self.nworld, dtype=wp.float32, device=self.device)
        self.action_wp = wp.zeros((self.nworld, 1), dtype=wp.float32, device=self.device)
        self.prev_potential_wp = wp.zeros(self.nworld, dtype=wp.float32, device=self.device)
        self.elapsed_wp = wp.zeros(self.nworld, dtype=wp.int32, device=self.device)
        self.held_steps_wp = wp.zeros(self.nworld, dtype=wp.int32, device=self.device)
        self.max_held_steps_wp = wp.zeros(self.nworld, dtype=wp.int32, device=self.device)
        self.horizon_steps_wp = wp.zeros(self.nworld, dtype=wp.int32, device=self.device)
        self.reset_count_wp = wp.zeros(self.nworld, dtype=wp.int32, device=self.device)
        self.final_reward_wp = wp.zeros(self.nworld, dtype=wp.float32, device=self.device)
        self.truncation_wp = wp.zeros(self.nworld, dtype=wp.float32, device=self.device)
        self.rollout_max_held_steps_wp = wp.zeros(self.nworld, dtype=wp.int32, device=self.device)
        self.rollout_max_strict_score_wp = wp.zeros(self.nworld, dtype=wp.float32, device=self.device)

    def reset_worlds(
        self,
        qpos_wp,
        qvel_wp,
        ctrl_wp,
        pose: str,
        seed: int,
        reset_all: bool = False,
        synchronize: bool = True,
        random_horizon: bool = False,
        min_horizon: int = 160,
        max_horizon: int = 512,
    ):
        pose_mode = 1 if pose == "hold" else 2 if pose == "mixed" else 3 if pose == "down-heavy" else 0
        wp.launch(
            reset_state_kernel,
            dim=self.nworld,
            inputs=[
                qpos_wp,
                qvel_wp,
                ctrl_wp,
                self.terminal_wp,
                self.truncation_wp,
                self.last_action_wp,
                self.elapsed_wp,
                self.held_steps_wp,
                self.max_held_steps_wp,
                self.horizon_steps_wp,
                self.reset_count_wp,
                int(self.links),
                int(pose_mode),
                int(seed),
                1 if reset_all else 0,
                1 if random_horizon else 0,
                int(min_horizon),
                int(max_horizon),
            ],
            device=self.device,
        )
        if synchronize:
            wp.synchronize()

    def apply_actions(self, actions: np.ndarray, ctrl_wp) -> np.ndarray:
        self.action_wp.assign(np.asarray(actions, dtype=np.float32).reshape(self.nworld, 1))
        wp.launch(
            action_to_ctrl_kernel,
            dim=self.nworld,
            inputs=[self.action_wp, ctrl_wp, self.last_action_wp, float(self.action_scale)],
            device=self.device,
        )
        wp.synchronize()
        return self.last_action_wp.numpy()

    def apply_scripted_actions(self, ctrl_wp, step_index: int, total_steps: int, synchronize: bool = False):
        wp.launch(
            scripted_action_to_ctrl_kernel,
            dim=self.nworld,
            inputs=[ctrl_wp, self.last_action_wp, int(step_index), int(total_steps), float(self.action_scale)],
            device=self.device,
        )
        if synchronize:
            wp.synchronize()

    def apply_action_buffer(self, action_plan_wp, ctrl_wp, step_index: int, synchronize: bool = False):
        wp.launch(
            action_buffer_to_ctrl_kernel,
            dim=self.nworld,
            inputs=[action_plan_wp, ctrl_wp, self.last_action_wp, int(step_index), int(self.nworld), float(self.action_scale)],
            device=self.device,
        )
        if synchronize:
            wp.synchronize()

    def apply_action_vector(self, action_vector_wp, ctrl_wp, synchronize: bool = False):
        wp.launch(
            action_vector_to_ctrl_kernel,
            dim=self.nworld,
            inputs=[action_vector_wp, ctrl_wp, self.last_action_wp, float(self.action_scale)],
            device=self.device,
        )
        if synchronize:
            wp.synchronize()

    def score_from_warp_arrays(self, qpos_wp, qvel_wp, last_action: np.ndarray) -> dict:
        self.last_action_wp.assign(np.asarray(last_action, dtype=np.float32).reshape(self.nworld))
        return self.score_from_current_last_action(qpos_wp, qvel_wp)

    def score_from_current_last_action(self, qpos_wp, qvel_wp) -> dict:
        self.score_device(qpos_wp, qvel_wp, synchronize=True)
        return self.to_numpy()

    def score_device(self, qpos_wp, qvel_wp, synchronize: bool = False):
        self.obs_wp.zero_()
        wp.launch(
            score_obs_kernel,
            dim=self.nworld,
            inputs=[
                qpos_wp,
                qvel_wp,
                self.last_action_wp,
                int(self.links),
                float(self.action_scale),
                float(self.terminal_boundary),
                self.obs_wp,
                self.reward_wp,
                self.strict_score_wp,
                self.potential_wp,
                self.mean_tip_height_wp,
                self.energy_error_wp,
                self.catch_basin_wp,
                self.near_top_fast_wp,
                self.whip_wp,
                self.terminal_wp,
            ],
            device=self.device,
        )
        if synchronize:
            wp.synchronize()

    def reset_rollout_state(
        self,
        prev_potential: np.ndarray,
        elapsed: np.ndarray | None = None,
        held_steps: np.ndarray | None = None,
        max_held_steps: np.ndarray | None = None,
    ):
        self.prev_potential_wp.assign(np.asarray(prev_potential, dtype=np.float32).reshape(self.nworld))
        zeros = np.zeros(self.nworld, dtype=np.int32)
        self.elapsed_wp.assign(zeros if elapsed is None else np.asarray(elapsed, dtype=np.int32).reshape(self.nworld))
        self.held_steps_wp.assign(zeros if held_steps is None else np.asarray(held_steps, dtype=np.int32).reshape(self.nworld))
        self.max_held_steps_wp.assign(zeros if max_held_steps is None else np.asarray(max_held_steps, dtype=np.int32).reshape(self.nworld))

    def sync_reset_potential(self, synchronize: bool = True):
        wp.launch(
            sync_reset_potential_kernel,
            dim=self.nworld,
            inputs=[self.potential_wp, self.terminal_wp, self.truncation_wp, self.prev_potential_wp],
            device=self.device,
        )
        if synchronize:
            wp.synchronize()

    def initialize_prev_potential_from_current(self, synchronize: bool = False):
        wp.launch(
            copy_float_kernel,
            dim=self.nworld,
            inputs=[self.potential_wp, self.prev_potential_wp],
            device=self.device,
        )
        if synchronize:
            wp.synchronize()

    def post_step(self, pose_hold: bool, horizon: int) -> dict:
        self.post_step_device(pose_hold, horizon, synchronize=True)
        return {
            "reward": self.final_reward_wp.numpy(),
            "truncation": self.truncation_wp.numpy(),
            "elapsed": self.elapsed_wp.numpy(),
            "heldSteps": self.held_steps_wp.numpy(),
            "maxHeldSteps": self.max_held_steps_wp.numpy(),
            "prevPotential": self.prev_potential_wp.numpy(),
        }

    def post_step_device(self, pose_hold: bool, horizon: int, synchronize: bool = False):
        wp.launch(
            post_step_kernel,
            dim=self.nworld,
            inputs=[
                self.reward_wp,
                self.potential_wp,
                self.strict_score_wp,
                self.terminal_wp,
                1 if pose_hold else 0,
                int(horizon),
                self.prev_potential_wp,
                self.elapsed_wp,
                self.held_steps_wp,
                self.max_held_steps_wp,
                self.horizon_steps_wp,
                self.rollout_max_held_steps_wp,
                self.rollout_max_strict_score_wp,
                self.final_reward_wp,
                self.truncation_wp,
            ],
            device=self.device,
        )
        if synchronize:
            wp.synchronize()

    def to_numpy(self) -> dict:
        return {
            "observation": self.obs_wp.numpy().reshape(self.nworld, OBS_DIM),
            "reward": self.reward_wp.numpy(),
            "strictScore": self.strict_score_wp.numpy(),
            "meanTipHeight": self.mean_tip_height_wp.numpy(),
            "energyError": self.energy_error_wp.numpy(),
            "potential": self.potential_wp.numpy(),
            "catchBasin": self.catch_basin_wp.numpy(),
            "nearTopFast": self.near_top_fast_wp.numpy(),
            "whip": self.whip_wp.numpy(),
            "terminal": self.terminal_wp.numpy(),
        }


def run_kernel(qpos: np.ndarray, qvel: np.ndarray, last_action: np.ndarray, links: int, action_scale: float, device: str) -> dict:
    runner = WarpScoreKernel(qpos.shape[0], links, action_scale, device)
    qpos_wp = wp.array(qpos, dtype=wp.float32, device=device)
    qvel_wp = wp.array(qvel, dtype=wp.float32, device=device)
    return runner.score_from_warp_arrays(qpos_wp, qvel_wp, last_action)


def max_abs_error(expected: np.ndarray, actual: np.ndarray) -> float:
    return float(np.max(np.abs(expected.astype(np.float32) - actual.astype(np.float32))))


def main():
    from six_pendulum_mjwarp_env import score_batch

    parser = argparse.ArgumentParser(description="Parity-test six-pendulum score/obs math in a Warp kernel.")
    parser.add_argument("--nworld", type=int, default=512)
    parser.add_argument("--seed", type=int, default=426210)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--action-scale", type=float, default=DEFAULT_ACTION_SCALE)
    parser.add_argument("--max-links", type=int, default=MAX_LINKS)
    parser.add_argument("--tolerance", type=float, default=2e-5)
    parser.add_argument("--write-result", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    started = time.time()
    wp.init()
    qpos, qvel, last_action = deterministic_batch(args.nworld, args.action_scale, args.seed)
    keys = ["observation", "reward", "strictScore", "meanTipHeight", "energyError", "potential", "catchBasin", "nearTopFast", "whip", "terminal"]
    link_errors = []
    max_links = max(1, min(MAX_LINKS, args.max_links))
    for links in range(1, max_links + 1):
        expected = score_batch(qpos, qvel, last_action, links=links, action_scale=args.action_scale)
        expected["terminal"] = (np.abs(qpos[:, 0]) > 2.35).astype(np.float32)
        actual = run_kernel(qpos, qvel, last_action, links, args.action_scale, args.device)
        errors = {key: max_abs_error(expected[key], actual[key]) for key in keys}
        link_errors.append(
            {
                "links": links,
                "maxError": max(errors.values()),
                "errors": errors,
            }
        )

    max_error = max(entry["maxError"] for entry in link_errors)
    passed = max_error <= args.tolerance
    result = {
        "schema": "six-pendulum-mjwarp-score-kernel-smoke-v2",
        "status": "passed" if passed else "failed",
        "createdAtUnix": time.time(),
        "elapsedSeconds": time.time() - started,
        "linksCovered": list(range(1, max_links + 1)),
        "nworld": args.nworld,
        "device": args.device,
        "warpVersion": getattr(wp, "__version__", "unknown"),
        "actionScale": args.action_scale,
        "tolerance": args.tolerance,
        "maxError": max_error,
        "linkErrors": link_errors,
        "coveredTerms": [
            "observation",
            "reward",
            "strictScore",
            "potential",
            "meanTipHeight",
            "energyError",
            "catchBasin",
            "nearTopFast",
            "whip",
            "terminal",
        ],
        "integrationStatus": "score, observation, cart terminal, action scaling, ctrl write, held-step, truncation, max-held, potential-delta reward, reset sampling, and reset writes have Warp kernels; CPU done trigger, CPU metric copies, and Puffer rollout integration still pending",
    }
    args.write_result.parent.mkdir(parents=True, exist_ok=True)
    args.write_result.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
