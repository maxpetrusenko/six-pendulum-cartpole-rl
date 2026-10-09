#!/usr/bin/env python3
"""Render a frozen-target trajectory bank to MP4."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation
import numpy as np


def wrap_pi(values: np.ndarray) -> np.ndarray:
    return (values + np.pi) % (2.0 * np.pi) - np.pi


def load_bank(path: Path) -> tuple[np.ndarray, dict]:
    with np.load(path, allow_pickle=False) as data:
        qpos = np.asarray(data["qpos"], dtype=np.float64)
        metadata = {}
        if "metadataJson" in data.files:
            metadata = json.loads(str(data["metadataJson"].item()))
    return qpos, metadata


def joint_positions(qpos_row: np.ndarray, link_length: float) -> tuple[np.ndarray, np.ndarray]:
    rel = wrap_pi(qpos_row[1:])
    abs_angles = wrap_pi(np.cumsum(rel))
    xs = [float(qpos_row[0])]
    ys = [0.0]
    for angle in abs_angles:
        xs.append(xs[-1] + np.sin(angle) * link_length)
        ys.append(ys[-1] + np.cos(angle) * link_length)
    return np.asarray(xs), np.asarray(ys)


def render(bank: Path, out: Path, fps: int, speed: float, link_length: float) -> dict:
    qpos, metadata = load_bank(bank)
    dt = float(metadata.get("dt", 0.002))
    links = int(metadata.get("links", qpos.shape[1] - 1))
    stride = max(1, int(round(speed / (fps * dt))))
    frames = np.arange(0, len(qpos), stride)
    if frames[-1] != len(qpos) - 1:
        frames = np.append(frames, len(qpos) - 1)

    x_min = float(np.min(qpos[:, 0]) - links * link_length - 0.5)
    x_max = float(np.max(qpos[:, 0]) + links * link_length + 0.5)
    y_min = -links * link_length - 0.7
    y_max = links * link_length + 0.7

    fig, ax = plt.subplots(figsize=(9, 5), dpi=140)
    ax.set_aspect("equal")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.axhline(0.0, color="#8a8f98", lw=0.8)
    (line,) = ax.plot([], [], "o-", color="#0f6b8d", lw=3.0, ms=5.5)
    cart = plt.Rectangle((0, -0.09), 0.32, 0.18, color="#d6452f")
    ax.add_patch(cart)
    label = ax.text(0.015, 0.96, "", transform=ax.transAxes, va="top", fontsize=10)
    ax.set_title("six-pendulum frozen-target replay")
    ax.set_xlabel("cart x")
    ax.set_ylabel("height")

    def update(frame_index: int):
        idx = int(frames[frame_index])
        xs, ys = joint_positions(qpos[idx], link_length)
        line.set_data(xs, ys)
        cart.set_x(float(qpos[idx, 0]) - 0.16)
        cart.set_y(-0.09)
        label.set_text(f"t={idx * dt:05.2f}s  cart={qpos[idx, 0]:+.2f}m")
        return line, cart, label

    animation = FuncAnimation(fig, update, frames=len(frames), blit=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=fps, codec="libx264", bitrate=2600, extra_args=["-pix_fmt", "yuv420p"])
    animation.save(out, writer=writer)
    plt.close(fig)
    return {
        "schema": "six-pendulum-frozen-trajectory-render-v1",
        "bank": str(bank),
        "output": str(out),
        "frames": int(len(frames)),
        "dt": dt,
        "fps": int(fps),
        "speed": float(speed),
        "durationSeconds": float(qpos.shape[0] * dt),
        "links": links,
        "cartMin": float(np.min(qpos[:, 0])),
        "cartMax": float(np.max(qpos[:, 0])),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--speed", type=float, default=2.0)
    parser.add_argument("--link-length", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = render(args.bank, args.out, args.fps, args.speed, args.link_length)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
