#!/usr/bin/env python3
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import validate_six_pendulum_frozen_target as validator


def write_bank(path: Path, *, strict_steps: int, total_steps: int = 700, dt: float = 0.002) -> None:
    qpos = np.zeros((total_steps, 2), dtype=np.float64)
    qvel = np.zeros((total_steps, 2), dtype=np.float64)
    qpos[:, 1] = np.pi
    if strict_steps:
        qpos[-strict_steps:, 1] = 0.0
    metadata = {
        "schema": "test-trajectory-bank-v1",
        "profile": "thread-matched",
        "dt": dt,
        "links": 1,
    }
    np.savez(
        path,
        qpos=qpos,
        qvel=qvel,
        metadataJson=json.dumps(metadata),
    )


def write_fake_logged_solve_with_zero_controls(path: Path, *, total_steps: int = 700) -> None:
    qpos = np.zeros((total_steps, 2), dtype=np.float64)
    qvel = np.zeros((total_steps, 2), dtype=np.float64)
    qpos[:, 1] = 0.0
    qpos[0, 1] = np.pi
    metadata = {
        "schema": "test-trajectory-bank-v1",
        "profile": "thread-matched",
        "dt": 0.002,
        "links": 1,
    }
    np.savez(
        path,
        qpos=qpos,
        qvel=qvel,
        controls=np.zeros(total_steps, dtype=np.float64),
        metadataJson=json.dumps(metadata),
    )


class FrozenTargetValidatorTest(unittest.TestCase):
    def test_learned_policy_rollout_promotes_after_one_second_exact_down_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "solved.npz"
            write_bank(path, strict_steps=520)
            result = validator.validate_artifact(path, "learned-policy-rollout")
        self.assertEqual(result["status"], "pass")
        self.assertTrue(result["metrics"]["solvedOneSecond"])
        self.assertTrue(result["promotion"]["eligible"])
        self.assertGreaterEqual(result["metrics"]["maxHeldSeconds"], 1.0)

    def test_subsecond_hold_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "subsecond.npz"
            write_bank(path, strict_steps=400)
            result = validator.validate_artifact(path, "learned-policy-rollout")
        self.assertEqual(result["status"], "fail")
        self.assertIn("strict-hold-under-one-second", result["failures"])
        self.assertFalse(result["promotion"]["eligible"])

    def test_teacher_trajectory_can_pass_validation_but_not_promote_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "teacher.npz"
            write_bank(path, strict_steps=520)
            result = validator.validate_artifact(path, "teacher-trajectory")
        self.assertEqual(result["status"], "pass")
        self.assertTrue(result["metrics"]["solvedOneSecond"])
        self.assertFalse(result["promotion"]["eligible"])
        self.assertEqual(result["promotion"]["reason"], "teacher-trajectory-does-not-promote-learned-policy")

    def test_summary_json_cannot_promote_without_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.json"
            path.write_text(json.dumps({"maxHeldSeconds": 1.2, "links": 1}))
            result = validator.validate_artifact(path, "training-summary")
        self.assertEqual(result["status"], "fail")
        self.assertIn("summary-only-not-replayable", result["failures"])
        self.assertFalse(result["promotion"]["eligible"])

    def test_controls_force_mujoco_replay_instead_of_trusting_logged_states(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fake-solve.npz"
            write_fake_logged_solve_with_zero_controls(path)
            result = validator.validate_artifact(path, "learned-policy")
        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["metrics"]["scoredSource"], "mujoco-control-replay")
        self.assertIn("logged-states-differ-from-mujoco-replay", result["failures"])
        self.assertIn("strict-hold-under-one-second", result["failures"])
        self.assertFalse(result["promotion"]["eligible"])


if __name__ == "__main__":
    unittest.main()
