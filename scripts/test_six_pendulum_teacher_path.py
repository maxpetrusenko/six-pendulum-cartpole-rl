#!/usr/bin/env python3
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import audit_six_pendulum_mjcf_profile as audit_profile
import solve_six_pendulum_mujoco_ilqr_teacher as ilqr_teacher


class SixPendulumTeacherPathTest(unittest.TestCase):
    def test_thread_matched_profile_audits_canonical_physics(self):
        result = audit_profile.run_audit("thread-matched")
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["canonical"]["maxLinks"], 7)
        self.assertEqual(result["canonical"]["linkLength"], 0.5)
        self.assertEqual(result["canonical"]["linkMass"], 0.15)
        self.assertEqual(result["canonical"]["hingeDamping"], 0.05)
        self.assertEqual(result["canonical"]["forceCap"], 120.0)
        self.assertEqual(result["canonical"]["dt"], 0.002)

    def test_ilqr_teacher_exports_bc_ready_bank(self):
        with tempfile.TemporaryDirectory() as tmp:
            bank = Path(tmp) / "teacher.npz"
            summary = Path(tmp) / "teacher.json"
            result = ilqr_teacher.export_teacher(
                links=1,
                profile="thread-matched",
                horizon=16,
                iterations=0,
                seed=426210,
                write_trajectories=bank,
                write_summary=summary,
            )
            self.assertTrue(bank.exists())
            self.assertTrue(summary.exists())
            self.assertEqual(result["schema"], "six-pendulum-mujoco-ilqr-teacher-bank-v1")
            self.assertEqual(result["profile"], "thread-matched")
            self.assertEqual(result["forceScale"], 120.0)
            with np.load(bank) as data:
                self.assertEqual(data["observations"].shape, (16, 33))
                self.assertEqual(data["actions"].shape, (16, 1))
                self.assertEqual(data["qpos"].shape, (16, 2))
                self.assertEqual(data["qvel"].shape, (16, 2))
                self.assertLessEqual(float(np.max(np.abs(data["actions"]))), 1.0)
                metadata = json.loads(str(data["metadataJson"]))
            self.assertTrue(metadata["gates"]["ppoBlockedUntilTeacherExportExists"])
            self.assertEqual(metadata["gates"]["runPodMaxWallclockMinutes"], 60)


if __name__ == "__main__":
    unittest.main()
