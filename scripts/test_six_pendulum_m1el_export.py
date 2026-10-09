#!/usr/bin/env python3
import unittest

import numpy as np

import export_six_pendulum_m1el_teacher_to_frozen_target as exporter


class M1elExportTest(unittest.TestCase):
    def test_absolute_angles_convert_to_relative_mujoco_joints(self):
        theta_abs = np.asarray([[0.1, 0.3, -0.2]], dtype=np.float64)
        thetad_abs = np.asarray([[1.0, 1.5, -0.5]], dtype=np.float64)
        qpos, qvel = exporter.absolute_to_mujoco_relative(
            theta_abs,
            thetad_abs,
            np.asarray([0.25]),
            np.asarray([-0.5]),
        )
        np.testing.assert_allclose(qpos[0], [0.25, 0.1, 0.2, -0.5], atol=1e-12)
        np.testing.assert_allclose(qvel[0], [-0.5, 1.0, 0.5, -2.0], atol=1e-12)

    def test_interpolation_preserves_matrix_shape(self):
        times = np.asarray([0.0, 1.0])
        values = np.asarray([[[1.0, 3.0]], [[5.0, 7.0]]])
        out = exporter.interp_columns(times, values, 0.5)
        self.assertEqual(out.shape, (1, 2))
        np.testing.assert_allclose(out, [[3.0, 5.0]])


if __name__ == "__main__":
    unittest.main()
