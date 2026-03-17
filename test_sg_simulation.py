"""
Tests for the SG water-level control simulation.

These tests exercise the main components of sg_simulation.py:
  - PIDController  (unit tests)
  - steam_flow_at / dsteam_flow_dt  (utility functions)
  - run_simulation  (integration test – verifies physical plausibility)
"""

import math
import importlib
import sys
import os
import unittest

# Make sure the parent directory is on the path so we can import the module
# even when running tests from a sub-directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sg_simulation as sim


class TestPIDController(unittest.TestCase):
    """Unit tests for the discrete PID controller."""

    def test_proportional_only(self):
        """Output equals Kp * error when Ki=Kd=0."""
        pid = sim.PIDController(kp=5.0, ki=0.0, kd=0.0, dt=1.0)
        out = pid.step(2.0)
        self.assertAlmostEqual(out, 10.0, places=6)

    def test_integral_accumulation(self):
        """Integral accumulates correctly over multiple steps."""
        pid = sim.PIDController(kp=0.0, ki=1.0, kd=0.0, dt=1.0)
        pid.step(1.0)
        pid.step(1.0)
        out = pid.step(1.0)
        # After 3 steps with error=1, integral = 3.0  → I-term = 3.0
        self.assertAlmostEqual(out, 3.0, places=6)

    def test_integral_anti_windup(self):
        """Integral stops accumulating when saturated=True."""
        pid = sim.PIDController(kp=0.0, ki=1.0, kd=0.0, dt=1.0,
                                i_clamp=None)
        pid.step(1.0)
        pid.step(1.0)  # integral = 2
        out = pid.step(1.0, saturated=True)   # integral should NOT grow
        self.assertAlmostEqual(out, 2.0, places=6)

    def test_integral_i_clamp(self):
        """Integral is clamped by i_clamp."""
        pid = sim.PIDController(kp=0.0, ki=1.0, kd=0.0, dt=1.0, i_clamp=2.0)
        for _ in range(10):
            pid.step(1.0)
        # Even after 10 steps, the I-term should not exceed ki * i_clamp = 2.0
        self.assertLessEqual(abs(pid._integral), 2.0 + 1e-9)

    def test_reset(self):
        """reset() zeroes all internal state."""
        pid = sim.PIDController(kp=1.0, ki=1.0, kd=1.0, dt=1.0)
        for _ in range(5):
            pid.step(1.0)
        pid.reset()
        out = pid.step(0.0)
        self.assertAlmostEqual(out, 0.0, places=6)

    def test_zero_error_zero_output(self):
        """With constant zero error, PID output stays zero."""
        pid = sim.PIDController(kp=10.0, ki=2.0, kd=5.0, dt=1.0)
        for _ in range(20):
            out = pid.step(0.0)
        self.assertAlmostEqual(out, 0.0, places=6)


class TestSteamFlowSchedule(unittest.TestCase):
    """Tests for the steam-flow schedule functions."""

    def test_nominal_before_disturbances(self):
        """Before any disturbance, steam flow equals FS_NOMINAL."""
        self.assertAlmostEqual(
            sim.steam_flow_at(0.0), sim.FS_NOMINAL, places=4)

    def test_nominal_between_disturbances(self):
        """Between disturbance windows, steam flow returns to FS_NOMINAL."""
        # Between D1 end (1200 s) and D2 start (1800 s)
        t_between = 1500.0
        self.assertAlmostEqual(
            sim.steam_flow_at(t_between), sim.FS_NOMINAL, places=4)

    def test_full_disturbance_in_steady_region(self):
        """At t well inside a disturbance window, the full delta is applied."""
        for t_start, t_end, delta in sim.DISTURBANCES:
            t_mid = (t_start + t_end) / 2.0
            if t_mid > t_start + sim.RAMP_TIME and t_mid < t_end - sim.RAMP_TIME:
                expected = sim.FS_NOMINAL + delta
                self.assertAlmostEqual(
                    sim.steam_flow_at(t_mid), expected, places=3)

    def test_ramp_monotone_at_start(self):
        """Steam flow ramps monotonically at the start of D1."""
        t_start, _, delta = sim.DISTURBANCES[0]
        values = [sim.steam_flow_at(t_start + i)
                  for i in range(int(sim.RAMP_TIME) + 1)]
        if delta > 0:
            for a, b in zip(values, values[1:]):
                self.assertLessEqual(a, b + 1e-6)
        else:
            for a, b in zip(values, values[1:]):
                self.assertGreaterEqual(a, b - 1e-6)

    def test_derivative_finite(self):
        """dsteam_flow_dt returns a finite number everywhere."""
        for t in range(0, int(sim.T_END), 10):
            val = sim.dsteam_flow_dt(float(t))
            self.assertTrue(math.isfinite(val))


class TestRunSimulation(unittest.TestCase):
    """Integration tests that run the full simulation and check plausibility."""

    @classmethod
    def setUpClass(cls):
        cls.time, cls.h, cls.fw, cls.fs_arr, cls.fw_sp_arr, cls.err = (
            sim.run_simulation()
        )

    def test_output_lengths(self):
        """All output arrays have the same length."""
        n = len(self.time)
        for arr in (self.h, self.fw, self.fs_arr, self.fw_sp_arr, self.err):
            self.assertEqual(len(arr), n)

    def test_initial_level_at_setpoint(self):
        """Simulation starts at the level setpoint."""
        self.assertAlmostEqual(self.h[0], sim.H_SETPOINT, places=6)

    def test_level_stays_within_physical_bounds(self):
        """Level deviation never exceeds ±1 m (physically plausible)."""
        self.assertLess(self.h.max(), 1.0,
                        "Max level exceeded +1 m")
        self.assertGreater(self.h.min(), -1.0,
                           "Min level dropped below -1 m")

    def test_feedwater_within_limits(self):
        """Actual feedwater flow stays within [FW_MIN, FW_MAX]."""
        self.assertGreaterEqual(self.fw.min(), sim.FW_MIN - 1e-6)
        self.assertLessEqual(self.fw.max(), sim.FW_MAX + 1e-6)

    def test_rms_error_acceptable(self):
        """RMS level error stays below 10 cm for the full run."""
        rms = math.sqrt((self.err ** 2).mean())
        self.assertLess(rms, 0.10, f"RMS error {rms * 100:.2f} cm exceeds 10 cm")

    def test_level_recovers_after_disturbance(self):
        """Level settles back near setpoint after each disturbance ends."""
        # After D1 ends (t = 1200 s) check that level is within 10 cm
        # after a recovery period of 300 s.
        dt = self.time[1] - self.time[0]
        t_check = 1200 + 300        # 5 min after D1 end
        idx = int(round(t_check / dt))
        self.assertLess(abs(self.h[idx]), 0.10,
                        f"Level {self.h[idx]*100:.1f} cm still off setpoint "
                        f"300 s after D1 ends")


if __name__ == "__main__":
    unittest.main()
