"""
SG Water Level Control Simulation (SG 水位控制仿真)

Simulates a pressurized-water-reactor steam generator (SG) water-level
control system using a three-element feedwater controller.

Physics model
-------------
The steam generator water level is governed by a simplified mass-balance:

    dh/dt = (Fw - Fs) / A + Ks * dFs/dt

where
    h   : water level [m], measured from the normal operating level (0 m)
    Fw  : feedwater (FW) mass flow rate [kg/s]
    Fs  : steam (main-steam) mass flow rate [kg/s]
    A   : effective SG cross-sectional area [m²]
    Ks  : shrink-and-swell coefficient [m·s/kg]
         (positive: level swells when steam demand increases)

The "shrink-and-swell" term models the transient level change caused by the
change in void fraction when steam pressure varies: an increase in steam
demand briefly raises the level before the water inventory falls.

Three-element controller
------------------------
The setpoint for feedwater flow Fw_sp is computed as:

    error   = h_sp - h(t)          (level error)
    Fw_sp   = Fs + Kp*error + Ki*∫error dt + Kd*d(error)/dt

i.e. the steam flow provides feed-forward compensation while the PID
block corrects for level error.

Feedwater valve dynamics are approximated by a first-order lag:

    τv * dFw/dt = Fw_sp - Fw

Usage
-----
    python sg_simulation.py

Outputs
-------
* sg_simulation_result.png   – time-history plots (level, flows, error)
* Prints a brief summary to stdout
"""

import math
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend – no display required
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------

DT = 1.0           # integration time-step [s]
T_END = 3600.0     # total simulation time [s]  (60 min)

# SG physics – effective parameter that reflects the large secondary-side
# water inventory; a 1 kg/s imbalance changes level at 1/A_SG m/s.
A_SG = 500.0       # effective cross-sectional area [m²]
KSS = 0.02         # shrink-and-swell coefficient  [m·s/kg]

# Normal operating steam flow [kg/s]
FS_NOMINAL = 500.0

# Level setpoint [m]  (0 = normal operating level)
H_SETPOINT = 0.0

# PID gains (three-element controller)
KP = 80.0          # proportional gain  [kg/(s·m)]
KI = 1.5           # integral gain      [kg/(s·m·s)]
KD = 200.0         # derivative gain    [kg/(s·m/s)]

# Integrator anti-windup clamp [kg/s]
I_CLAMP = 300.0

# Feedwater valve first-order lag time constant [s]
TAU_VALVE = 5.0

# Feedwater flow limits [kg/s]
FW_MIN = 0.0
FW_MAX = 1000.0

# Disturbance schedule  [(t_start [s], t_end [s], ΔFs [kg/s])]
# Each entry is a step change in steam demand during [t_start, t_end).
DISTURBANCES = [
    (300.0,  1200.0,  +50.0),    # +10 % load increase at t = 5 min
    (1800.0, 2700.0,  -75.0),    # -15 % load decrease at t = 30 min
    (3000.0, T_END,   +30.0),    # +6 % partial recovery at t = 50 min
]


# ---------------------------------------------------------------------------
# Helper: PID controller (with anti-windup)
# ---------------------------------------------------------------------------

class PIDController:
    """Discrete PID controller with derivative filtering and anti-windup."""

    def __init__(self, kp, ki, kd, dt, i_clamp=None, n_filter=10.0):
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt!r}")
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.i_clamp = i_clamp
        # Derivative low-pass filter coefficient
        self.alpha = n_filter / (n_filter + dt)

        self._integral = 0.0
        self._prev_error = 0.0
        self._deriv_filtered = 0.0

    def reset(self):
        self._integral = 0.0
        self._prev_error = 0.0
        self._deriv_filtered = 0.0

    def step(self, error, saturated=False):
        """
        Compute PID output for the current error sample.

        Parameters
        ----------
        error      : current setpoint error
        saturated  : if True, halt integration (anti-windup back-calculation)
        """
        # Integral (with anti-windup hold)
        if not saturated:
            self._integral += error * self.dt
            if self.i_clamp is not None:
                self._integral = max(-self.i_clamp,
                                     min(self.i_clamp, self._integral))

        # Derivative (filtered)
        raw_deriv = (error - self._prev_error) / self.dt
        self._deriv_filtered = (self.alpha * self._deriv_filtered
                                 + (1.0 - self.alpha) * raw_deriv)
        self._prev_error = error

        return (self.kp * error
                + self.ki * self._integral
                + self.kd * self._deriv_filtered)


# ---------------------------------------------------------------------------
# Steam demand schedule  (each step is ramped over RAMP_TIME seconds)
# ---------------------------------------------------------------------------

RAMP_TIME = 30.0   # duration of each load-change ramp [s]


def steam_flow_at(t):
    """Return steam flow [kg/s] at time t.

    Each disturbance entry is applied as a linear ramp over RAMP_TIME seconds
    at both the start and end of the disturbance interval, which avoids the
    numerical spikes that would arise from instantaneous step changes.
    """
    fs = FS_NOMINAL
    for t_start, t_end, delta in DISTURBANCES:
        # Rising ramp at start
        if t_start <= t < t_start + RAMP_TIME:
            fs += delta * (t - t_start) / RAMP_TIME
        elif t_start + RAMP_TIME <= t < t_end - RAMP_TIME:
            fs += delta
        elif t_end - RAMP_TIME <= t < t_end:
            fs += delta * (t_end - t) / RAMP_TIME
    return fs


def dsteam_flow_dt(t, dt=None):
    """Numerical derivative of steam flow [kg/s²]."""
    step = dt if dt is not None else DT
    return (steam_flow_at(t + step) - steam_flow_at(t)) / step


# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------

def run_simulation():
    n_steps = int(T_END / DT) + 1
    time = np.linspace(0.0, T_END, n_steps)

    # State arrays
    h = np.zeros(n_steps)        # water level [m]
    fw = np.zeros(n_steps)       # actual feedwater flow [kg/s]
    fs_arr = np.zeros(n_steps)   # steam flow [kg/s]
    fw_sp_arr = np.zeros(n_steps)  # FW setpoint [kg/s]
    error_arr = np.zeros(n_steps)  # level error [m]

    # Initial conditions – start at steady state
    h[0] = 0.0
    fw[0] = FS_NOMINAL
    fs_arr[0] = steam_flow_at(0.0)

    pid = PIDController(KP, KI, KD, DT, i_clamp=I_CLAMP)
    saturated_prev = False

    for k in range(n_steps - 1):
        t = time[k]

        # Current steam demand
        fs = steam_flow_at(t)
        fs_arr[k] = fs

        # Three-element controller: FF (steam flow) + PID (level error)
        # Pass saturation status from previous step for proper anti-windup.
        err = H_SETPOINT - h[k]
        error_arr[k] = err
        fw_sp = fs + pid.step(err, saturated=saturated_prev)
        fw_sp = max(FW_MIN, min(FW_MAX, fw_sp))  # hard limits
        fw_sp_arr[k] = fw_sp

        saturated_prev = (fw_sp >= FW_MAX or fw_sp <= FW_MIN)

        # Feedwater valve lag: first-order ODE  τ * dFw/dt = Fw_sp - Fw
        dfw_dt = (fw_sp - fw[k]) / TAU_VALVE
        fw_next = fw[k] + dfw_dt * DT
        fw_next = max(FW_MIN, min(FW_MAX, fw_next))

        # Level dynamics:  dh/dt = (Fw - Fs)/A + Ks * dFs/dt
        fs_dot = dsteam_flow_dt(t)
        dh_dt = (fw[k] - fs) / A_SG + KSS * fs_dot

        # Forward Euler integration
        h[k + 1] = h[k] + dh_dt * DT
        fw[k + 1] = fw_next

    # Fill last step
    fs_arr[-1] = steam_flow_at(time[-1])
    fw_sp_arr[-1] = fw_sp_arr[-2]
    error_arr[-1] = H_SETPOINT - h[-1]

    return time, h, fw, fs_arr, fw_sp_arr, error_arr


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(time, h, fw, fs_arr, fw_sp_arr, error_arr,
                 filename="sg_simulation_result.png"):
    t_min = time / 60.0   # convert to minutes for readability

    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(3, 1, hspace=0.40)

    # --- subplot 1: water level ---
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(t_min, h * 100.0, color="steelblue", linewidth=1.8,
             label="Water level")
    ax1.axhline(H_SETPOINT * 100.0, color="red", linestyle="--",
                linewidth=1.2, label="Setpoint")
    ax1.axhline(+30.0, color="orange", linestyle=":", linewidth=1.0,
                label="Hi alarm (+30 cm)")
    ax1.axhline(-30.0, color="orange", linestyle=":", linewidth=1.0,
                label="Lo alarm (−30 cm)")
    ax1.set_ylabel("Water level [cm]")
    ax1.set_title("SG Water Level Control Simulation  (SG 水位控制仿真)")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)
    _add_disturbance_bands(ax1, t_min)

    # --- subplot 2: flow rates ---
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.plot(t_min, fs_arr, color="firebrick", linewidth=1.8,
             label="Steam flow (Fs)")
    ax2.plot(t_min, fw, color="steelblue", linewidth=1.8,
             label="Feedwater flow (Fw)")
    ax2.plot(t_min, fw_sp_arr, color="steelblue", linewidth=1.0,
             linestyle="--", label="FW setpoint")
    ax2.set_ylabel("Flow rate [kg/s]")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)
    _add_disturbance_bands(ax2, t_min)

    # --- subplot 3: level error ---
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.plot(t_min, error_arr * 100.0, color="purple", linewidth=1.5,
             label="Level error")
    ax3.axhline(0.0, color="black", linestyle="-", linewidth=0.8)
    ax3.set_ylabel("Level error [cm]")
    ax3.set_xlabel("Time [min]")
    ax3.legend(loc="upper right", fontsize=8)
    ax3.grid(True, alpha=0.3)
    _add_disturbance_bands(ax3, t_min)

    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"Plot saved to '{filename}'")
    return fig


def _add_disturbance_bands(ax, t_min):
    """Shade the time intervals where steam disturbances are active."""
    colors = ["#FFC0CB", "#ADD8E6", "#90EE90"]
    for i, (t_start, t_end, delta) in enumerate(DISTURBANCES):
        ax.axvspan(t_start / 60.0, t_end / 60.0,
                   alpha=0.12, color=colors[i % len(colors)],
                   label=f"Disturbance {i + 1}")


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def print_summary(time, h, fw, fs_arr, error_arr):
    t_min = time / 60.0
    print("=" * 60)
    print("  SG Water Level Control Simulation – Summary")
    print("=" * 60)
    print(f"  Total simulation time  : {T_END / 60:.0f} min")
    print(f"  Integration time-step  : {DT} s")
    print(f"  Level setpoint         : {H_SETPOINT * 100:.1f} cm")
    print(f"  FW valve time constant : {TAU_VALVE} s")
    print()
    print("  Level statistics:")
    print(f"    Max level    : {h.max() * 100:+.2f} cm  "
          f"at t = {t_min[h.argmax()]:.1f} min")
    print(f"    Min level    : {h.min() * 100:+.2f} cm  "
          f"at t = {t_min[h.argmin()]:.1f} min")
    print(f"    Final level  : {h[-1] * 100:+.2f} cm")
    print(f"    RMS error    : {math.sqrt(np.mean(error_arr**2)) * 100:.3f} cm")
    print()
    print("  Disturbance schedule:")
    for i, (ts, te, df) in enumerate(DISTURBANCES):
        print(f"    [{i + 1}]  t = {ts / 60:.0f}–{te / 60:.0f} min  "
              f"DFs = {df:+.0f} kg/s  ({df / FS_NOMINAL * 100:+.0f} %)")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running SG water-level control simulation …")
    time, h, fw, fs_arr, fw_sp_arr, error_arr = run_simulation()
    print_summary(time, h, fw, fs_arr, error_arr)
    plot_results(time, h, fw, fs_arr, fw_sp_arr, error_arr)
    print("Done.")
