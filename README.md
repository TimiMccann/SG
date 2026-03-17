# SG 水位控制仿真 — Steam Generator Water-Level Control Simulation

A Python simulation of the secondary-side water-level control system for a
pressurised-water-reactor (PWR) **Steam Generator (SG)**.

## Background

Maintaining the correct water level in the secondary side of an SG is a
critical control function in a PWR plant.  Too high a level risks water carry-
over into the main-steam lines; too low a level uncovers the tube bundle and
reduces heat transfer.  The challenge is exacerbated by the *shrink-and-swell*
phenomenon: a sudden increase in steam demand briefly *raises* the level (void
fraction increases as pressure drops) before the level falls due to reduced
water inventory.

This simulation implements the classic **three-element feedwater control**
strategy:

```
          ┌─────────────┐   steam flow (Fs)
          │  Feed-forward│──────────────────────────┐
          └─────────────┘                           │   ┌────────────┐
Water level ──► [PID] ──────────────────────────────┴──►│ FW valve   │──► Fw
                                                        │ (1st-order │
                                                        │  lag, τ)   │
                                                        └────────────┘
```

The feedwater (FW) setpoint is:

```
  Fw_sp = Fs  +  Kp·e  +  Ki·∫e dt  +  Kd·de/dt
```

where `e = h_sp − h` is the level error.  The steam-flow term provides
feed-forward compensation so that the PID only needs to correct the residual
level error caused by measurement noise and model mismatch.

## Physics model

```
  dh/dt  = (Fw − Fs) / A  +  Ks · dFs/dt

  τv · dFw/dt  = Fw_sp − Fw
```

| Symbol | Description | Units |
|--------|-------------|-------|
| `h`    | Water level (from normal operating level) | m |
| `Fw`   | Actual feedwater flow | kg/s |
| `Fs`   | Steam (main-steam) flow | kg/s |
| `A`    | Effective cross-sectional area of SG | m² |
| `Ks`   | Shrink-and-swell coefficient | m·s/kg |
| `τv`   | Feedwater valve lag time constant | s |

Steam-demand disturbances are modelled as linear ramps (not instantaneous
steps) to reproduce realistic load-following transients.

## Requirements

```
pip install numpy matplotlib
```

Python >= 3.8.

## Usage

```bash
python sg_simulation.py
```

The script prints a summary table to stdout and saves a plot to
`sg_simulation_result.png`.

### Example output

```
============================================================
  SG Water Level Control Simulation - Summary
============================================================
  Total simulation time  : 60 min
  Integration time-step  : 1.0 s
  Level setpoint         : 0.0 cm
  FW valve time constant : 5.0 s

  Level statistics:
    Max level    : +23.11 cm  at t = 44.6 min
    Min level    : -23.11 cm  at t = 30.1 min
    Final level  : -3.02 cm
    RMS error    :  2.95 cm

  Disturbance schedule:
    [1]  t = 5-20 min   DFs = +50 kg/s  (+10 %)
    [2]  t = 30-45 min  DFs = -75 kg/s  (-15 %)
    [3]  t = 50-60 min  DFs = +30 kg/s  ( +6 %)
============================================================
```

The generated plot shows three panels:

1. **Water level** - with setpoint and +-30 cm alarm lines.
2. **Flow rates** - steam flow, actual feedwater flow, and FW setpoint.
3. **Level error** - deviation from setpoint over time.

Shaded regions mark the active disturbance windows.

## Running the tests

```bash
pip install pytest
python -m pytest test_sg_simulation.py -v
```

## File structure

```
sg_simulation.py        Main simulation (model + controller + plotting)
test_sg_simulation.py   Unit and integration tests
README.md               This file
```

## Controller tuning

Key parameters in `sg_simulation.py`:

| Parameter   | Default | Description |
|-------------|---------|-------------|
| `KP`        | 80.0    | Proportional gain [kg/(s*m)] |
| `KI`        | 1.5     | Integral gain [kg/(s*m*s)] |
| `KD`        | 200.0   | Derivative gain [kg*s/(s*m)] |
| `I_CLAMP`   | 300.0   | Integrator anti-windup clamp [kg/s] |
| `TAU_VALVE` | 5.0     | FW valve lag time constant [s] |
| `A_SG`      | 500.0   | Effective SG cross-sectional area [m^2] |
| `KSS`       | 0.02    | Shrink-and-swell coefficient [m*s/kg] |
