"""
Flight Data Exporter for the AeroHarvest 3D Simulator.

Runs the REAL closed-loop AeroHarvest stack (physics -> observer -> GP -> state machine -> MPPI)
and Baseline 1 (A*) on the same wind seed, then writes everything the WebGL viewer needs to
`results/sim/flight_data.js`:
  * 10 Hz trajectory frames (position, velocity, thrust axis, true & estimated wind, GP variance,
    battery, reserve gate, power, tilt, tactical mode)
  * Mode / waypoint / target events
  * 1 Hz snapshots of sampled MPPI rollouts (subset of the K=1024)
  * GP predictive-variance snapshots over the canyon every 5 s
  * Terrain heightmap and a steady-state 3D wind grid
  * Baseline 1 trajectory and a side-by-side summary

Usage:
    python scripts/export_flight_data.py --wind_speed 7.5 --seed 42
"""

import sys
import os
import json
import time
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from aeroharvest.autonomy_stack import AeroHarvestStack
from aeroharvest.physics.canyon_environment import MountainCanyon, MissionScenario
from aeroharvest.physics.wind_field import DynamicWindField
from aeroharvest.physics.hexacopter_dynamics import HexacopterDynamics
from aeroharvest.models.power_model import MultirotorPowerModel
from aeroharvest.baselines.a_star_baseline import AStarGeometricBaseline

MODES = ["INITIALIZATION", "MISSION_CRUISE", "WIND_ADAPT", "ACTIVE_PROBE", "STABLE_CAPTURE", "RETURN_TO_BASE"]
FRAME_FIELDS = [
    "t", "x", "y", "z", "vx", "vy", "vz", "ux", "uy", "uz",
    "wtx", "wty", "wtz", "wex", "wey", "wez",
    "var", "batt", "reserve", "power", "tilt", "mode",
]
DT = 0.02          # 50 Hz physics
FRAME_EVERY = 5    # 10 Hz frames
ROLLOUT_EVERY = 50 # 1 Hz MPPI snapshots
GP_EVERY = 250     # GP snapshot every 5 s
HOME_RADIUS = 15.0


def rnd(values, nd=2):
    return [round(float(v), nd) for v in values]


def path_stats(positions, powers, tilts, dt):
    pos = np.asarray(positions)
    dist_km = float(np.sum(np.linalg.norm(np.diff(pos, axis=0), axis=1))) / 1000.0 if len(pos) > 1 else 0.0
    energy_wh = float(np.sum(powers) * dt / 3600.0)
    tilts = np.asarray(tilts)
    return {
        "distance_km": round(dist_km, 3),
        "energy_wh": round(energy_wh, 2),
        "wh_per_km": round(energy_wh / max(dist_km, 1e-3), 2),
        "tilt_rms_deg": round(float(np.sqrt(np.mean(tilts ** 2))) if len(tilts) else 0.0, 2),
        "tilt_max_deg": round(float(np.max(tilts)) if len(tilts) else 0.0, 2),
    }


def run_aeroharvest(args):
    stack = AeroHarvestStack(base_wind_speed=args.wind_speed, seed=args.seed)
    canyon, scenario = stack.canyon, stack.scenario
    home = scenario.launch_point.copy()

    # GP snapshot query grid (draped 45 m above terrain)
    g = np.linspace(-480.0, 480.0, 41)
    GX, GY = np.meshgrid(g, g)
    GZ = canyon.get_elevation(GX, GY) + 45.0
    gp_query = np.column_stack([GX.ravel(), GY.ravel(), GZ.ravel()])

    frames, rollouts, gp_snaps, events = [], [], [], []
    last_mode, wp_prev, targets_done = None, 0, set()
    mission_done_t, landed, step = None, False, 0
    t_wall = time.time()

    while stack.sim_time < args.max_time:
        info = stack.step()
        step += 1
        t = info["time"]

        if info["mode"] != last_mode:
            events.append({"t": round(t, 2), "type": "mode", "label": info["mode"]})
            last_mode = info["mode"]
        if scenario.current_waypoint_idx != wp_prev:
            wp_prev = scenario.current_waypoint_idx
            events.append({"t": round(t, 2), "type": "waypoint", "label": f"Waypoint {wp_prev} reached"})
        for tgt in scenario.thermal_targets:
            if tgt.completed and tgt.target_id not in targets_done:
                targets_done.add(tgt.target_id)
                events.append({"t": round(t, 2), "type": "target", "label": f"Thermal target {tgt.target_id} captured"})

        if step % FRAME_EVERY == 0:
            st = stack.vehicle.state
            up = st.attitude_rot[:, 2]
            frames.append(
                rnd([t], 2) + rnd(st.position, 2) + rnd(st.velocity, 2) + rnd(up, 3)
                + rnd(info["wind_true"], 2) + rnd(info["wind_est"], 2)
                + rnd([info["uncertainty_var"]], 3)
                + rnd([info["battery_pct"], info["battery_reserve_wh"], info["power_w"], info["tilt_deg"]], 2)
                + [MODES.index(info["mode"])]
            )

        if step % ROLLOUT_EVERY == 0 and stack.current_rollouts is not None:
            ro = stack.current_rollouts.detach().cpu().numpy()
            pick = np.linspace(0, ro.shape[0] - 1, 14).astype(int)
            sub = ro[pick][:, ::3, :]
            rollouts.append({"t": round(t, 2), "paths": np.round(sub, 1).tolist()})

        if step % GP_EVERY == 0:
            _, var = stack.gp_model.predict(gp_query)
            gp_snaps.append({"t": round(t, 2), "var": np.round(var, 2).tolist()})

        if info["is_mission_complete"] and mission_done_t is None:
            mission_done_t = t
            events.append({"t": round(t, 2), "type": "mission", "label": "Mission complete - returning home"})
        if mission_done_t is not None and np.linalg.norm(info["position"] - home) < HOME_RADIUS:
            landed = True
            events.append({"t": round(t, 2), "type": "home", "label": "Arrived at launch point"})
            break
        if info["battery_pct"] <= 0.01:
            events.append({"t": round(t, 2), "type": "fail", "label": "Battery depleted"})
            break
        if step % 2500 == 0:
            print(f"  [AeroHarvest] t={t:6.1f}s  mode={info['mode']:15s} wp={scenario.current_waypoint_idx}/6 "
                  f"targets={len(targets_done)}/3  batt={info['battery_pct']:5.1f}%  ({time.time() - t_wall:.0f}s wall)")

    h = stack.history
    summary = path_stats(h["position"], h["power_w"], h["tilt_deg"], DT)
    wind_err = np.linalg.norm(np.array(h["wind_est"]) - np.array(h["wind_true"]), axis=1)
    summary.update({
        "flight_time_s": round(stack.sim_time, 1),
        "mission_complete": mission_done_t is not None,
        "returned_home": landed,
        "waypoints": f"{scenario.current_waypoint_idx}/6",
        "targets": f"{len(targets_done)}/3",
        "battery_final_pct": round(h["battery_pct"][-1], 1),
        "wind_rmse_m_s": round(float(np.sqrt(np.mean(wind_err ** 2))), 2),
        "mean_loop_latency_ms": round(float(np.mean(h["latency_ms"])), 2),
    })
    return stack, frames, rollouts, gp_snaps, events, summary, g


def run_baseline(args):
    canyon = MountainCanyon()
    scenario = MissionScenario(canyon)
    wind = DynamicWindField(base_wind_speed=args.wind_speed, seed=args.seed)
    vehicle = HexacopterDynamics()
    vehicle.reset(scenario.launch_point)
    power = MultirotorPowerModel()
    planner = AStarGeometricBaseline()
    home = scenario.launch_point.copy()

    v_cmd, yaw_cmd, t, step = np.zeros(3), 0.0, 0.0, 0
    positions, powers, tilts, frames = [], [], [], []
    done_t, landed, min_agl = None, False, 1e9

    while t < args.max_time:
        pos = vehicle.state.position.copy()
        scenario.update_targets(pos, DT)
        scenario.check_waypoint_reached(pos)
        w = wind.get_wind_at(pos, t)
        p = power.compute_power_np(vehicle.state.velocity - w)
        tel = vehicle.step(v_cmd, yaw_cmd, w, DT, p)
        v_cmd, yaw_cmd, _ = planner.step(pos, scenario.get_current_waypoint(), tel["battery_percentage"], home)
        t += DT
        step += 1
        positions.append(pos)
        powers.append(p)
        tilts.append(tel["tilt_angle_deg"])
        min_agl = min(min_agl, canyon.get_altitude_above_ground(pos))
        if step % FRAME_EVERY == 0:
            frames.append(rnd([t], 2) + rnd(vehicle.state.position, 2) + rnd([tel["battery_percentage"]], 1))
        if scenario.is_mission_complete() and done_t is None:
            done_t = t
        if done_t is not None and np.linalg.norm(pos - home) < HOME_RADIUS:
            landed = True
            break
        if tel["battery_percentage"] <= 0.01:
            break

    summary = path_stats(positions, powers, tilts, DT)
    summary.update({
        "flight_time_s": round(t, 1),
        "mission_complete": done_t is not None,
        "returned_home": landed,
        "waypoints": f"{scenario.current_waypoint_idx}/6",
        "targets": f"{sum(tg.completed for tg in scenario.thermal_targets)}/3",
        "battery_final_pct": round(vehicle.state.battery_energy_wh / vehicle.battery_capacity_wh * 100, 1),
        "min_agl_m": round(min_agl, 1),
    })
    return frames, summary


def main():
    parser = argparse.ArgumentParser(description="Export AeroHarvest flight data for the 3D viewer")
    parser.add_argument("--wind_speed", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_time", type=float, default=600.0)
    parser.add_argument("--output", type=str, default="results/sim/flight_data.js")
    args = parser.parse_args()

    print("=== AeroHarvest flight export ===")
    print(f"Wind {args.wind_speed} m/s | seed {args.seed} | max {args.max_time}s")
    print("Running full AeroHarvest stack (real physics + observer + GP + state machine + MPPI)...")
    stack, frames, rollouts, gp_snaps, events, ah_summary, gp_grid = run_aeroharvest(args)
    print(f"  -> {len(frames)} frames, {len(events)} events, mission complete: {ah_summary['mission_complete']}")

    print("Running Baseline 1 (A* geometric) on the same wind seed...")
    base_frames, base_summary = run_baseline(args)
    print(f"  -> {len(base_frames)} frames, mission complete: {base_summary['mission_complete']}")

    canyon, scenario = stack.canyon, stack.scenario

    # Terrain heightmap (129 x 129 over the 1 km x 1 km domain), indexed [iy][ix]
    tg = np.linspace(-500.0, 500.0, 129)
    TX, TY = np.meshgrid(tg, tg)
    heights = np.round(canyon.get_elevation(TX, TY), 1)

    # Steady-state wind grid (t = 0) for particle advection
    wx = np.linspace(-500.0, 500.0, 21)
    wz = [20.0, 60.0, 100.0, 150.0, 200.0, 260.0]
    wind_field = stack.wind_field
    wind_grid = [[[rnd(wind_field.get_wind_at(np.array([x, y, z]), 0.0), 2) for x in wx] for y in wx] for z in wz]

    data = {
        "meta": {
            "wind_speed": args.wind_speed,
            "seed": args.seed,
            "frame_dt": DT * FRAME_EVERY,
            "frame_fields": FRAME_FIELDS,
            "modes": MODES,
            "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "terrain": {"min": -500.0, "max": 500.0, "n": 129, "heights": heights.tolist()},
        "wind_grid": {"xy": wx.tolist(), "z": wz, "v": wind_grid},
        "mission": {
            "launch": rnd(scenario.launch_point),
            "waypoints": [rnd(w) for w in scenario.perimeter_waypoints],
            "targets": [{"id": t.target_id, "pos": rnd(t.position), "radius": t.radius, "dwell": t.required_dwell_time}
                        for t in scenario.thermal_targets],
        },
        "frames": frames,
        "events": events,
        "rollouts": rollouts,
        "gp": {"grid": gp_grid.tolist(), "agl": 45.0, "snapshots": gp_snaps},
        "baseline": {"fields": ["t", "x", "y", "z", "batt"], "frames": base_frames},
        "summary": {"aeroharvest": ah_summary, "baseline": base_summary},
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("window.AEROHARVEST_DATA = ")
        json.dump(data, f, separators=(",", ":"))
        f.write(";\n")
    size_kb = os.path.getsize(args.output) / 1024
    print(f"\n[OK] Wrote {args.output} ({size_kb:.0f} KB)")
    print("AeroHarvest:", json.dumps(ah_summary))
    print("Baseline 1: ", json.dumps(base_summary))


if __name__ == "__main__":
    main()
