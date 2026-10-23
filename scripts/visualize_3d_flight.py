"""
Standalone Interactive 3D Flight Simulation Visualizer.
Launches the high-fidelity cinematic Three.js WebGL 3D simulator featuring:
- Procedural 1km x 1km x 300m Mountain Canyon with rocky terrain texturing & lighting
- 3D Hexacopter UAV model with spinning rotors, chassis, navigation lights, and camera gimbal
- Glowing multi-mode aerodynamic ribbon trajectory pathway
- 3D spatial wind particle streamlines and Venturi drafts
- MPPI (K=1024) lookahead rollout candidate fan
- Multi-camera director: Chase Cam, Payload Gimbal Cam, Free Orbit, Top-Down Tactical
- Real-time aerospace telemetry HUD
"""

import sys
import os
import argparse
import webbrowser

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def main():
    parser = argparse.ArgumentParser(description="AeroHarvest Cinematic 3D Flight Visualizer")
    parser.add_argument("--html", type=str, default="results/plots/live_3d_simulation.html", help="Path to 3D simulation HTML")
    parser.add_argument("--no-open", action="store_true", help="Do not automatically launch browser")
    args = parser.parse_args()

    html_path = os.path.abspath(args.html)
    if not os.path.exists(html_path):
        print(f"[Error] Simulation file not found at {html_path}")
        return

    print("==================================================================")
    print("      AEROHARVEST CINEMATIC 3D FLIGHT SIMULATION LAUNCHER        ")
    print("==================================================================")
    print(f"Simulation File: {html_path}")
    print("\nFeatures:")
    print("  * 3D Hexacopter with spinning rotors & pitch/roll/yaw attitude dynamics")
    print("  * Glowing aerodynamic trajectory ribbon color-coded by tactical mode")
    print("  * Canyon wind particle streamlines and orographic drafts")
    print("  * MPPI (K=1024) lookahead rollouts branching from vehicle")
    print("  * 4 Camera Modes: Chase Cam, Payload Gimbal, Free Orbit, Top-Down")
    print("  * Real-time 50 Hz aerospace telemetry HUD overlay")
    print("==================================================================")

    if not args.no_open:
        file_url = f"file:///{html_path.replace(os.sep, '/')}"
        print(f"\nOpening 3D simulation in your default browser:\n👉 {file_url}\n")
        webbrowser.open(file_url)


if __name__ == "__main__":
    main()
