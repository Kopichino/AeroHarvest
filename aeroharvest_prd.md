# PROJECT REQUIREMENTS DOCUMENT (PRD)

## PROJECT TITLE: AeroHarvest: A Mission-Aware, Confidence-Gated Wind-Adaptive Autonomy Stack with Active Informative Path Planning and Camera-Stability Trajectory Optimization for Multirotor UAVs in Dynamic Mountain Valleys

**Document Version:** 2.0.0

**Status:** Approved for Implementation

**Academic Target:** Capstone Engineering Major Project

**Target Platform:** PX4 SITL + Gazebo Harmonic + ROS 2 Humble + GPyTorch CUDA

## 1. EXECUTIVE SUMMARY & SYSTEM OVERVIEW

Small multirotor Unmanned Aerial Vehicles (UAVs) operating in complex mountain valleys and forest wildfire monitoring zones face spatial wind disturbances that degrade battery life, destabilize camera payloads, and cause return-to-base failures. Standard flight controllers rely on reactive PID/L1 disturbance rejection or pre-planned geometric paths ($\text{A}^*$, Dijkstra), ignoring the energy-saving and information-gathering potential of spatial wind fields.

**AeroHarvest** introduces a software-in-the-loop (SITL) autonomy architecture that turns spatial wind fields from a flight hazard into a structured resource. The platform integrates:

1. A $50\text{ Hz}$ telemetry-based disturbance observer that estimates $3\text{D}$ wind vectors without dedicated pitot hardware.

2. A GPU-accelerated Sparse Gaussian Process (GPyTorch, Matérn $5/2$ covariance kernel) that builds an online $3\text{D}$ wind belief map ($\mu_w, \sigma_w^2$).

3. A novel Mission-Aware Decision Engine that continuously trades off target progress, active mutual information gain, motor electrical power, wind hazard risk, camera vibration jitter, and time delay.

4. An Adaptive Confidence-Gated Battery Reserve that dynamically inflates return-to-base safety thresholds based on line-of-sight wind variance.

5. A CUDA-accelerated Model Predictive Path Integral (MPPI) local planner executing $1,024$ parallel rollouts at $20\text{ Hz}$.

## 2. HIGH-LEVEL SYSTEM ARCHITECTURE

The AeroHarvest architecture consists of five closed-loop runtime layers:

```
+---------------------------------------------------------------------------------------------------+
|                                   AEROHARVEST SYSTEM ARCHITECTURE                                 |
+---------------------------------------------------------------------------------------------------+
|                                                                                                   |
|  [ LAYER 1: SIMULATION & ENVIRONMENT PHYSICS ]                                                     |
|  +---------------------------------------------------------------------------------------------+  |
|  | Gazebo Harmonic (Garden/Harmonic C++ Plugin)                                                |  |
|  | - Mountain Canyon Mesh (1 km x 1 km x 300 m)                                                |  |
|  | - Spatially Varying Wind Vector Field + Gaussian Micro-Gusts + Canyon Drafts                |  |
|  | - Hexacopter Airframe (PX4 v1.14 SITL Bridge via Micro-XRCE-DDS)                             |  |
|  +---------------------------------------------------------------------------------------------+  |
|                                                |                                                  |
|                                                v (50 Hz Telemetry Stream)                         |
|  [ LAYER 2: PERCEPTION & DISTURBANCE OBSERVER ]                                                   |
|  +---------------------------------------------------------------------------------------------+  |
|  | Telemetry Disturbance Observer (ROS 2 C++ Node)                                             |  |
|  | - Inputs: IMU Acceleration a_meas, Orientation Quaternion R_b^I, Thrust Setpoint T           |  |
|  | - Inverts Body Drag Equation -> Computes Local Relative Wind Vector v_rel & Wind v_wind     |  |
|  +---------------------------------------------------------------------------------------------+  |
|                                                |                                                  |
|                                                v (Sparse Wind Observations)                       |
|  [ LAYER 3: SPATIAL BELIEF ESTIMATION ]                                                           |
|  +---------------------------------------------------------------------------------------------+  |
|  | GPU-Accelerated Sparse Gaussian Process (Python / GPyTorch CUDA Node)                        |  |
|  | - Covariance: Matérn 5/2 Kernel                                                             |  |
|  | - Inducing Points: M = 128 sliding-window active landmarks                                  |  |
|  | - Output: 3D Wind Mean mu_w(x,y,z) & Predictive Variance sigma_w^2(x,y,z)                    |  |
|  +---------------------------------------------------------------------------------------------+  |
|                                                |                                                  |
|                                                v (Voxel Grid Mean + Uncertainty Map)              |
|  [ LAYER 4: DECISION ENGINE & TACTICAL STATE MACHINE ]                                            |
|  +---------------------------------------------------------------------------------------------+  |
|  | Mission-Aware Tactical Decision Engine (ROS 2 Python/C++ Node)                               |  |
|  | - Evaluates Objective Function: Score = Progress + Gamma*Info - Lambda1*Energy - ...        |  |
|  | - Mode Switching Logic: CRUISE | WIND_ADAPT | ACTIVE_PROBE | STABLE_CAPTURE | RETURN_TO_BASE |  |
|  | - Adaptive Confidence-Gated Energy Reserve Gate                                             |  |
|  +---------------------------------------------------------------------------------------------+  |
|                                                |                                                  |
|                                                v (Target State Setpoint & Cost Weights)           |
|  [ LAYER 5: LOCAL TRAJECTORY OPTIMIZATION & CONTROL ]                                             |
|  +---------------------------------------------------------------------------------------------+  |
|  | CUDA-Accelerated MPPI Planner (C++/LibTorch CUDA)                                            |  |
|  | - 1,024 Parallel Sampled Trajectories over a 6.0 s Horizon (K=1024, N=30, dt=0.2 s)         |  |
|  | - Integrates Power Model, Camera Vibration Cost, and Obstacle Penalty Functions             |  |
|  | - Outputs: Velocity Vectors (v_x, v_y, v_z) & Yaw Rates -> Transmitted to PX4 Offboard Node  |  |
|  +---------------------------------------------------------------------------------------------+  |
|                                                |                                                  |
|                                                v (PX4 Offboard Trajectory Setpoints)              |
|  +---------------------------------------------------------------------------------------------+  |
|  | PX4 Autopilot Actuator Control -> Motor PWM Signals -> Gazebo Physics Loop                  |  |
|  +---------------------------------------------------------------------------------------------+  |
+---------------------------------------------------------------------------------------------------+

```

## 3. MATHEMATICAL FORMULATIONS

### 3.1 Multirotor Telemetry-Based Disturbance Observer

The hexacopter translational equation of motion in the inertial frame is given by:

$$
m \mathbf{a}_{\text{meas}} = \mathbf{R}_b^I \begin{bmatrix} 0 \\ 0 \\ T_{\text{telemetry}} \end{bmatrix} - m g \mathbf{e}_3 + \mathbf{F}_{\text{ext}}
$$

where $m$ is the vehicle mass ($2.4\text{ kg}$), $\mathbf{a}_{\text{meas}}$ is the IMU linear acceleration vector, $\mathbf{R}_b^I$ is the body-to-inertial rotation matrix, $T_{\text{telemetry}}$ is the total thrust computed from motor telemetry, and $\mathbf{e}_3 = [0, 0, 1]^T$.

Isolating the external aerodynamic force $\mathbf{F}_{\text{ext}}$:

$$
\mathbf{F}_{\text{ext}} = m \mathbf{a}_{\text{meas}} - \mathbf{R}_b^I \begin{bmatrix} 0 \\ 0 \\ T_{\text{telemetry}} \end{bmatrix} + m g \mathbf{e}_3
$$

Assuming parasite drag dominates during forward flight, the relative air-velocity magnitude $\Vert{}\mathbf{v}_{\text{rel}}\Vert{}$ and vector $\mathbf{v}_{\text{rel}}$ are inverted using:

$$
\mathbf{F}_{\text{ext}} \approx -\frac{1}{2} \rho C_{D, \text{quad}} A_{\text{proj}} \Vert{}\mathbf{v}_{\text{rel}}\Vert{} \mathbf{v}_{\text{rel}}
$$

$$
\mathbf{v}_{\text{rel}} = -\sqrt{\frac{2 \Vert{}\mathbf{F}_{\text{ext}}\Vert{}}{\rho C_{D, \text{quad}} A_{\text{proj}}}} \cdot \frac{\mathbf{F}_{\text{ext}}}{\Vert{}\mathbf{F}_{\text{ext}}\Vert{}}
$$

The estimated inertial wind velocity $\hat{\mathbf{v}}_{\text{wind}}$ is extracted via vector subtraction:

$$
\hat{\mathbf{v}}_{\text{wind}} = \mathbf{v}_{\text{ground}} - \mathbf{v}_{\text{rel}}
$$

### 3.2 GPU-Accelerated Sparse Gaussian Process (Matérn 5/2)

Spatial wind vector field components $w_k \in \{v_{w,x}, v_{w,y}, v_{w,z}\}$ are modeled as independent Gaussian Processes over $3\text{D}$ spatial coordinates $\mathbf{x} = [x, y, z]^T$:

$$
w_k(\mathbf{x}) \sim \mathcal{GP}\left(0, k(\mathbf{x}, \mathbf{x}')\right)
$$

The spatial correlation is modeled using the Matérn $5/2$ covariance kernel:

$$
k(\mathbf{x}, \mathbf{x}') = \sigma_f^2 \left( 1 + \frac{\sqrt{5} d}{l} + \frac{5 d^2}{3 l^2} \right) \exp\left( -\frac{\sqrt{5} d}{l} \right)
$$

where $d = \Vert{}\mathbf{x} - \mathbf{x}'\Vert{}$, $\sigma_f^2$ is the signal variance, and $l$ is the spatial length scale ($l = 40.0\text{ m}$).

To maintain real-time GPU performance at $20\text{ Hz}$, Sparse Variational Gaussian Process (SVGP) inference is used with $M = 128$ inducing points $\mathbf{Z} \in \mathbb{R}^{M \times 3}$:

$$
q(\mathbf{f}) = \int p(\mathbf{f} \vert{} \mathbf{u}) q(\mathbf{u}) d\mathbf{u}, \quad q(\mathbf{u}) = \mathcal{N}(\mathbf{m}, \mathbf{S})
$$

Predictive mean $\mu_w(\mathbf{x}_*)$ and predictive variance $\sigma_w^2(\mathbf{x}_*)$ at query position $\mathbf{x}_*$ are evaluated in batch over a $3\text{D}$ voxel grid:

$$
\mu_w(\mathbf{x}_*) = \mathbf{k}_{* \mathbf{u}} \mathbf{K}_{\mathbf{u}\mathbf{u}}^{-1} \mathbf{m}
$$

$$
\sigma_w^2(\mathbf{x}_*) = k(\mathbf{x}_*, \mathbf{x}_*) - \mathbf{k}_{* \mathbf{u}} \mathbf{K}_{\mathbf{u}\mathbf{u}}^{-1} \mathbf{k}_{\mathbf{u} *} + \mathbf{k}_{* \mathbf{u}} \mathbf{K}_{\mathbf{u}\mathbf{u}}^{-1} \mathbf{S} \mathbf{K}_{\mathbf{u}\mathbf{u}}^{-1} \mathbf{k}_{\mathbf{u} *}
$$

### 3.3 Multirotor Electrical Power Model

The instant electrical power consumption $P_{\text{elec}}$ (Watts) accounts for momentum-theory induced hover power, parasite drag power, avionics base draw, and gimbal actuator load:

$$
P_{\text{elec}}(\mathbf{v}_{\text{rel}}) = P_{\text{hover}} \left( \sqrt{ \sqrt{1 + \left( \frac{\Vert{}\mathbf{v}_{\text{rel}}\Vert{}}{2 v_i} \right)^4 } - \left( \frac{\Vert{}\mathbf{v}_{\text{rel}}\Vert{}}{2 v_i} \right)^2 } \right)^{-1} + \frac{1}{2} \rho C_{D, \text{body}} A_{\text{front}} \Vert{}\mathbf{v}_{\text{rel}}\Vert{}^3 + P_{\text{electronics}}
$$

where:

* $P_{\text{hover}} = \frac{(m g)^{1.5}}{\sqrt{2 \rho A_{\text{rotor}}}} = 380\text{ W}$ (for $2.4\text{ kg}$ hexacopter)

* $v_i = \sqrt{\frac{m g}{2 \rho A_{\text{rotor}}}} = 5.2\text{ m/s}$ (induced velocity in hover)

* $P_{\text{electronics}} = 25\text{ W}$ (PX4 flight controller, Jetson Orin Nano, optical camera, 2-axis gimbal)

### 3.4 Camera Vibration & Image Jitter Penalty Model

High airspeeds or turbulent gusts force the multirotor to pitch or roll aggressively to maintain equilibrium, inducing high-frequency mechanical vibrations and axis movement beyond the gimbal's compensation bandwidth.

The camera stability cost $C_{\text{camera}}$ evaluated along trajectory candidate $j$ during an active capture window is:

$$
C_{\text{camera}} = \int_{t_0}^{t_0 + T} \left( \alpha_1 \Vert{}\boldsymbol{\omega}_{\text{body}}(t)\Vert{}^2 + \alpha_2 \max\left(0, \theta_{\text{tilt}}(t) - \theta_{\text{max}}\right)^2 + \alpha_3 \Vert{}\dot{\mathbf{v}}_{\text{wind}}(t)\Vert{}^2 \right) dt
$$

where $\boldsymbol{\omega}_{\text{body}} = [\dot{\phi}, \dot{\theta}, \dot{\psi}]^T$ is the body rotation rate vector, $\theta_{\text{tilt}} = \arccos(\mathbf{R}_{33})$ is the total roll/pitch lean angle, $\theta_{\text{max}} = 15^\circ$ is the stability limit, and $\Vert{}\dot{\mathbf{v}}_{\text{wind}}\Vert{}$ is the temporal gust derivative (turbulence rate).

### 3.5 Unified Tactical Decision Score

Every candidate high-level tactical action $a_k$ (waypoint vector) is evaluated using the composite decision function:

$$
\text{Score}(a_k) = w_1 \cdot \text{Progress}(a_k) + w_2 \cdot I(\mathbf{w}_*; \mathbf{y}) - w_3 \cdot P_{\text{elec}}(a_k) - w_4 \cdot R_{\text{wind}}(a_k) - w_5 \cdot C_{\text{camera}}(a_k) - w_6 \cdot \Delta T(a_k)
$$

where $I(\mathbf{w}_*; \mathbf{y})$ is the mutual information gain along action path $a_k$:

$$
I(\mathbf{w}_*; \mathbf{y}) = \frac{1}{2} \ln \det \left( \mathbf{I} + \sigma_n^{-2} \mathbf{K}_{**} \right)
$$

and $R_{\text{wind}}$ is the wind hazard penalty ($R_{\text{wind}} = \max(0, \Vert{}\mu_w\Vert{} + 2 \sigma_w - v_{\text{limit}})$).

### 3.6 Adaptive Confidence-Gated Battery Reserve Gate

The required energy threshold $E_{\text{reserve}}$ to trigger immediate Return-To-Base (`RETURN_TO_BASE`) is dynamically scaled by the predictive uncertainty of the return path:

$$
E_{\text{reserve}} = E_{\text{nominal\_return}} \cdot \left( 1.0 + \beta \int_{\text{path}_{\text{RTB}}} \sigma_w(\mathbf{p}) \, dp \right)
$$

If remaining battery energy $E_{\text{battery}} \le E_{\text{reserve}}$, the decision engine overrides all mission objectives and forces execution of mode `RETURN_TO_BASE`.

## 4. OPERATIONAL MODES & STATE MACHINE

```
                     +---------------------------------------+
                     |             INITIALIZATION            |
                     +---------------------------------------+
                                         |
                                         v
                     +---------------------------------------+
                     |            MISSION_CRUISE             |
                     |  (Standard Waypoint Progress Mode)    |
                     +---------------------------------------+
                        /                |                \
                       /                 |                 \
                      v                  v                  v
+---------------------------+  +-------------------+  +---------------------------+
|        WIND_ADAPT         |  |   ACTIVE_PROBE    |  |      STABLE_CAPTURE       |
| (Exploit Tailwinds/Ducts) |  | (Informative Detour)| | (Camera Jitter Minimizer) |
+---------------------------+  +-------------------+  +---------------------------+
                      \                  |                 /
                       \                 |                /
                        v                v               v
                     +---------------------------------------+
                     |            RETURN_TO_BASE             |
                     |   (Confidence-Gated Energy Reserve)   |
                     +---------------------------------------+

```

### Operational Mode Specifications:

1. **`MISSION_CRUISE` (Nominal Mode):**

   * *Condition:* Wind speed $\Vert{}\hat{\mathbf{v}}_{\text{wind}}\Vert{} < 4.0\text{ m/s}$, map uncertainty $\sigma_w^2 < 0.2\text{ m}^2/\text{s}^2$, camera capture inactive.

   * *Behavior:* Follows primary mission search grid via direct paths at nominal velocity ($8.0\text{ m/s}$).

2. **`WIND_ADAPT` (Energy-Exploitation Mode):**

   * *Condition:* Wind speed $\Vert{}\hat{\mathbf{v}}_{\text{wind}}\Vert{} \ge 4.0\text{ m/s}$, candidate low-resistance corridor detected.

   * *Behavior:* Re-routes trajectory into mountain canyon drafts or tailwind corridors. Adjusts flight altitude to minimize headwind drag or utilize localized updrafts.

3. **`ACTIVE_PROBE` (Informative Path Planning Mode):**

   * *Condition:* Local GP variance $\sigma_w^2 > 0.8\text{ m}^2/\text{s}^2$ along upcoming mission leg AND mutual information efficiency metric $\frac{I(\mathbf{w}_*; \mathbf{y})}{\Delta E} > \eta_{\text{threshold}}$.

   * *Behavior:* Executes a short $30\text{--}60\text{ m}$ exploratory detour into unmapped airspace to sample the wind field, updating the GP covariance matrix to reduce variance for subsequent return or traverse calculations.

4. **`STABLE_CAPTURE` (Payload-Aware Inspection Mode):**

   * *Condition:* Drone enters an active thermal/RGB inspection radius around a simulated fire boundary or target zone.

   * *Behavior:* Restricts total tilt angle to $\theta_{\text{tilt}} \le 12^\circ$, limits angular velocity rates to $\Vert{}\boldsymbol{\omega}\Vert{} \le 0.1\text{ rad/s}$, and orientates heading into wind to minimize camera frame jitter.

5. **`RETURN_TO_BASE` (Failsafe Mode):**

   * *Condition:* Onboard battery energy $E_{\text{battery}} \le E_{\text{reserve}}(\sigma_w)$.

   * *Behavior:* Overrides all active search objectives. Selects the minimum-energy return path calculated by the Sparse GP model and flies directly to the launch coordinate.

## 5. HARDWARE & SOFTWARE STACK SPECIFICATIONS

```
+-----------------------------------------------------------------------------------+
|                            SOFTWARE & HARDWARE STACK                              |
+-----------------------------------------------------------------------------------+
| Operating System    | Ubuntu 22.04 LTS (Jammy Jellyfish)                          |
| Middleware          | ROS 2 Humble Hawksbill (DDS: CycloneDDS)                    |
| Flight Controller   | PX4 Autopilot SITL v1.14.0 (Micro-XRCE-DDS Agent)          |
| Simulator           | Gazebo Harmonic (v7.x / Garden-Harmonic Bridge)             |
| ML/GP Framework     | PyTorch 2.2 + GPyTorch 1.11 (CUDA 12.1 Acceleration)        |
| Math/Optimization   | Eigen 3.4, LibTorch C++ API, CUDA C++ Trajectory Kernels    |
| Primary Vehicle     | Generic Hexacopter Airframe (Rotors: 6, Diagonal: 650 mm)   |
| Hardware Baseline   | NVIDIA Jetson Orin Nano (Target) / RTX 4080 (Simulation)    |
+-----------------------------------------------------------------------------------+

```

## 6. EXPERIMENTAL EVALUATION PLAN

### 6.1 Simulation Environment

* **Map Environment:** $1\text{ km} \times 1\text{ km} \times 300\text{ m}$ $3\text{D}$ Gazebo Harmonic Mountain Canyon with a variable $3\text{D}$ vector wind field ($0\text{--}14\text{ m/s}$ mean, $3.0\text{ m/s}$ peak gust variations).

* **Mission Task:** Complete a 6-waypoint perimeter inspection task while maintaining camera stability over 3 thermal inspection targets.

* **Evaluation Protocol:** **30-Run Monte Carlo Simulation** with randomized wind seed initializations per trial.

### 6.2 Baselines for Comparison

1. **Baseline 1 (Geometric Shortest Path -** $\text{A}^*$**):** Static 3D path planner ignoring wind, using standard PX4 position controller.

2. **Baseline 2 (Reactive Disturbance Control - L1 Adaptive):** Trajectory planner with local disturbance rejection; no spatial map or look-ahead.

3. **Baseline 3 (GP Energy-Only Planner):** Uses Gaussian Process and power model, but omits Active Probing, Camera Stability, and Dynamic Battery Reserves.

### 6.3 Ablation Studies

1. **Ablation 1 (GP-Disabled):** Full pipeline with GP spatial prediction replaced by local instantaneous wind estimates.

2. **Ablation 2 (MPPI-Disabled):** Full pipeline with MPPI local planner replaced by standard geometric waypoint interpolation.

### 6.4 Expected Quantitative Performance Benchmarks

| Metric | Baseline 1 ($\text{A}^*$) | Baseline 2 (Reactive) | Baseline 3 (GP-Energy) | AeroHarvest (Full) | 
| ----- | ----- | ----- | ----- | ----- | 
| **Mean Energy Consumption (**$\text{Wh/km}$**)** | $38.4 \pm 2.1$ | $31.2 \pm 1.8$ | $24.1 \pm 1.4$ | $21.6 \pm 1.2$ | 
| **Mission Completion Rate (**$\%$**)** | $72.0\%$ | $81.5\%$ | $89.0\%$ | $96.5\%$ | 
| **Return-to-Base Failure Rate (**$\%$**)** | $18.0\%$ | $12.0\%$ | $8.0\%$ | $0.0\%$ | 
| **Camera Angular Jitter RMS (**$\text{deg}$**)** | $8.4^\circ$ | $6.2^\circ$ | $5.1^\circ$ | $1.8^\circ$ | 
| **Wind Velocity Estimation RMSE (**$\text{m/s}$**)** | N/A | $1.15 \pm 0.22$ | $0.28 \pm 0.04$ | $0.26 \pm 0.03$ | 
| **Mean Control Loop Latency (**$\text{ms}$**)** | $4.2\text{ ms}$ | $8.1\text{ ms}$ | $15.2\text{ ms}$ | $18.4\text{ ms}$ | 

## 7. TEAM ROLES & WORK DIVISION

```
+---------------------------------------------------------------------------------------------------+
|                                     TEAM WORK DIVISION                                            |
+---------------------------------------------------------------------------------------------------+
| TEAM MEMBER                  | CORE TECHNICAL RESPONSIBILITIES                                    |
+------------------------------+--------------------------------------------------------------------+
| Koppesh P                    | - Lead Simulation Physics & Aerodynamics Architecture               |
| (Reg No: 23BAI1113)          | - Gazebo Harmonic 3D Mountain Canyon World Plugin & Mesh Build     |
|                              | - C++ Wind Vector Field & Micro-Gust Generator Plugin              |
|                              | - Multirotor Body Aerodynamics & Drag Surface Calibration          |
|                              | - Automated Headless Monte Carlo Simulation Runner Scripts         |
+------------------------------+--------------------------------------------------------------------+
| Shaik Abdulkhader Masthan    | - Lead Perception & Machine Learning Pipeline                      |
| (Reg No: 23BAI1123)          | - 50 Hz ROS 2 C++ Telemetry Disturbance Observer Node Implementation|
|                              | - GPyTorch CUDA Sparse Gaussian Process Implementation             |
|                              | - Active Informative Path Planning (Mutual Information Module)     |
|                              | - RViz2 Real-Time 3D Wind Vector & Uncertainty Visualizer Dashboard|
+------------------------------+--------------------------------------------------------------------+
| Santhosh Kumar               | - Lead Tactical Decision Engine, MPPI Control & Autopilot Bridge  |
| (Reg No: 23BAI1236)          | - CUDA-Accelerated MPPI Trajectory Planner (1,024 Parallel Rollouts)|
|                              | - Mission-Aware State Machine & Unified Decision Scoring Node      |
|                              | - PX4 SITL Micro-XRCE-DDS Offboard Trajectory Bridge Node          |
|                              | - System Benchmarking Suite & Statistical Metric Logger            |
+------------------------------+--------------------------------------------------------------------+

```

## 8. 14-WEEK IMPLEMENTATION TIMELINE

```
+----------------------------------------------------------------------------------------------------+
| WEEK  | MILESTONE & DELIVERABLES                                                                   |
+-------+--------------------------------------------------------------------------------------------+
| W1-2  | - Repository setup, ROS 2 Humble workspace build, PX4 v1.14 SITL bridge verification.      |
|       | - Initial Gazebo Harmonic world creation (Canyon mesh setup).                              |
+-------+--------------------------------------------------------------------------------------------+
| W3-4  | - Implementation of Gazebo C++ Wind Field & Micro-Gust Plugin.                            |
|       | - Build $50\text{ Hz}$ C++ Telemetry Disturbance Observer Node (Validation against GT wind).|
+-------+--------------------------------------------------------------------------------------------+
| W5-6  | - GPyTorch CUDA Sparse Gaussian Process (SVGP) node implementation.                        |
|       | - Integration of Matérn $5/2$ kernel and online $128$ inducing points sliding window.      |
|       | - RViz2 3D voxel grid display for live wind mean and variance visualization.               |
+-------+--------------------------------------------------------------------------------------------+
| W7-8  | - Formulation and C++ build of Multirotor Power Model and Camera Vibration Penalty.       |
|       | - CUDA MPPI Local Planner setup ($1,024$ rollouts, $20\text{ Hz}$ execution).              |
+-------+--------------------------------------------------------------------------------------------+
| W9-10 | - Build Mission-Aware Tactical Decision Engine (5 operational modes + mode switching).     |
|       | - Integration of Adaptive Confidence-Gated Battery Reserve safety gate.                    |
+-------+--------------------------------------------------------------------------------------------+
| W11-12| - Closed-loop integration of complete stack (Observer -> GP -> Decision -> MPPI -> PX4).   |
|       | - Implementation of Baseline planners ($\text{A}^*$, Reactive, GP-Energy).                 |
+-------+--------------------------------------------------------------------------------------------+
| W13-14| - Automated execution of 30-run Monte Carlo evaluations across all scenario variants.     |
|       | - Statistical data extraction, graph generation, final report, and live demo rehearsal.   |
+----------------------------------------------------------------------------------------------------+

```

## 9. RISK REGISTER & CONTINGENCY PLAN

| Risk ID | Risk Description | Severity | Likelihood | Mitigation / Fallback Plan | 
| ----- | ----- | ----- | ----- | ----- | 
| **R1** | **GP CUDA Latency Exceeds** $50\text{ ms}$ | High | Medium | Reduce inducing points from $M=128$ to $M=64$. Downsample query grid resolution from $5\text{ m}$ to $10\text{ m}$ voxels. | 
| **R2** | **PX4 Offboard Mode Rejection / DDS Drops** | High | Low | Implement automatic fallback to local altitude/position hold within the bridge node. Increase DDS buffer queue sizes. | 
| **R3** | **MPPI Trajectory Divergence in Extreme Wind** | High | Low | Enforce hard input clamping on MPPI control setpoints ($\theta_{\text{max}} \le 25^\circ$). Fall back to PX4 internal position control mode if cost spikes. | 
| **R4** | **Gazebo Physics GPU Memory Overload** | Medium | Medium | Run Gazebo headless during Monte Carlo evaluation sweeps. Limit visual rendering meshes to active camera frustum. | 

## 10. 5-MINUTE LIVE DEMO SCRIPT FOR PROJECT COMMITTEE REVIEW

### Minute 0:00 – 1:00: Baseline Failure Demonstration

* **Presenter:** Koppesh P

* **Action:** Launch Gazebo Harmonic Mountain Canyon world. Trigger Baseline 1 ($\text{A}^*$ planner) on a $300\text{ m}$ canyon traversal with a $10\text{ m/s}$ crosswind gust.

* **Visual:** Point to the drone struggling against headwind, tilting aggressively ($\theta > 25^\circ$), camera feed showing severe angular shaking, and battery percentage dropping rapidly.

* **Narrative:** *"Standard planners ignore spatial wind. Notice how the baseline drone forces its way through the canyon headwind, drawing peak current and shaking the camera payload."*

### Minute 1:00 – 2:30: Real-Time Disturbance Observation & Sparse GP Belief Mapping

* **Presenter:** Shaik Abdulkhader Masthan

* **Action:** Switch screen to RViz2 Dashboard while AeroHarvest launches.

* **Visual:** Show $50\text{ Hz}$ disturbance observer arrows matching ground-truth wind vectors. As the drone flies, watch the GPyTorch Sparse GP model populate colorful $3\text{D}$ wind velocity vectors ($\mu_w$) and translucent uncertainty variance spheres ($\sigma_w^2$).

* **Narrative:** *"AeroHarvest infers wind forces directly from IMU and motor thrust telemetry at* $50\text{ Hz}$*. In real-time, our GPU Sparse GP updates a* $3\text{D}$ *spatial wind map in under* $15\text{ ms}$*."*

### Minute 2:30 – 3:45: Tactical Decision Engine & Active Informative Probing

* **Presenter:** Santhosh Kumar

* **Action:** Trigger target inspection event requiring high camera stability. Show terminal state logs transitioning: `MISSION_CRUISE` $\rightarrow$ `ACTIVE_PROBE` $\rightarrow$ `STABLE_CAPTURE`.

* **Visual:** The hexacopter executes a brief $40\text{ m}$ exploratory detour into an unmapped ridge, drops map variance, then enters the inspection target while holding horizontal attitude ($\theta < 10^\circ$) despite ambient wind. Show live side-by-side camera comparison (Baseline shaking vs. Ours stabilized).

* **Narrative:** *"When faced with high uncertainty, our decision engine executes an Active Probe detour to gather information, then switches to `STABLE_CAPTURE` mode to lock the camera angle for sharp image acquisition."*

### Minute 3:45 – 5:00: Adaptive Battery Reserve & Performance Results Summary

* **Presenter:** Santhosh Kumar / Team

* **Action:** Simulate dynamic headwind increase along return route. Show `RETURN_TO_BASE` triggering early due to uncertainty reserve expansion. Display final quantitative metrics dashboard.

* **Visual:** The baseline drone runs out of battery $150\text{ m}$ short of the home base, while AeroHarvest lands safely with $12\%$ remaining battery. Show summary table: $43.7\%$ energy savings, $0\%$ RTB failure rate, $78.5\%$ jitter reduction.

* **Narrative:** *"AeroHarvest dynamically inflates its battery safety reserve based on route variance. We achieve safe, stabilized mission execution with over* $40\%$ *energy improvement over standard methods."*

## 11. REFERENCES

1. J. Williams et al., "Information-theoretic model predictive control: Theory and applications to autonomous driving," *IEEE Transactions on Robotics*, vol. 34, no. 6, pp. 1603–1622, 2018.

2. J. Gardner et al., "GPyTorch: Blackbox Matrix-Matrix Gaussian Process Inference with GPU Acceleration," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2018.

3. G. Chowdhury et al., "Wind estimation and obstacle avoidance for small UAVs in turbulent urban environments," *Journal of Guidance, Control, and Dynamics*, vol. 42, no. 4, pp. 811–827, 2019.

4. M. Spacek et al., "Fast GPU-accelerated MPPI for multirotor obstacle avoidance in dynamic environments," *IEEE Robotics and Automation Letters*, vol. 7, no. 2, pp. 3410–3417, 2022.

5. Autonomous Systems Lab, "PX4 Autopilot Software Architecture and SITL Simulation Framework," *Dronecode Foundation*, 2024. \[Online\]. Available: https://px4.io/