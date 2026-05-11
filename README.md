# Rigid-Body Dynamics + IK + Torque & Impedance Control (2-DoF Arm)

This project demonstrates joint-level robot modeling and control for a planar 2-DoF manipulator.

## Features
- Forward kinematics (end-effector position)
- Jacobian computation
- Inverse kinematics (damped least squares)
- Rigid-body dynamics in joint space:
  - Mass matrix M(q)
  - Coriolis/centrifugal vector C(q,qdot)qdot
  - Gravity vector g(q)
- Inverse dynamics / computed torque control for trajectory tracking
- Joint impedance control (spring-damper behavior) with disturbance rejection

## Model
Robot dynamics:
M(q) qdd + C(q,qdot) + g(q) = tau

Computed torque control:
\tau = M(q)(qdd_d + Kd(qd_dot-qdot) + Kp(qd-q)) + C + g

Impedance control:
\tau = K(qd-q) - D qdot + g

# Model-Based Control and Robustness Analysis for a 5-DOF Robotic Manipulator

This project develops a simulation framework for a 5-DOF serial robotic manipulator and uses it to benchmark classical model-based controllers for trajectory tracking under disturbances and uncertainty.

The goal is to build a clean, research-style robotics simulation pipeline that demonstrates manipulator kinematics, simplified dynamics, trajectory generation, controller implementation, and quantitative performance evaluation.

## Motivation

Robotics control problems are rarely just about making a robot move from one point to another. In practice, a useful control framework should answer questions such as:

- How well does the manipulator track a desired trajectory?
- How much control effort is required?
- How sensitive is performance to disturbances or model mismatch?
- What is gained by using more model information in the controller?

This project addresses those questions by comparing multiple controllers on the same 5-DOF manipulator model.

## Features

- 5-DOF serial manipulator modeled with Denavit–Hartenberg parameters
- Forward kinematics and geometric Jacobian
- Damped least-squares inverse kinematics
- Smooth joint-space trajectory generation
- Structured simplified joint-space dynamics
- RK4-based numerical simulation
- Controller comparison:
  - PD control
  - PD + gravity compensation
  - Computed torque control


## System Model

The manipulator dynamics are modeled in the standard form:

\[
M(q)\ddot{q} + C(q,\dot{q})\dot{q} + G(q) + F(\dot{q}) = \tau + \tau_d
\]

where:

- \( q \) is the vector of joint coordinates
- \( M(q) \) is the inertia matrix
- \( C(q,\dot{q})\dot{q} \) captures Coriolis and centrifugal effects
- \( G(q) \) is the gravity vector
- \( F(\dot{q}) \) is viscous friction
- \( \tau \) is the control torque
- \( \tau_d \) is an external disturbance torque

The current version uses a structured simplified dynamic model rather than a full rigid-body identification model. The purpose is to preserve the main control-relevant effects while keeping the simulation transparent and easy to extend.

## Controllers

### 1. PD Control

\[
\tau = K_p(q_d - q) + K_d(\dot{q}_d - \dot{q})
\]

This is the baseline controller and serves as the simplest benchmark.

### 2. PD + Gravity Compensation

\[
\tau = K_p(q_d - q) + K_d(\dot{q}_d - \dot{q}) + G(q)
\]

This compensates for gravity explicitly and usually improves steady tracking performance.

### 3. Computed Torque Control

\[
\tau = M(q)\left(\ddot{q}_d + K_d(\dot{q}_d - \dot{q}) + K_p(q_d - q)\right) + C(q,\dot{q}) + G(q)
\]

This controller uses the manipulator model to cancel nonlinear effects and impose approximate linear error dynamics.

## Simulation Workflow

The simulation proceeds as follows:

1. Define manipulator geometry and physical parameters
2. Compute a reachable goal using inverse kinematics
3. Generate a smooth joint-space reference trajectory
4. Simulate the manipulator under a chosen controller
5. Inject disturbances if desired
6. Record joint motion, end-effector motion, and control torques
7. Compute tracking and effort metrics
8. Visualize results

## Metrics

The framework reports the following metrics:

- Joint RMS tracking error
- End-effector RMS tracking error
- Maximum absolute torque
- Torque energy

These metrics allow a structured comparison between controllers beyond simple visual inspection.



# Contact-Aware Control of a 5-DOF Robotic Manipulator using Impedance and Admittance Strategies

This project develops a simulation framework for manipulator-environment interaction using a 5-DOF serial robotic arm. The focus is on compliant contact behavior and the comparison of two classical interaction-control strategies:

- Impedance control
- Admittance control

The manipulator interacts with a virtual compliant wall, and the simulation is used to study force response, penetration depth, motion compliance, and control effort.

## Motivation

Free-space tracking is only part of robotics. Many real robotic systems must interact with their environment, including:

- tool guidance
- surface following
- contact-rich manipulation
- human-robot interaction
- medical robotics
- compliant assembly tasks

In such cases, position tracking alone is not enough. The controller must manage the tradeoff between motion accuracy and safe, stable interaction.

This project addresses that problem by simulating a manipulator in contact with a compliant environment and comparing impedance and admittance control strategies.

## Features

- 5-DOF serial manipulator with DH-based kinematics
- Forward kinematics and geometric Jacobian
- Structured simplified joint-space dynamics
- Spring-damper compliant wall model
- Contact force computation in Cartesian space
- Force-to-joint mapping through Jacobian transpose
- Cartesian impedance control
- Cartesian admittance reference generation
- Low-level joint tracking controller for admittance control
- Quantitative comparison of interaction behavior
- Research-style Jupyter notebook workflow

## Core Question

The project is designed to answer:

- How does a 5-DOF manipulator behave when interacting with a compliant environment?
- How do impedance and admittance control differ in force response and motion behavior?
- What tradeoffs appear between compliance, penetration, stability, and control effort?

## Environment Model

The environment is modeled as a compliant wall with a spring-damper contact law.

If the end-effector penetrates the wall, the environment produces a restoring force:

\[
F_c =
\begin{cases}
-k_e(x - x_{\text{wall}}) - b_e \dot{x}, & x > x_{\text{wall}} \\
0, & x \le x_{\text{wall}}
\end{cases}
\]

where:

- \( x_{\text{wall}} \) is the wall location
- \( k_e \) is wall stiffness
- \( b_e \) is wall damping

The resulting Cartesian contact force is mapped into joint torques using:

\[
\tau_c = J_v^T F_c
\]

## Manipulator Dynamics

The manipulator dynamics are modeled as:

\[
M(q)\ddot{q} + C(q,\dot{q})\dot{q} + G(q) + F(\dot{q}) = \tau - \tau_c
\]

where \( \tau_c \) represents the torque induced by environmental contact.

As in Project 1, the current model is structured and simplified rather than a full identified rigid-body model. This keeps the framework transparent and efficient while preserving the main interaction-relevant dynamics.

## Controllers

## 1. Impedance Control

Impedance control defines a virtual spring-damper relationship between position error and commanded Cartesian force:

\[
F_{\text{cmd}} = K_x(x_d - x) + D_x(\dot{x}_d - \dot{x})
\]

The corresponding joint torque is:

\[
\tau = J_v^T F_{\text{cmd}} + G(q)
\]

This controller directly shapes how the manipulator responds to position error in Cartesian space.

### Interpretation
- Higher stiffness improves tracking but reduces compliance
- Higher damping suppresses oscillations
- Good for studying force-position tradeoffs during contact

## 2. Admittance Control

Admittance control modifies the reference motion using measured contact force:

\[
M_a \ddot{x}_r + D_a \dot{x}_r + K_a(x_r - x_d) = F_c
\]

where:

- \( x_d \) is the nominal desired position
- \( x_r \) is the compliant reference
- \( F_c \) is measured contact force

The compliant Cartesian reference is converted into a joint-space reference using inverse kinematics, and then tracked by a lower-level joint controller.

### Interpretation
- Force drives reference motion
- Useful for compliant interaction through reference adaptation
- Often smoother when direct position tracking would be too aggressive

## Simulation Workflow

The main simulation loop is:

1. Compute the current end-effector position and velocity
2. Compute the contact force from the wall model
3. Map the contact force to joint torques
4. Compute the commanded control torque using impedance or admittance control
5. Integrate the manipulator dynamics using RK4
6. Store states, torques, contact force, and penetration depth
7. Plot and compare controller behavior

## Metrics

The project reports metrics such as:

- RMS position error
- Maximum contact force
- Mean contact force
- Maximum penetration depth
- Torque energy

These metrics help compare compliance and aggressiveness across controllers.

# 5-DOF MuJoCo Manipulator: RRT* Pick-and-Place with IK and PD Control

This project implements a simulated 5-DOF robotic manipulator in MuJoCo for a pick-and-place task with obstacle avoidance. The pipeline combines task-space RRT* motion planning, damped least-squares inverse kinematics, cubic joint trajectory generation, and PD control with MuJoCo bias compensation.


## Project Overview

The goal is to plan and execute a collision-avoiding transfer motion for a 5-DOF manipulator moving a block from an initial position to a target position.

The main pipeline is:

1. Build a 5-DOF serial manipulator in MuJoCo
2. Define pick, lift, transfer, place, and retreat waypoints
3. Use task-space RRT* to plan the transfer path around an obstacle
4. Convert Cartesian waypoints to joint configurations using damped least-squares IK
5. Generate smooth cubic joint-space trajectories
6. Track the trajectory using PD control with gravity/Coriolis bias compensation
7. Evaluate tracking error, torque commands, end-effector path, and final block placement error

## Methods

### RRT* Motion Planning

The transfer motion is planned in 3D task space using RRT*. The planner includes:

- Goal-biased random sampling
- Nearest-neighbor expansion
- Collision checking against inflated axis-aligned bounding boxes
- Parent selection based on path cost
- Rewiring for cost improvement
- Path simplification using shortcutting
- Cartesian path densification before IK




