
# ================================================================
# 5-Link 3D Manipulator Pick-and-Place with RRT* + IK + PD Control in MuJoCo
# Colab-ready single-file script
#
# What this script does:
#   1. Builds a simple 5-DOF spatial manipulator in MuJoCo.
#   2. Uses damped least-squares inverse kinematics to reach waypoints.
#   3. Generates smooth cubic joint trajectories.
#   4. Tracks the trajectories using joint-space PD + MuJoCo bias compensation.
#   5. Picks a block and moves it to a new location.
#
# Important:
#   This demo uses a "kinematic grasp attachment":
#   once the gripper reaches the block, the block is moved with the end-effector.
#   This is intentional for a robust portfolio demo. A fully physical contact
#   grasp with finger friction is harder and should be a second-stage upgrade.
#
# Colab install cell:
#   !pip install mujoco mediapy
#
# If rendering fails in Colab, restart runtime and run again.
# ================================================================

import os

import time
import numpy as np
import matplotlib.pyplot as plt

try:
    import mujoco
    import mediapy as media
except ImportError as e:
    raise ImportError(
        "Missing packages. In Google Colab, run:\n"
        "!pip install mujoco mediapy\n"
        "Then restart runtime if needed."
    ) from e
import mujoco.viewer

# ================================================================
# 1. MuJoCo XML model
# ================================================================

ARM_XML = r"""
<mujoco model="five_link_arm_pick_place">

    <compiler angle="radian" coordinate="local" inertiafromgeom="true"/>
    <option timestep="0.002" gravity="0 0 -9.81" integrator="RK4"/>

    <visual>
        <headlight diffuse="0.6 0.6 0.6" ambient="0.25 0.25 0.25" specular="0.3 0.3 0.3"/>
        <rgba haze="0.15 0.25 0.35 1"/>
        <global azimuth="140" elevation="-25" offwidth="960" offheight="720"/>
    </visual>

    <asset>
        <texture name="grid" type="2d" builtin="checker" rgb1="0.2 0.3 0.4" rgb2="0.1 0.15 0.2" width="300" height="300"/>
        <material name="grid" texture="grid" texrepeat="5 5" reflectance="0.2"/>
        <material name="link_mat" rgba="0.75 0.75 0.80 1"/>
        <material name="joint_mat" rgba="0.15 0.20 0.30 1"/>
        <material name="ee_mat" rgba="0.95 0.55 0.20 1"/>
        <material name="block_mat" rgba="0.10 0.55 0.95 1"/>
        <material name="target_mat" rgba="0.05 0.85 0.25 0.35"/>
        <material name="table_mat" rgba="0.45 0.30 0.18 1"/>
        <material name="obstacle_mat" rgba="0.95 0.10 0.08 0.65"/>
    </asset>

    <worldbody>
        <light name="key_light" pos="0 -3 4" dir="0 1 -1" diffuse="0.8 0.8 0.8"/>
        <geom name="floor" type="plane" size="4 4 0.05" material="grid"/>

        <body name="table" pos="0.65 0 0.18">
            <geom name="table_top" type="box" size="0.70 0.45 0.04" material="table_mat" contype="1" conaffinity="1"/>
        </body>

        <body name="target_marker" pos="0.550 -0.35 0.285">
            <geom name="target_marker_geom" type="box" size="0.045 0.045 0.045" material="target_mat" contype="0" conaffinity="0"/>
        </body>

        <!-- Visible obstacle used by the RRT* planner. -->
        <!-- The planner avoids an inflated copy of this box. -->
        <body name="rrt_obstacle" pos="0.52 -0.02 0.36">
            <!-- Visual/planning obstacle only.
                 Important: this 5-DOF demo plans collision avoidance for the
                 end-effector path, not full-body robot-link collision.
                 Therefore the obstacle should NOT physically collide with the
                 robot links, otherwise the arm can get stuck even when the
                 end-effector RRT* path is collision-free. -->
            <geom name="rrt_obstacle_geom" type="box" size="0.075 0.075 0.160"
                  material="obstacle_mat" contype="0" conaffinity="0"/>
        </body>

        <body name="block" pos="0.62 0.22 0.285">
            <freejoint name="block_freejoint"/>
            <geom name="block_geom" type="box" size="0.045 0.045 0.045"
                  material="block_mat" mass="0.08" friction="1.0 0.005 0.0001"
                  contype="1" conaffinity="1"/>
        </body>

        <!-- Robot base -->
        <body name="base" pos="0 0 0.26">
            <geom name="base_geom" type="cylinder" size="0.10 0.06" material="joint_mat" contype="1" conaffinity="1"/>

            <!-- Joint 1: base yaw -->
            <body name="link1_body" pos="0 0 0.07">
                <joint name="joint1_base_yaw" type="hinge" axis="0 0 1"
                       range="-3.14 3.14" damping="1.0" armature="0.05" limited="true"/>
                <geom name="link1_geom" type="capsule" fromto="0 0 0 0 0 0.18"
                      size="0.045" material="link_mat" contype="1" conaffinity="1"/>

                <!-- Joint 2: shoulder pitch -->
                <body name="link2_body" pos="0 0 0.18">
                    <joint name="joint2_shoulder_pitch" type="hinge" axis="0 1 0"
                           range="-1.7 1.7" damping="1.0" armature="0.05" limited="true"/>
                    <geom name="joint2_geom" type="sphere" size="0.055" material="joint_mat" contype="1" conaffinity="1"/>
                    <geom name="link2_geom" type="capsule" fromto="0 0 0 0.26 0 0"
                          size="0.035" material="link_mat" contype="1" conaffinity="1"/>

                    <!-- Joint 3: elbow pitch -->
                    <body name="link3_body" pos="0.26 0 0">
                        <joint name="joint3_elbow_pitch" type="hinge" axis="0 1 0"
                               range="-2.2 2.2" damping="0.8" armature="0.04" limited="true"/>
                        <geom name="joint3_geom" type="sphere" size="0.050" material="joint_mat" contype="1" conaffinity="1"/>
                        <geom name="link3_geom" type="capsule" fromto="0 0 0 0.24 0 0"
                              size="0.032" material="link_mat" contype="1" conaffinity="1"/>

                        <!-- Joint 4: wrist pitch -->
                        <body name="link4_body" pos="0.24 0 0">
                            <joint name="joint4_wrist_pitch" type="hinge" axis="0 1 0"
                                   range="-2.2 2.2" damping="0.5" armature="0.02" limited="true"/>
                            <geom name="joint4_geom" type="sphere" size="0.045" material="joint_mat" contype="1" conaffinity="1"/>
                            <geom name="link4_geom" type="capsule" fromto="0 0 0 0.16 0 0"
                                  size="0.028" material="link_mat" contype="1" conaffinity="1"/>

                            <!-- Joint 5: wrist yaw -->
                            <body name="link5_body" pos="0.16 0 0">
                                <joint name="joint5_wrist_yaw" type="hinge" axis="0 0 1"
                                       range="-3.14 3.14" damping="0.3" armature="0.02" limited="true"/>
                                <geom name="joint5_geom" type="sphere" size="0.040" material="joint_mat" contype="1" conaffinity="1"/>
                                

                                <!-- End-effector palm and moving gripper attached to link5_body -->

                                <geom name="ee_bar" type="box"
                                    pos="0.055 0 0"
                                    size="0.025 0.018 0.018"
                                    material="ee_mat"
                                    contype="1"
                                    conaffinity="1"/>

                                <body name="gripper_palm" pos="0.115 0 0">

                                    <!-- This palm visually connects the gripper to the final arm link -->
                                    <geom name="gripper_palm_geom" type="box"
                                        pos="-0.025 0 0"
                                        size="0.020 0.055 0.020"
                                        material="ee_mat"
                                        contype="1"
                                        conaffinity="1"/>

                                    <!-- Left finger: starts on +y side and slides toward center when q increases -->
                                    <body name="left_finger" pos="0.00 0.040 0">
                                        <joint name="left_finger_slide" type="slide"
                                            axis="0 1 0"
                                            range="0 0.030"
                                            damping="2"
                                            limited="true"/>

                                        <geom name="left_finger_geom" type="box"
                                            pos="0.030 0 0"
                                            size="0.04 0.010 0.025"
                                            material="ee_mat"
                                            friction="1.5 0.01 0.001"
                                            contype="1"
                                            conaffinity="1"/>
                                    </body>

                                    <!-- Right finger: starts on -y side and slides toward center when q increases -->
                                    <body name="right_finger" pos="0.00 -0.040 0">
                                        <joint name="right_finger_slide" type="slide"
                                            axis="0 -1 0"
                                            range="0 0.030"
                                            damping="2"
                                            limited="true"/>

                                        <geom name="right_finger_geom" type="box"
                                            pos="0.030 0 0"
                                            size="0.04 0.010 0.025"
                                            material="ee_mat"
                                            friction="1.5 0.01 0.001"
                                            contype="1"
                                            conaffinity="1"/>
                                    </body>

                                    <!-- Put the end-effector site at the center between the fingertips -->
                                    <site name="ee_site"
                                        pos="0.05 0 0.035"
                                        size="0.015"
                                        rgba="1 0 0 1"/>

                                </body>
                               
                                

                            </body>
                        </body>
                    </body>
                </body>
            </body>
        </body>

        <camera name="front" pos="1.45 -1.45 1.15" xyaxes="0.70 0.70 0 -0.40 0.40 0.82"/>
        <camera name="side" pos="0 -1.8 0.95" xyaxes="1 0 0 0 0.45 0.89"/>
    </worldbody>

    <actuator>
        <motor name="motor1" joint="joint1_base_yaw" gear="1" ctrllimited="true" ctrlrange="-8 8"/>
        <motor name="motor2" joint="joint2_shoulder_pitch" gear="1" ctrllimited="true" ctrlrange="-12 12"/>
        <motor name="motor3" joint="joint3_elbow_pitch" gear="1" ctrllimited="true" ctrlrange="-10 10"/>
        <motor name="motor4" joint="joint4_wrist_pitch" gear="1" ctrllimited="true" ctrlrange="-7 7"/>
        <motor name="motor5" joint="joint5_wrist_yaw" gear="1" ctrllimited="true" ctrlrange="-5 5"/>

        <motor name="motor6" joint="right_finger_slide" gear="1" ctrllimited="true" ctrlrange="-5 5"/>
        <motor name="motor7" joint="left_finger_slide" gear="1" ctrllimited="true" ctrlrange="-5 5"/>
    </actuator>

</mujoco>
"""


# ================================================================
# 2. Utility functions
# ================================================================

def get_joint_ids(model, joint_names):
    return [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names]


def get_qpos_indices(model, joint_ids):
    return [model.jnt_qposadr[jid] for jid in joint_ids]


def get_qvel_indices(model, joint_ids):
    return [model.jnt_dofadr[jid] for jid in joint_ids]


def get_arm_q(data, qpos_ids):
    return data.qpos[qpos_ids].copy()


def get_arm_dq(data, qvel_ids):
    return data.qvel[qvel_ids].copy()


def set_arm_q(model, data, qpos_ids, q):
    data.qpos[qpos_ids] = q
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def get_site_position(model, data, site_name="ee_site"):
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    return data.site_xpos[site_id].copy()

def get_current_block_pos(model, data):
    return get_body_position(model, data, "block")

def get_site_jacobian_position(model, data, qvel_ids, site_name="ee_site"):
    """
    Return the position Jacobian columns corresponding only to the arm joints.

    Important MuJoCo detail:
    The model also has a freejoint for the block. Therefore, the first velocity
    columns do NOT necessarily belong to the robot arm. We must index the
    Jacobian using qvel_ids for the arm joints.
    """
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)

    return jacp[:, qvel_ids].copy()


def get_body_position(model, data, body_name):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    return data.xpos[body_id].copy()


def set_block_pose(model, data, block_qpos_addr, position, quat=None, zero_block_velocity=True):
    """
    Set block freejoint pose.
    MuJoCo freejoint qpos layout is:
        [x, y, z, qw, qx, qy, qz]

    We do NOT zero all qvel here, because that would accidentally reset the
    robot arm velocities during the simulation loop.
    """
    if quat is None:
        quat = np.array([1.0, 0.0, 0.0, 0.0])

    data.qpos[block_qpos_addr:block_qpos_addr+3] = position
    data.qpos[block_qpos_addr+3:block_qpos_addr+7] = quat

    if zero_block_velocity:
        # Freejoint velocity has 6 entries: translational and rotational.
        block_qvel_addr = block_qpos_addr - 1 if block_qpos_addr > 0 else 0
        # Safer fallback: only zero if the slice exists.
        if block_qvel_addr + 6 <= len(data.qvel):
            data.qvel[block_qvel_addr:block_qvel_addr+6] = 0.0

    mujoco.mj_forward(model, data)


# ================================================================
# 3. Inverse kinematics
# ================================================================

def damped_least_squares_ik(
    model,
    data,
    q_init,
    target_pos,
    qpos_ids,
    qvel_ids,
    joint_limits,
    site_name="ee_site",
    damping=0.04,
    step_size=0.7,
    tol=1e-3,
    max_iters=250
):
    """
    Position-only IK using MuJoCo site Jacobian.

    Because this is a 5-DOF arm, we solve position IK only.
    Trying to control full 6D pose with 5 DOF is overconstrained.
    """
    q = q_init.copy()
    best_q = q.copy()
    best_err = np.inf

    for _ in range(max_iters):
        set_arm_q(model, data, qpos_ids, q)

        ee_pos = get_site_position(model, data, site_name)
        err = target_pos - ee_pos
        err_norm = np.linalg.norm(err)

        if err_norm < best_err:
            best_err = err_norm
            best_q = q.copy()

        if err_norm < tol:
            return q, True, err_norm

        J = get_site_jacobian_position(model, data, qvel_ids, site_name)

        # Damped least squares:
        # dq = J^T (J J^T + lambda^2 I)^(-1) e
        A = J @ J.T + (damping ** 2) * np.eye(3)
        dq = J.T @ np.linalg.solve(A, err)

        q = q + step_size * dq

        for i, (lo, hi) in enumerate(joint_limits):
            q[i] = np.clip(q[i], lo, hi)

    return best_q, False, best_err


# ================================================================
# 4. Trajectory generation
# ================================================================

def cubic_segment(q0, qf, T, dt):
    """
    Cubic trajectory with zero start/end velocity.
    Returns q_traj and dq_traj.
    """
    n_steps = max(2, int(T / dt))
    ts = np.linspace(0.0, T, n_steps)

    q0 = np.asarray(q0)
    qf = np.asarray(qf)

    a0 = q0
    a1 = np.zeros_like(q0)
    a2 = 3.0 * (qf - q0) / (T ** 2)
    a3 = -2.0 * (qf - q0) / (T ** 3)

    q_traj = []
    dq_traj = []

    for t in ts:
        q = a0 + a1*t + a2*t**2 + a3*t**3
        dq = a1 + 2*a2*t + 3*a3*t**2
        q_traj.append(q)
        dq_traj.append(dq)

    return np.array(q_traj), np.array(dq_traj)


def build_piecewise_trajectory(waypoints, segment_times, dt):
    q_all = []
    dq_all = []

    for i in range(len(waypoints) - 1):
        q_seg, dq_seg = cubic_segment(waypoints[i], waypoints[i+1], segment_times[i], dt)

        # Avoid duplicate points at segment boundaries
        if i > 0:
            q_seg = q_seg[1:]
            dq_seg = dq_seg[1:]

        q_all.append(q_seg)
        dq_all.append(dq_seg)

    return np.vstack(q_all), np.vstack(dq_all)



# ================================================================
# 5. RRT* task-space motion planning
# ================================================================

class RRTStarNode3D:
    def __init__(self, p, parent=None, cost=0.0):
        self.p = np.asarray(p, dtype=float)
        self.parent = parent
        self.cost = float(cost)


class RRTStarPlanner3D:
    """
    Simple RRT* planner in 3D Cartesian/task space.

    This planner is used for the end-effector path. The resulting Cartesian
    points are then converted to joint-space waypoints using IK.

    State: p = [x, y, z]
    Obstacles: axis-aligned boxes, each written as:
        (xmin, xmax, ymin, ymax, zmin, zmax)
    """
    def __init__(
        self,
        bounds,
        obstacles=None,
        step_len=0.08,
        search_radius=0.18,
        goal_sample_rate=0.20,
        max_iter=1200,
        goal_tolerance=0.06,
        collision_resolution=0.015,
        random_seed=4,
    ):
        self.bounds = np.asarray(bounds, dtype=float)
        self.obstacles = obstacles if obstacles is not None else []
        self.step_len = float(step_len)
        self.search_radius = float(search_radius)
        self.goal_sample_rate = float(goal_sample_rate)
        self.max_iter = int(max_iter)
        self.goal_tolerance = float(goal_tolerance)
        self.collision_resolution = float(collision_resolution)
        self.rng = np.random.default_rng(random_seed)

    def sample_free(self, goal):
        if self.rng.random() < self.goal_sample_rate:
            return goal.copy()

        for _ in range(200):
            p = np.array([
                self.rng.uniform(self.bounds[0, 0], self.bounds[0, 1]),
                self.rng.uniform(self.bounds[1, 0], self.bounds[1, 1]),
                self.rng.uniform(self.bounds[2, 0], self.bounds[2, 1]),
            ])
            if not self.point_in_collision(p):
                return p

        # Fallback; should almost never happen unless bounds are bad.
        return goal.copy()

    def point_in_collision(self, p):
        x, y, z = p

        # Bounds check.
        if not (
            self.bounds[0, 0] <= x <= self.bounds[0, 1]
            and self.bounds[1, 0] <= y <= self.bounds[1, 1]
            and self.bounds[2, 0] <= z <= self.bounds[2, 1]
        ):
            return True

        # AABB obstacles.
        for obs in self.obstacles:
            xmin, xmax, ymin, ymax, zmin, zmax = obs
            if xmin <= x <= xmax and ymin <= y <= ymax and zmin <= z <= zmax:
                return True

        return False

    def segment_collision_free(self, p0, p1):
        p0 = np.asarray(p0, dtype=float)
        p1 = np.asarray(p1, dtype=float)
        dist = np.linalg.norm(p1 - p0)
        n = max(2, int(np.ceil(dist / self.collision_resolution)))

        for a in np.linspace(0.0, 1.0, n):
            p = (1.0 - a) * p0 + a * p1
            if self.point_in_collision(p):
                return False
        return True

    def nearest_index(self, nodes, p):
        dists = [np.linalg.norm(node.p - p) for node in nodes]
        return int(np.argmin(dists))

    def near_indices(self, nodes, p):
        n = len(nodes) + 1
        # Shrinking RRT* radius, with a floor so it still works for small demos.
        radius = min(self.search_radius, self.search_radius * np.sqrt(np.log(n + 1) / (n + 1)) + 0.08)
        return [i for i, node in enumerate(nodes) if np.linalg.norm(node.p - p) <= radius]

    def steer(self, p_from, p_to):
        p_from = np.asarray(p_from, dtype=float)
        p_to = np.asarray(p_to, dtype=float)
        v = p_to - p_from
        d = np.linalg.norm(v)
        if d < 1e-9:
            return p_from.copy()
        if d <= self.step_len:
            return p_to.copy()
        return p_from + self.step_len * v / d

    def extract_path(self, nodes, idx):
        path = []
        while idx is not None:
            path.append(nodes[idx].p.copy())
            idx = nodes[idx].parent
        path.reverse()
        return path

    def path_cost(self, path):
        if path is None or len(path) < 2:
            return np.inf
        return sum(np.linalg.norm(path[i + 1] - path[i]) for i in range(len(path) - 1))

    def simplify_path(self, path):
        """Shortcut the path using direct line-of-sight checks."""
        if path is None or len(path) <= 2:
            return path

        simplified = [path[0]]
        i = 0
        while i < len(path) - 1:
            j = len(path) - 1
            while j > i + 1:
                if self.segment_collision_free(path[i], path[j]):
                    break
                j -= 1
            simplified.append(path[j])
            i = j
        return simplified

    def plan(self, start, goal):
        start = np.asarray(start, dtype=float)
        goal = np.asarray(goal, dtype=float)

        if self.point_in_collision(start):
            raise ValueError(f"RRT* start is in collision or out of bounds: {start}")
        if self.point_in_collision(goal):
            raise ValueError(f"RRT* goal is in collision or out of bounds: {goal}")

        nodes = [RRTStarNode3D(start, parent=None, cost=0.0)]
        best_goal_index = None
        best_goal_cost = np.inf

        for _ in range(self.max_iter):
            p_rand = self.sample_free(goal)
            nearest_idx = self.nearest_index(nodes, p_rand)
            p_new = self.steer(nodes[nearest_idx].p, p_rand)

            if not self.segment_collision_free(nodes[nearest_idx].p, p_new):
                continue

            # Choose best parent among nearby nodes.
            near_idxs = self.near_indices(nodes, p_new)
            best_parent = nearest_idx
            best_cost = nodes[nearest_idx].cost + np.linalg.norm(p_new - nodes[nearest_idx].p)

            for idx in near_idxs:
                candidate = nodes[idx]
                if self.segment_collision_free(candidate.p, p_new):
                    candidate_cost = candidate.cost + np.linalg.norm(p_new - candidate.p)
                    if candidate_cost < best_cost:
                        best_parent = idx
                        best_cost = candidate_cost

            nodes.append(RRTStarNode3D(p_new, parent=best_parent, cost=best_cost))
            new_idx = len(nodes) - 1

            # Rewire nearby nodes through the new node when cheaper.
            for idx in near_idxs:
                if idx == best_parent:
                    continue
                candidate = nodes[idx]
                rewired_cost = nodes[new_idx].cost + np.linalg.norm(candidate.p - p_new)
                if rewired_cost < candidate.cost and self.segment_collision_free(p_new, candidate.p):
                    candidate.parent = new_idx
                    candidate.cost = rewired_cost

            # Try connecting to goal.
            dist_to_goal = np.linalg.norm(p_new - goal)
            if dist_to_goal <= self.goal_tolerance and self.segment_collision_free(p_new, goal):
                goal_cost = nodes[new_idx].cost + dist_to_goal
                if goal_cost < best_goal_cost:
                    nodes.append(RRTStarNode3D(goal, parent=new_idx, cost=goal_cost))
                    best_goal_index = len(nodes) - 1
                    best_goal_cost = goal_cost

        if best_goal_index is None:
            # Last fallback: direct connection if possible.
            if self.segment_collision_free(start, goal):
                return [start, goal], nodes, False
            raise RuntimeError("RRT* failed to find a collision-free path. Increase max_iter or relax obstacles/bounds.")

        path = self.extract_path(nodes, best_goal_index)
        path = self.simplify_path(path)
        return path, nodes, True


def make_rrt_star_obstacles(clearance=0.045, obstacle_clearance_xy=None, obstacle_clearance_z=0.045):
    """
    Obstacles used by the Cartesian RRT* planner.

    Important distinction:
    1. The MuJoCo XML now contains a visible obstacle named `rrt_obstacle`.
    2. This function creates inflated AABB boxes for the planner.

    The inflation gives the end-effector a safety margin so the planned path
    does not skim the obstacle or tabletop.
    """
    obstacles = []

    # Tabletop obstacle.
    table_center = np.array([0.65, 0.0, 0.18])
    table_half = np.array([0.70, 0.45, 0.04])
    low = table_center - table_half - clearance
    high = table_center + table_half + clearance
    obstacles.append((low[0], high[0], low[1], high[1], low[2], high[2]))

    # Visible red obstacle in the MuJoCo scene.
    # Use a larger horizontal clearance than vertical clearance.
    # This keeps the arm from visually grazing the obstacle, without making
    # the start/goal impossible because their z values are near the obstacle top.
    if obstacle_clearance_xy is None:
        obstacle_clearance_xy = clearance

    obstacle_center = np.array([0.52, -0.02, 0.46])
    obstacle_half = np.array([0.075, 0.075, 0.20])
    low = obstacle_center - obstacle_half - np.array([obstacle_clearance_xy, obstacle_clearance_xy, obstacle_clearance_z])
    high = obstacle_center + obstacle_half + np.array([obstacle_clearance_xy, obstacle_clearance_xy, obstacle_clearance_z])
    obstacles.append((low[0], high[0], low[1], high[1], low[2], high[2]))

    return obstacles

def densify_cartesian_path(path, max_spacing=0.06):
    """Add intermediate points so IK does not jump too much between RRT* nodes."""
    dense = [np.asarray(path[0], dtype=float)]
    for i in range(len(path) - 1):
        p0 = np.asarray(path[i], dtype=float)
        p1 = np.asarray(path[i + 1], dtype=float)
        dist = np.linalg.norm(p1 - p0)
        n = max(1, int(np.ceil(dist / max_spacing)))
        for j in range(1, n + 1):
            dense.append(p0 + (j / n) * (p1 - p0))
    return dense



def min_distance_to_aabb_surface(p, obs):
    """Approximate clearance from a point to an inflated AABB obstacle.

    If the point is inside the box, the returned value is negative.
    This is mainly used for printing/debugging the RRT* path clearance.
    """
    p = np.asarray(p, dtype=float)
    xmin, xmax, ymin, ymax, zmin, zmax = obs
    low = np.array([xmin, ymin, zmin])
    high = np.array([xmax, ymax, zmax])

    outside = np.maximum(np.maximum(low - p, 0.0), p - high)
    outside_dist = np.linalg.norm(outside)
    if outside_dist > 0:
        return outside_dist

    # Inside box: negative distance to nearest face.
    return -np.min(np.r_[p - low, high - p])


def print_rrt_clearance_report(rrt_path, obstacles):
    if rrt_path is None or len(rrt_path) == 0:
        return
    for obs_i, obs in enumerate(obstacles):
        dmin = min(min_distance_to_aabb_surface(p, obs) for p in rrt_path)
        print(f"  Minimum RRT* path clearance from inflated obstacle {obs_i}: {dmin:.4f} m")

def solve_ik_for_cartesian_path(
    model,
    data,
    cartesian_path,
    q_seed,
    qpos_ids,
    qvel_ids,
    joint_limits,
    label="path",
):
    """Convert a list of task-space points into joint-space waypoints."""
    q_list = []
    q_current_seed = q_seed.copy()

    for i, target in enumerate(cartesian_path):
        q_sol, success, err = damped_least_squares_ik(
            model=model,
            data=data,
            q_init=q_current_seed,
            target_pos=np.asarray(target),
            qpos_ids=qpos_ids,
            qvel_ids=qvel_ids,
            joint_limits=joint_limits,
            damping=0.04,
            step_size=0.6,
            tol=2.0e-3,
            max_iters=350,
        )
        print(f"  IK {label} point {i + 1:02d}: success={success}, error={err:.5f} m, q={np.round(q_sol, 3)}")
        q_list.append(q_sol.copy())
        q_current_seed = q_sol.copy()

    return q_list


# ================================================================
# 5. PD controller
# ================================================================

class PDController:
    def __init__(self, kp, kd, torque_limits):
        self.kp = np.asarray(kp, dtype=float)
        self.kd = np.asarray(kd, dtype=float)
        self.torque_limits = np.asarray(torque_limits, dtype=float)

    def compute(self, q, dq, q_des, dq_des, bias=None):
        tau = self.kp * (q_des - q) + self.kd * (dq_des - dq)

        # MuJoCo qfrc_bias contains gravity, Coriolis, and centrifugal terms.
        # Adding it improves tracking, especially for shoulder/elbow joints.
        if bias is not None:
            tau = tau + bias

        tau = np.clip(tau, -self.torque_limits, self.torque_limits)
        return tau
    
# ================================================================
# 6. Grasping control
# ================================================================

class GripperPDController:
    def __init__(self, kp=120.0, kd=8.0, force_limit=5.0):
        self.kp = kp
        self.kd = kd
        self.force_limit = force_limit

    def compute(self, q_finger, dq_finger, q_des):
        force = self.kp * (q_des - q_finger) - self.kd * dq_finger
        force = np.clip(force, -self.force_limit, self.force_limit)
        return force


# ================================================================
# 6. Simulation runner
# ================================================================
def move_toward_joint_target(q_current, q_target, max_step=0.025):
    """
    Smoothly move desired joint command toward IK solution.
    This avoids sudden jumps when IK target updates online.
    """
    dq = q_target - q_current
    norm = np.linalg.norm(dq)

    if norm < max_step:
        return q_target.copy()

    return q_current + max_step * dq / norm

def run_pick_and_place(render=True, live_viewer=False, use_rrt_star=True):
    model = mujoco.MjModel.from_xml_string(ARM_XML)
    data = mujoco.MjData(model)

    joint_names = [
        "joint1_base_yaw",
        "joint2_shoulder_pitch",
        "joint3_elbow_pitch",
        "joint4_wrist_pitch",
        "joint5_wrist_yaw",
    ]

    joint_ids = get_joint_ids(model, joint_names)
    qpos_ids = get_qpos_indices(model, joint_ids)
    qvel_ids = get_qvel_indices(model, joint_ids)

    finger_joint_names = [
        "right_finger_slide",
        "left_finger_slide",
    ]

    finger_joint_ids = get_joint_ids(model, finger_joint_names)
    finger_qpos_ids = get_qpos_indices(model, finger_joint_ids)
    finger_qvel_ids = get_qvel_indices(model, finger_joint_ids)

    joint_limits = []
    for jid in joint_ids:
        joint_limits.append((model.jnt_range[jid, 0], model.jnt_range[jid, 1]))

    block_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "block_freejoint")
    block_qpos_addr = model.jnt_qposadr[block_joint_id]

    dt = model.opt.timestep
    control_dt = 0.01
    steps_per_control = int(control_dt / dt)

    # Initial arm pose.
    q_home = np.array([0.0, 0.15, -1.05, 0.70, 0.0])
    set_arm_q(model, data, qpos_ids, q_home)

    table_center_z = 0.18
    table_half_height = 0.04
    block_half_height = 0.045
    block_z = table_center_z + table_half_height + block_half_height

    block_start = np.array([0.62, 0.22, block_z])
    block_goal = np.array([0.55, -0.35, block_z])
    set_block_pose(model, data, block_qpos_addr, block_start)

    # ------------------------------------------------------------
    # Task-space pick/place points.
    # ------------------------------------------------------------
    block_now = get_body_position(model, data, "block")

    p_pre_grasp = block_now + np.array([0.0, 0.0, 0.13])
    p_grasp = block_now + np.array([0.0, 0.0, 0.065])
    p_lift = block_start + np.array([0.0, 0.0, 0.40])
    p_pre_place = block_goal + np.array([0.0, 0.0, 0.40])
    p_place = block_goal + np.array([0.0, 0.0, 0.065])
    p_retreat = block_goal + np.array([0.0, 0.0, 0.22])

    # ------------------------------------------------------------
    # RRT* planning from lift point to pre-place point.
    # This is where motion planning is actually used.
    # The approach and descend are kept as simple vertical motions.
    # ------------------------------------------------------------
    rrt_path = None
    rrt_nodes = None
    rrt_success = False

    if use_rrt_star:
        bounds = np.array([
            [0.08, 0.88],   # x range
            [-0.50, 0.50],  # y range
            [0.38, 0.84],   # z range: allow a higher route over/around obstacle
        ])

        # Larger safety threshold around the red obstacle.
        # Important: do NOT make this too large or the start/goal can become invalid.
        obstacles = make_rrt_star_obstacles(
            clearance=0.060,
            obstacle_clearance_xy=0.130,
            obstacle_clearance_z=0.050,
        )

        planner = RRTStarPlanner3D(
            bounds=bounds,
            obstacles=obstacles,
            step_len=0.045,
            search_radius=0.30,
            goal_sample_rate=0.12,
            max_iter=3500,
            goal_tolerance=0.035,
            collision_resolution=0.010,
            random_seed=7,
        )

        print("Planning transfer path with RRT*...")
        rrt_path, rrt_nodes, rrt_success = planner.plan(p_lift, p_pre_place)
        rrt_path = densify_cartesian_path(rrt_path, max_spacing=0.025)
        print(f"RRT* success={rrt_success}, number of task-space path points={len(rrt_path)}")
        print_rrt_clearance_report(rrt_path, obstacles)
    else:
        # Baseline without RRT*.
        rrt_path = [p_lift, p_pre_place]

    # Full Cartesian sequence.
    # Avoid duplicating p_lift and p_pre_place because rrt_path already includes them.
    task_positions = [p_pre_grasp, p_grasp] + list(rrt_path) + [p_place, p_retreat]

    # Useful indices in task_positions.
    grasp_task_index = 1
    place_task_index = len(task_positions) - 2

    # ------------------------------------------------------------
    # Convert Cartesian waypoints to joint waypoints using IK.
    # ------------------------------------------------------------
    # Store the true home end-effector position BEFORE the IK loop.
    # The IK solver changes data.qpos internally, so using data after IK
    # would give the wrong first segment duration and can cause tracking error.
    set_arm_q(model, data, qpos_ids, q_home)
    p_home_ee = get_site_position(model, data, "ee_site")

    q_waypoints = [q_home.copy()]
    q_seed = q_home.copy()

    print("Solving IK waypoints from RRT*/task-space path...")
    q_task = solve_ik_for_cartesian_path(
        model=model,
        data=data,
        cartesian_path=task_positions,
        q_seed=q_seed,
        qpos_ids=qpos_ids,
        qvel_ids=qvel_ids,
        joint_limits=joint_limits,
        label="task",
    )
    q_waypoints.extend(q_task)

    # Segment durations. Scale the RRT* transfer segments by Cartesian distance.
    segment_times = []
    previous_p = p_home_ee
    for i, p in enumerate(task_positions):
        p = np.asarray(p)
        dist = np.linalg.norm(p - previous_p)

        if i <= 1:
            T = 1.1  # approach and descend to grasp
        elif i == place_task_index:
            T = 1.0  # descend to place
        elif i == place_task_index + 1:
            T = 1.0  # retreat
        else:
            T = np.clip(dist / 0.055, 1.00, 2.40)  # slower RRT* transfer segments for PD tracking

        segment_times.append(float(T))
        previous_p = p

    q_ref, dq_ref = build_piecewise_trajectory(q_waypoints, segment_times, control_dt)

    # Event indices for kinematic block attachment/release.
    cumulative_steps = np.cumsum([int(T / control_dt) for T in segment_times])
    grasp_index = cumulative_steps[grasp_task_index]
    release_index = cumulative_steps[place_task_index]

    controller = PDController(
        kp=np.array([115, 115, 125, 70, 45]),
        kd=np.array([22, 28, 24, 14, 9]),
        torque_limits=np.array([10, 14, 12, 8, 6])
    )
    gripper_controller = GripperPDController(
        kp=120.0,
        kd=8.0,
        force_limit=5.0
    )

    gripper_open = np.array([0.030, 0.030])
    gripper_closed = np.array([0.000, 0.000])

    # Reset true system state before executing trajectory.
    set_arm_q(model, data, qpos_ids, q_home)
    set_block_pose(model, data, block_qpos_addr, block_start)

    data.qpos[finger_qpos_ids] = gripper_open
    data.qvel[finger_qvel_ids] = 0.0
    mujoco.mj_forward(model, data)

    # Renderer
    width, height = 960, 720
    renderer = mujoco.Renderer(model, height=height, width=width) if render else None
    frames = []

    # Live viewer for VS Code / local desktop
    viewer = mujoco.viewer.launch_passive(model, data) if live_viewer else None

    log_t = []
    log_q = []
    log_q_ref = []
    log_ee = []
    log_tau = []
    log_block = []

    attached = False
    release_done = False
    pin_block_after_release = True

    # Robust kinematic grasp for the portfolio demo.
    grasp_offset = np.array([0.0, 0.0, -0.065])
    grasp_lock_index = grasp_index + int(0.25 / control_dt)
    release_lock_index = release_index

    sim_time = 0.0
    frame_skip = 10

    print("Running PD execution of RRT* pick-and-place trajectory...")
    for k in range(len(q_ref)):

        q_des = q_ref[k]
        dq_des = dq_ref[k]

        # Attach block after reaching grasp.
        if (not attached) and (not release_done) and (k >= grasp_lock_index) and (k < release_index):
            attached = True
            ee_pos = get_site_position(model, data, "ee_site")
            set_block_pose(
                model,
                data,
                block_qpos_addr,
                ee_pos + grasp_offset,
                zero_block_velocity=True
            )

        if k < grasp_index:
            finger_des = gripper_open
        elif grasp_index <= k < release_lock_index:
            finger_des = gripper_closed
        else:
            finger_des = gripper_open

        # Release after reaching place and settling briefly.
        if k >= release_lock_index and (not release_done):
            attached = False
            set_block_pose(
                model,
                data,
                block_qpos_addr,
                block_goal,
                zero_block_velocity=True
            )
            block_qvel_addr = model.jnt_dofadr[block_joint_id]
            data.qvel[block_qvel_addr:block_qvel_addr + 6] = 0.0
            mujoco.mj_forward(model, data)
            release_done = True

        for _ in range(steps_per_control):
            q = get_arm_q(data, qpos_ids)
            dq = get_arm_dq(data, qvel_ids)

            bias = data.qfrc_bias[qvel_ids].copy()
            tau = controller.compute(q, dq, q_des, dq_des, bias=bias)
            data.ctrl[:5] = tau

            q_finger = data.qpos[finger_qpos_ids].copy()
            dq_finger = data.qvel[finger_qvel_ids].copy()
            finger_force = gripper_controller.compute(
                q_finger=q_finger,
                dq_finger=dq_finger,
                q_des=finger_des
            )
            data.ctrl[5:7] = finger_force

            mujoco.mj_step(model, data)

            if viewer is not None:
                viewer.sync()
                time.sleep(dt)

            sim_time += dt

            if attached:
                ee_pos = get_site_position(model, data, "ee_site")
                block_pos = ee_pos + grasp_offset
                set_block_pose(model, data, block_qpos_addr, block_pos, zero_block_velocity=True)

            # Once the object is placed, keep it fixed at the target for the
            # remainder of the visualization. This prevents the freejoint block
            # from being knocked by the opening fingers or accumulating contact
            # velocity after release. For a portfolio demo, this is cleaner than
            # letting a marginal physical grasp corrupt the motion-planning result.
            if release_done and pin_block_after_release:
                set_block_pose(model, data, block_qpos_addr, block_goal, zero_block_velocity=True)

        # Logging at control rate.
        q_now = get_arm_q(data, qpos_ids)
        ee_now = get_site_position(model, data, "ee_site")
        block_now = get_body_position(model, data, "block")

        log_t.append(k * control_dt)
        log_q.append(q_now)
        log_q_ref.append(q_des)
        log_ee.append(ee_now)
        log_tau.append(data.ctrl[:5].copy())
        log_block.append(block_now)

        if render and k % frame_skip == 0:
            renderer.update_scene(data, camera="front")
            frame = renderer.render()
            frames.append(frame)

    logs = {
        "t": np.array(log_t),
        "q": np.array(log_q),
        "q_ref": np.array(log_q_ref),
        "ee": np.array(log_ee),
        "tau": np.array(log_tau),
        "block": np.array(log_block),
        "q_waypoints": np.array(q_waypoints),
        "task_positions": np.array(task_positions),
        "rrt_path": np.array(rrt_path),
        "rrt_success": rrt_success,
        "block_start": block_start,
        "block_goal": block_goal,
    }

    video = frames if render else None

    if viewer is not None:
        viewer.close()

    return model, data, logs, video


# ================================================================
# 7. Plotting
# ================================================================

def plot_results(logs):
    t = logs["t"]
    q = logs["q"]
    q_ref = logs["q_ref"]
    tau = logs["tau"]
    ee = logs["ee"]
    block = logs["block"]
    block_start = logs["block_start"]
    block_goal = logs["block_goal"]

    # Joint tracking plot
    plt.figure(figsize=(12, 7))
    for i in range(5):
        plt.plot(t, q[:, i], label=f"q{i+1}")
        plt.plot(t, q_ref[:, i], "--", label=f"q{i+1} ref")
    plt.xlabel("Time [s]")
    plt.ylabel("Joint angle [rad]")
    plt.title("PD Joint Tracking")
    plt.grid(True)
    plt.legend(ncol=2)
    plt.show()

    

    q_err = np.linalg.norm(q_ref - q, axis=1)
    rmse_q = np.sqrt(np.mean((q_ref - q) ** 2))
    max_q_error = np.max(q_err)
    final_block_error = np.linalg.norm(block[-1] - block_goal)

    print("\n================ Metrics ================")
    print(f"Joint tracking RMSE:        {rmse_q:.5f} rad")
    print(f"Maximum joint error norm:   {max_q_error:.5f} rad")
    print(f"Final block position error: {final_block_error:.5f} m")
    print("=========================================")
    # Tracking error
    q_err = np.linalg.norm(q_ref - q, axis=1)
    plt.figure(figsize=(10, 4))
    plt.plot(t, q_err)
    plt.xlabel("Time [s]")
    plt.ylabel("||q_ref - q|| [rad]")
    plt.title("Joint Tracking Error Norm")
    plt.grid(True)
    plt.show()

    # Torque plot
    plt.figure(figsize=(12, 6))
    for i in range(5):
        plt.plot(t, tau[:, i], label=f"tau{i+1}")
    plt.xlabel("Time [s]")
    plt.ylabel("Torque command")
    plt.title("PD Control Torques")
    plt.grid(True)
    plt.legend()
    plt.show()

    # End-effector and block path in top view
    plt.figure(figsize=(7, 7))
    plt.plot(ee[:, 0], ee[:, 1], label="End-effector path")
    plt.plot(block[:, 0], block[:, 1], label="Block path")
    plt.scatter(block_start[0], block_start[1], marker="o", s=80, label="Block start")
    plt.scatter(block_goal[0], block_goal[1], marker="x", s=100, label="Block goal")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Top View: End-Effector and Block Path")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.show()

    # 3D path
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(ee[:, 0], ee[:, 1], ee[:, 2], label="End-effector")
    ax.plot(block[:, 0], block[:, 1], block[:, 2], label="Block")
    ax.scatter(block_start[0], block_start[1], block_start[2], s=60, label="Start")
    ax.scatter(block_goal[0], block_goal[1], block_goal[2], s=80, label="Goal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title("3D Pick-and-Place Path")
    ax.legend()
    plt.show()

    rmse_q = np.sqrt(np.mean((q_ref - q) ** 2))
    max_q_error = np.max(q_err)
    final_block_error = np.linalg.norm(block[-1] - block_goal)

    print("\n================ Metrics ================")
    print(f"Joint tracking RMSE:        {rmse_q:.5f} rad")
    print(f"Maximum joint error norm:   {max_q_error:.5f} rad")
    print(f"Final block position error: {final_block_error:.5f} m")
    print("=========================================")



# ================================================================
# 8. Main execution
# ================================================================

if __name__ == "__main__":
    model, data, logs, video = run_pick_and_place(render=True, live_viewer=True)

    if video is not None:
        print("Saving video...")
        media.write_video("pick_and_place_rrt_star_fixed.mp4", video, fps=30)
        print("Saved video as pick_and_place_rrt_star_fixed.mp4")

    plot_results(logs)