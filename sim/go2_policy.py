"""NVIDIA's published IsaacLab Go2 flat policy, with explicit DC-motor PD control.

Contract: IsaacLab v2.3.0 UNITREE_GO2_CFG and velocity/config/go2/flat_env_cfg.py.
The model weights are downloaded separately from NVIDIA's published checkpoints.
"""
from __future__ import annotations

import numpy as np
import torch
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.rotations import quat_to_rot_matrix
from isaacsim.core.utils.types import ArticulationAction
from pxr import Gf, UsdPhysics, UsdGeom


POLICY_JOINTS = [f"{leg}_{joint}_joint" for joint in ("hip", "thigh", "calf") for leg in ("FL", "FR", "RL", "RR")]
DEFAULTS = np.array([.1, -.1, .1, -.1, .8, .8, 1., 1., -1.5, -1.5, -1.5, -1.5], dtype=np.float32)


class Go2Policy:
    def __init__(self, stage, asset, checkpoint, position):
        prim = stage.DefinePrim("/World/Go2", "Xform")
        prim.GetReferences().AddReference(str(asset))
        prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(*position))
        self.robot = SingleArticulation("/World/Go2/base", name="go2")
        saved = torch.load(str(checkpoint), map_location="cpu", weights_only=True)["model_state_dict"]
        self.actor = torch.nn.Sequential(torch.nn.Linear(48, 128), torch.nn.ELU(), torch.nn.Linear(128, 128),
                                         torch.nn.ELU(), torch.nn.Linear(128, 128), torch.nn.ELU(), torch.nn.Linear(128, 12))
        self.actor.load_state_dict({key[len("actor."):]: value for key, value in saved.items() if key.startswith("actor.")})
        self.actor.eval()
        self.previous = np.zeros(12, dtype=np.float32)
        self.target = DEFAULTS.copy()
        self.count = 0

    def initialize(self):
        self.robot.initialize()
        self.indices = np.array([self.robot.dof_names.index(name) for name in POLICY_JOINTS])
        controller = self.robot.get_articulation_controller()
        controller.set_effort_modes("force")
        controller.switch_control_mode("effort")
        self.robot._articulation_view.set_gains(np.zeros((1, 12)), np.zeros((1, 12)))
        self.robot._articulation_view.set_max_efforts(np.full((1, 12), 23.5))
        self.robot.set_joint_positions(DEFAULTS, joint_indices=self.indices)
        self.robot.set_joint_velocities(np.zeros(12), joint_indices=self.indices)
        self.robot.set_solver_position_iteration_count(8)
        self.robot.set_solver_velocity_iteration_count(2)

    def step(self, command, platform_velocity=0.0):
        position, quat = self.robot.get_world_pose()
        rotation = quat_to_rot_matrix(quat).T
        velocity = self.robot.get_linear_velocity().copy()
        velocity[2] -= platform_velocity
        q = self.robot.get_joint_positions()[self.indices]
        qd = self.robot.get_joint_velocities()[self.indices]
        if self.count % 4 == 0:
            obs = np.concatenate((rotation @ velocity, rotation @ self.robot.get_angular_velocity(),
                                  rotation @ np.array([0, 0, -1]), command,
                                  q - DEFAULTS, qd, self.previous)).astype(np.float32)
            with torch.inference_mode():
                self.previous = self.actor(torch.from_numpy(obs)).numpy()
            self.target = DEFAULTS + .25 * self.previous
        torque = 25.0 * (self.target - q) - .5 * qd
        upper = np.clip(23.5 * (1 - qd / 30), 0, 23.5)
        lower = np.clip(23.5 * (-1 - qd / 30), -23.5, 0)
        torque = np.clip(torque, lower, upper)
        self.robot.apply_action(ArticulationAction(joint_efforts=torque, joint_indices=self.indices))
        self.count += 1
