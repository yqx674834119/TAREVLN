"""Official Isaac Lab articulation/actuator path, exposed to the Office adapter."""
import builtins
import numpy as np
import torch
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.types import ArticulationAction
from isaaclab.assets import Articulation
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG
from go2_policy import POLICY_JOINTS


def numpy(value):
    return value.detach().cpu().numpy() if isinstance(value,torch.Tensor) else value


class NumpyReader(SingleArticulation):
    def get_world_pose(self):return tuple(numpy(x) for x in super().get_world_pose())
    def get_linear_velocity(self):return numpy(super().get_linear_velocity())
    def get_angular_velocity(self):return numpy(super().get_angular_velocity())
    def get_joint_positions(self,*a,**kw):return numpy(super().get_joint_positions(*a,**kw))
    def get_joint_velocities(self,*a,**kw):return numpy(super().get_joint_velocities(*a,**kw))


class Go2LabPolicy:
    def __init__(self,stage,asset,checkpoint,position):
        builtins.ISAACLAB_CALLBACK_EXCEPTION=None
        cfg=UNITREE_GO2_CFG.replace(prim_path='/World/Go2')
        cfg.spawn.usd_path=str(asset)
        cfg.init_state.pos=tuple(position)
        self.lab=Articulation(cfg)
        self.robot=NumpyReader('/World/Go2/base',name='go2')
        self.actor=torch.nn.Sequential(torch.nn.Linear(48,128),torch.nn.ELU(),torch.nn.Linear(128,128),
                                       torch.nn.ELU(),torch.nn.Linear(128,128),torch.nn.ELU(),torch.nn.Linear(128,12))
        saved=torch.load(str(checkpoint),map_location='cpu',weights_only=True)['model_state_dict']
        self.actor.load_state_dict({k[6:]:v for k,v in saved.items() if k.startswith('actor.')})
        self.actor.eval()
        self.previous=np.zeros(12,dtype=np.float32)
        self.count=0

    def initialize(self):
        if builtins.ISAACLAB_CALLBACK_EXCEPTION:raise builtins.ISAACLAB_CALLBACK_EXCEPTION
        self.robot.initialize()
        if self.lab.joint_names!=POLICY_JOINTS:raise ValueError('Unexpected Isaac Lab joint order')
        self.indices=np.arange(12)
        self.lab.write_root_state_to_sim(self.lab.data.default_root_state.clone())
        self.lab.write_joint_state_to_sim(self.lab.data.default_joint_pos.clone(),self.lab.data.default_joint_vel.clone())
        self.lab.reset()
        self.target=numpy(self.lab.data.default_joint_pos[0]).copy()
        materials=self.lab.root_physx_view.get_material_properties().clone()
        materials[:]=torch.tensor([.8,.6,0.])
        self.lab.root_physx_view.set_material_properties(materials,torch.tensor([0],dtype=torch.int32))
        self.robot.get_applied_action=lambda:ArticulationAction(joint_efforts=numpy(self.lab.data.applied_torque[0]))

    def step(self,command,platform_velocity=0.):
        self.lab.update(.005)
        d=self.lab.data
        if platform_velocity:raise ValueError('Moving platforms are outside this single-scene adapter')
        if self.count%4==0:
            obs=torch.cat([d.root_lin_vel_b,d.root_ang_vel_b,d.projected_gravity_b,
                           torch.as_tensor(command,dtype=torch.float32).reshape(1,3),
                           d.joint_pos-d.default_joint_pos,d.joint_vel,
                           torch.from_numpy(self.previous).reshape(1,12)],dim=-1)
            with torch.inference_mode():action=self.actor(obs)
            self.previous=numpy(action[0]).copy()
            target=d.default_joint_pos+.25*action
            self.target=numpy(target[0]).copy()
            self.lab.set_joint_position_target(target)
        self.lab.write_data_to_sim()
        self.count+=1
