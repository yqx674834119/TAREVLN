"""Simulation-only Go2 policy refinement for navigation-scale velocity commands.

Upstream checkpoint stays read-only. The trained joint controller is never a
real-robot deployment artifact; hardware uses the manufacturer's high-level API.
"""
import argparse
import json
from pathlib import Path
import shutil
import traceback

from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--out", required=True)
parser.add_argument("--iterations", type=int, default=500)
parser.add_argument("--num-envs", type=int, default=2048)
parser.add_argument("--initial-checkpoint", type=Path)
parser.add_argument("--profile", choices=("v1", "turning_v2", "regularized_v3", "mesh_v4", "rails_v5"), default="v1")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args,multi_gpu=False).app
try:
    import torch
    from isaaclab.envs import ManagerBasedRLEnv, mdp
    from isaaclab.envs.mdp.commands import UniformVelocityCommand
    from isaaclab.managers import RewardTermCfg, SceneEntityCfg, TerminationTermCfg
    from isaaclab.utils.io import dump_yaml
    from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.flat_env_cfg import UnitreeGo2FlatEnvCfg
    from isaaclab_tasks.manager_based.locomotion.velocity import mdp as locomotion_mdp
    from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.agents.rsl_rl_ppo_cfg import UnitreeGo2FlatPPORunnerCfg
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner

    class NavigationVelocityCommand(UniformVelocityCommand):
        def _resample_command(self, env_ids):
            super()._resample_command(env_ids)
            ids = torch.as_tensor(env_ids, device=self.device)
            category = torch.randint(0, 5, (len(ids),), device=self.device)
            if args.profile in ("turning_v2", "regularized_v3", "mesh_v4", "rails_v5"):
                # 10% stand, 20% longitudinal, 20% lateral, 40% pure yaw, 10% mixed.
                category = torch.tensor([0,1,1,2,2,3,3,3,3,4],device=self.device)[torch.randint(0,10,(len(ids),),device=self.device)]
            # Equal probabilities for stand, longitudinal, lateral, yaw, mixed.
            self.vel_command_b[ids[category == 0]] = 0.
            for category_id, axis in ((1,0), (2,1), (3,2)):
                selected = ids[category == category_id]
                for other in range(3):
                    if other != axis:
                        self.vel_command_b[selected, other] = 0.
            self.is_standing_env[ids] = category == 0
            if args.profile in ("turning_v2", "regularized_v3", "mesh_v4", "rails_v5"):
                turning = ids[category == 3]
                yaw = self.vel_command_b[turning,2]
                self.vel_command_b[turning,2] = torch.sign(yaw)*(.3+.7*torch.rand(len(turning),device=self.device))

    def navigation_air_time(env, threshold, sensor_cfg, command_name):
        sensor = env.scene.sensors[sensor_cfg.name]
        first_contact = sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
        last_air_time = sensor.data.last_air_time[:, sensor_cfg.body_ids]
        reward = torch.sum((last_air_time-threshold)*first_contact, dim=1)
        command = env.command_manager.get_command(command_name)
        return reward*(torch.linalg.vector_norm(command, dim=1) > .08)

    root = Path(__file__).resolve().parents[2]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, out / Path(__file__).name)
    cfg = UnitreeGo2FlatEnvCfg()
    cfg.seed = 41
    cfg.sim.device = args.device
    cfg.scene.num_envs = args.num_envs
    cfg.scene.robot.spawn.usd_path = str(root / "assets/go2/go2_lab.usd")
    cfg.scene.terrain.visual_material = None
    cfg.scene.sky_light.spawn.texture_file = None
    if args.profile in ("mesh_v4","rails_v5"):
        from isaaclab.terrains import TerrainGeneratorCfg, MeshPlaneTerrainCfg, MeshRandomGridTerrainCfg, MeshRailsTerrainCfg
        terrains={'flat_mesh':MeshPlaneTerrainCfg(proportion=.5),
                  'low_grid':MeshRandomGridTerrainCfg(proportion=.5,grid_width=1.1,
                    grid_height_range=(0.,.02),platform_width=1.)}
        if args.profile=='rails_v5':
            # Change terrain exposure only. Rewards, commands and the actor
            # remain identical to v4; narrow rail widths/heights vary.
            terrains={'flat_mesh':MeshPlaneTerrainCfg(proportion=.5),
                      'low_rails':MeshRailsTerrainCfg(proportion=.5,rail_thickness_range=(.08,.15),
                        rail_height_range=(.01,.04),platform_width=1.8)}
        cfg.scene.terrain.terrain_type='generator'
        cfg.scene.terrain.terrain_generator=TerrainGeneratorCfg(seed=41,size=(12.,12.),
            num_rows=10,num_cols=10,border_width=20.,curriculum=False,use_cache=False,
            sub_terrains=terrains)
    cfg.commands.base_velocity.class_type = NavigationVelocityCommand
    cfg.commands.base_velocity.heading_command = False
    cfg.commands.base_velocity.rel_heading_envs = 0.
    cfg.commands.base_velocity.rel_standing_envs = 0.
    cfg.commands.base_velocity.debug_vis = False
    cfg.commands.base_velocity.resampling_time_range = (3., 6.)
    ranges = cfg.commands.base_velocity.ranges
    ranges.lin_vel_x = (-.45,.5)
    ranges.lin_vel_y = (-.3,.3)
    ranges.ang_vel_z = (-.8,.8)
    cfg.rewards.track_lin_vel_xy_exp.weight = 2.5
    cfg.rewards.track_ang_vel_z_exp.weight = 2.
    cfg.rewards.base_height = RewardTermCfg(func=mdp.base_height_l2, weight=-30., params={"target_height": .30})
    cfg.rewards.undesired_contacts = RewardTermCfg(func=mdp.undesired_contacts, weight=-2.,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_thigh", ".*_calf", "base", "Head.*"]), "threshold": 1.})
    cfg.rewards.feet_air_time.func = navigation_air_time
    cfg.rewards.feet_air_time.weight = .5
    cfg.rewards.feet_air_time.params["threshold"] = .15
    cfg.rewards.hip_deviation = RewardTermCfg(func=mdp.joint_deviation_l1, weight=-.05,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint"])})
    cfg.terminations.bad_orientation = TerminationTermCfg(func=mdp.bad_orientation, params={"limit_angle": .7})
    if args.profile in ("turning_v2", "regularized_v3", "mesh_v4", "rails_v5"):
        cfg.rewards.track_ang_vel_z_exp.weight = 6.
        cfg.rewards.feet_air_time.weight = 2.
        cfg.rewards.feet_slide = RewardTermCfg(func=locomotion_mdp.feet_slide,weight=-.5,
            params={"sensor_cfg":SceneEntityCfg("contact_forces",body_names=".*_foot"),
                    "asset_cfg":SceneEntityCfg("robot",body_names=".*_foot")})
    if args.profile in ("regularized_v3", "mesh_v4", "rails_v5"):
        # v2 exploited unbounded target offsets and relied on exploration noise.
        # Penalize those offsets; sharply distinguish low-speed tracking from standing.
        cfg.rewards.action_magnitude = RewardTermCfg(func=mdp.action_l2,weight=-.01)
        cfg.rewards.track_lin_vel_xy_exp.weight = 4.
        cfg.rewards.track_lin_vel_xy_exp.params['std'] = .2
        cfg.rewards.track_ang_vel_z_exp.params['std'] = .3
    agent = UnitreeGo2FlatPPORunnerCfg()
    agent.seed = 41
    agent.device = args.device
    agent.algorithm.learning_rate = 5e-4
    agent.save_interval = 50
    if args.profile == "turning_v2":
        agent.algorithm.entropy_coef = .03
    elif args.profile in ("regularized_v3", "mesh_v4", "rails_v5"):
        agent.algorithm.entropy_coef = .003
    env = ManagerBasedRLEnv(cfg=cfg)
    wrapper = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(wrapper, agent.to_dict(), log_dir=str(out), device=args.device)
    checkpoint = args.initial_checkpoint or root / "assets/go2/checkpoint.pt"
    runner.load(str(checkpoint), load_optimizer=False)
    if args.profile in ("turning_v2", "regularized_v3", "mesh_v4", "rails_v5"):
        with torch.no_grad():
            runner.alg.policy.std.fill_(.5)
    dump_yaml(str(out / "env.yaml"), cfg)
    dump_yaml(str(out / "agent.yaml"), agent)
    (out / "scope.json").write_text(json.dumps({"simulation_only":True,"seed":41,
        "initial_checkpoint":str(checkpoint),"iterations":args.iterations,"num_envs":args.num_envs,"profile":args.profile},indent=2))
    runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=True)
    env.close()
except Exception:
    traceback.print_exc()
    if 'env' in locals():
        env.close()
    raise
finally:
    app.close()
