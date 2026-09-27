"""Replay the same checkpoint through upstream Isaac Lab managers for comparison."""
import argparse
import json
from pathlib import Path
import traceback

from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--out", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app
try:
    import torch
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.flat_env_cfg import UnitreeGo2FlatEnvCfg_PLAY
    root = Path(__file__).resolve().parents[2]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    cfg = UnitreeGo2FlatEnvCfg_PLAY()
    cfg.scene.num_envs = 1
    cfg.sim.device = args.device
    cfg.seed = 0
    cfg.scene.robot.spawn.usd_path = str(root / "assets/go2/go2_lab.usd")
    cfg.scene.terrain.visual_material = None
    cfg.scene.sky_light.spawn.texture_file = None
    cfg.events.add_base_mass = None
    cfg.events.reset_base.params["pose_range"] = {}
    cfg.commands.base_velocity.debug_vis = False
    cfg.commands.base_velocity.heading_command = False
    cfg.commands.base_velocity.resampling_time_range = (10000., 10000.)
    cfg.commands.base_velocity.rel_standing_envs = 0.
    cfg.episode_length_s = 1000.
    env = ManagerBasedRLEnv(cfg=cfg)
    actor = torch.nn.Sequential(torch.nn.Linear(48, 128), torch.nn.ELU(), torch.nn.Linear(128, 128),
                                torch.nn.ELU(), torch.nn.Linear(128, 128), torch.nn.ELU(), torch.nn.Linear(128, 12)).to(args.device)
    saved = torch.load(str(root / "assets/go2/checkpoint.pt"), map_location=args.device, weights_only=True)["model_state_dict"]
    actor.load_state_dict({k[6:]: v for k, v in saved.items() if k.startswith("actor.")})
    actor.eval()
    obs, _ = env.reset()
    robot = env.scene["robot"]
    print("JOINTS", robot.joint_names, "DEFAULT", robot.data.default_joint_pos.tolist(), flush=True)
    stream = (out / "states.jsonl").open("w")
    schedule = [("settle", 2., [0, 0, 0]), ("stand", 3., [0, 0, 0]),
                ("forward", 4., [.3, 0, 0]), ("stop_1", 2., [0, 0, 0]),
                ("backward", 4., [-.2, 0, 0]), ("left", 4., [0, .2, 0]),
                ("right", 4., [0, -.2, 0]), ("yaw", 4., [0, 0, .5]), ("stop_2", 3., [0, 0, 0])]
    for phase, duration, command in schedule:
        for i in range(round(duration/env.step_dt)):
            with torch.inference_mode():
                env.command_manager.get_command("base_velocity")[:] = torch.tensor(command, device=args.device)
                obs = env.observation_manager.compute()
                action = actor(obs["policy"])
                obs, _, terminated, truncated, _ = env.step(action)
                state = {"phase": phase, "t": env.common_step_counter*env.step_dt, "command": command,
                         "position": robot.data.root_pos_w[0].tolist(),
                         "body_velocity": robot.data.root_lin_vel_b[0].tolist(),
                         "body_angular_velocity": robot.data.root_ang_vel_b[0].tolist(),
                         "joint_position": robot.data.joint_pos[0].tolist(),
                         "joint_velocity": robot.data.joint_vel[0].tolist(),
                         "action": action[0].tolist(), "observation": obs["policy"][0].tolist(),
                         "contact_forces": env.scene["contact_forces"].data.net_forces_w[0].tolist(),
                         "terminated": terminated[0].item(), "truncated": truncated[0].item()}
                stream.write(json.dumps(state) + "\n")
        stream.flush()
        print("PHASE", phase, json.dumps(state), flush=True)
    (out / "bodies.json").write_text(json.dumps(robot.body_names))
    (out / "config.txt").write_text(str(cfg.to_dict()))
    stream.close()
    env.close()
except Exception:
    traceback.print_exc()
    raise
finally:
    app.close()
