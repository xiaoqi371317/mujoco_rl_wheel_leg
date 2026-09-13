"""Native mjlab manager-based wheel-leg task. No Gym/SB3 wrapper is involved."""
from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET
import mujoco
import torch
from mjlab.actuator import XmlActuatorCfg
from mjlab.entity import EntityCfg, EntityArticulationInfoCfg
from mjlab.envs import mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg, JointEffortActionCfg
from mjlab.managers import EventTermCfg, ObservationTermCfg, RewardTermCfg, TerminationTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlModelCfg, RslRlPpoAlgorithmCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactSensorCfg, ContactMatch
from mjlab.sim import SimulationCfg, MujocoCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.viewer import ViewerConfig
from mjlab.utils.nan_guard import NanGuardCfg

MODEL = Path(__file__).parent / "robot/xmls/training_scene.xml"
DRIVES = ("lf0_Joint", "l20_Joint", "rf0_Joint", "r20_Joint")
WHEELS = ("l_wheel_Joint", "r_wheel_Joint")
WEIGHTS = dict(alive=.2, posture=2., velocity=1.5, yaw=.5, support=.5,
               action_rate=-.03, effort=-.02, angular_rate=-.03,
               leg_deviation=-.1, lateral_velocity=-.2, failure=-5.)


def robot_spec():
    # mjlab adds the terrain. Keep the imported robot's inertias and contacts.
    root = ET.parse(MODEL).getroot()
    root.find("compiler").set("meshdir", str((MODEL.parents[3] / "infantry_V4/meshes").resolve()))
    world = root.find("worldbody")
    for child in list(world):
        if child.tag != "body":
            world.remove(child)
    for tag in ("option", "statistic", "visual", "keyframe"):
        for node in list(root.findall(tag)):
            root.remove(node)
    return mujoco.MjSpec.from_string(ET.tostring(root, encoding="unicode"))


def indices(env):
    if not hasattr(env, "_infantry_indices"):
        robot = env.scene["robot"]
        def ids(names, kind="joint"):
            fn = robot.find_joints if kind == "joint" else robot.find_sites
            return fn(names, preserve_order=True)[0]
        front = tuple(f"{s}_front_site{i}" for s in ("Left", "Right") for i in (1, 2))
        rear = tuple(f"{s}_rear_site{i}" for s in ("Left", "Right") for i in (1, 2))
        env._infantry_indices = dict(legs=ids(DRIVES),
                                    nonwheel=ids(tuple(n for n in robot.joint_names if n not in WHEELS)),
                                    front=ids(front, "site"), rear=ids(rear, "site"))
    return env._infantry_indices


def wheel_support(env):
    return (env.scene["wheel_contact"].data.found > 0).float()


def bad_contact(env):
    sensor = env.scene["nonwheel_contact"].data
    bad = (sensor.found > 0).any(dim=1)
    if sensor.dist_history is not None:
        bad = bad | (sensor.dist_history < -1e-5).flatten(1).any(dim=1)
    return bad


def constraint_error(env):
    d = env.scene["robot"].data
    idx = indices(env)
    return (d.site_pos_w[:, idx["front"]] - d.site_pos_w[:, idx["rear"]]).abs().flatten(1).amax(dim=1)


def bad_constraint(env):
    return constraint_error(env) > .005


def tilt_angle(env):
    return torch.acos((-env.scene["robot"].data.projected_gravity_b[:, 2]).clamp(-1., 1.))


def height(env):
    return env.scene["robot"].data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]


def bad_pose(env):
    return (tilt_angle(env) > .65) | (height(env) < .14)


def observations(env):
    d = env.scene["robot"].data
    idx = indices(env)["nonwheel"]
    obs = torch.cat((d.projected_gravity_b, d.root_link_ang_vel_b * .25, d.root_link_lin_vel_b,
                     d.joint_pos[:, idx] - d.default_joint_pos[:, idx], d.joint_vel * .05,
                     env.action_manager.action, ((height(env) - .24) * 5.).unsqueeze(1),
                     env.command_manager.get_command("twist"), wheel_support(env)), dim=1)
    # NaN termination records invalid states independently; never feed NaN to PPO.
    return torch.nan_to_num(obs, nan=0., posinf=0., neginf=0.)


def reward_feature(env, name):
    d = env.scene["robot"].data
    command = env.command_manager.get_command("twist")
    posture = torch.exp(-(tilt_angle(env) / .25).square() - ((height(env) - .235) / .05).square())
    if name == "alive":
        return (~env.termination_manager.terminated).float()
    if name == "posture":
        return posture
    if name == "velocity":
        return posture * torch.exp(-((d.root_link_lin_vel_b[:, 0] - command[:, 0]) / .35).square())
    if name == "yaw":
        return posture * torch.exp(-((d.root_link_ang_vel_b[:, 2] - command[:, 2]) / .5).square())
    if name == "support":
        return posture * wheel_support(env).amin(dim=1)
    if name == "action_rate":
        return (env.action_manager.action - env.action_manager.prev_action).square().mean(dim=1)
    if name == "effort":
        return (d.actuator_force / 30.).square().mean(dim=1)
    if name == "angular_rate":
        return d.root_link_ang_vel_b[:, :2].square().sum(dim=1)
    if name == "leg_deviation":
        idx = indices(env)["legs"]
        return ((d.joint_pos[:, idx] - d.default_joint_pos[:, idx]) / .18).square().mean(dim=1)
    if name == "lateral_velocity":
        return d.root_link_lin_vel_b[:, 1].square()
    if name == "failure":
        return env.termination_manager.terminated.float()
    raise KeyError(name)


def success(env):
    return ((env.episode_length_buf >= env.max_episode_length) & ~env.termination_manager.terminated).float()


def velocity_error(env):
    return (env.scene["robot"].data.root_link_lin_vel_b[:, 0] - env.command_manager.get_command("twist")[:, 0]).abs()


def make_cfg(num_envs=64, task="balance", play=False):
    if task not in ("balance", "velocity"):
        raise ValueError(task)
    # Reuse mjlab's velocity task structure; replace biped-specific terms with
    # wheel-leg terms and a closed-chain-preserving reset.
    cfg = make_velocity_env_cfg()
    source = mujoco.MjModel.from_xml_path(str(MODEL))
    home = {source.joint(j).name: float(source.key_qpos[0, source.jnt_qposadr[j]])
            for j in range(1, source.njnt)}
    robot = EntityCfg(spec_fn=robot_spec,
                      init_state=EntityCfg.InitialStateCfg(pos=(0., 0., .244), joint_pos=home),
                      articulation=EntityArticulationInfoCfg(actuators=(
                          XmlActuatorCfg(target_names_expr=DRIVES, command_field="position"),
                          XmlActuatorCfg(target_names_expr=WHEELS, command_field="effort"))))
    cfg.scene = SceneCfg(num_envs=num_envs, env_spacing=2.,
                         terrain=TerrainEntityCfg(terrain_type="plane"), entities={"robot": robot},
                         sensors=(ContactSensorCfg(name="wheel_contact",
                            primary=ContactMatch(mode="geom", pattern="[lr]_wheel_Link_collision", entity="robot"),
                            secondary=ContactMatch(mode="body", pattern="terrain"), fields=("found",), reduce="none"),
                            ContactSensorCfg(name="nonwheel_contact",
                            primary=ContactMatch(mode="body", pattern=".*", entity="robot", exclude=("[lr]_wheel_Link",)),
                            secondary=ContactMatch(mode="body", pattern="terrain"), fields=("found", "dist"),
                            reduce="mindist", history_length=10)))
    cfg.sim = SimulationCfg(nconmax=64, njmax=256, contact_sensor_maxmatch=128,
                            mujoco=MujocoCfg(timestep=.002, integrator="implicitfast", solver="newton",
                                             iterations=60, ls_iterations=30, tolerance=1e-8),
                            nan_guard=NanGuardCfg(enabled=True, buffer_size=2,
                                output_dir=str(Path("artifacts/infantry_mjlab/nan_dumps").resolve())))
    cfg.decimation, cfg.episode_length_s = 10, 10.
    cfg.scale_rewards_by_dt = True
    cfg.actions = {
        # Names select actuator TARGET JOINTS in mjlab 1.3, in natural model order.
        "legs": JointPositionActionCfg(entity_name="robot", actuator_names=DRIVES, scale=.18, use_default_offset=True),
        "wheels": JointEffortActionCfg(entity_name="robot", actuator_names=WHEELS,
                                       scale={"l_wheel_Joint": 8., "r_wheel_Joint": -8.})}
    cfg.commands = {"twist": UniformVelocityCommandCfg(entity_name="robot", resampling_time_range=(5., 8.),
        rel_standing_envs=1. if task == "balance" else .35, rel_heading_envs=0., rel_forward_envs=0.,
        heading_command=False, debug_vis=play,
        ranges=UniformVelocityCommandCfg.Ranges(lin_vel_x=(-.3, .3), lin_vel_y=(0., 0.), ang_vel_z=(-.4, .4)))}
    obs = ObservationGroupCfg(terms={"state": ObservationTermCfg(func=observations)},
                              concatenate_terms=True, enable_corruption=False)
    cfg.observations = {"actor": obs, "critic": deepcopy(obs)}
    cfg.events = {"reset": EventTermCfg(func=mdp.reset_scene_to_default, mode="reset"),
                  "tilt": EventTermCfg(func=mdp.reset_root_state_uniform, mode="reset",
                    params={"pose_range": {"pitch": (-.025, .025)},
                            "velocity_range": {"x": (-.025, .025), "pitch": (-.025, .025)}})}
    cfg.rewards = {name: RewardTermCfg(func=reward_feature, weight=weight, params={"name": name})
                   for name, weight in WEIGHTS.items()}
    cfg.terminations = {"time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
                         "bad_pose": TerminationTermCfg(func=bad_pose),
                         "nonwheel_contact": TerminationTermCfg(func=bad_contact),
                         "closed_chain": TerminationTermCfg(func=bad_constraint),
                         "nan": TerminationTermCfg(func=mdp.nan_detection)}
    cfg.curriculum = {}
    cfg.metrics = {"tilt_rad": MetricsTermCfg(func=tilt_angle),
                    "velocity_mae": MetricsTermCfg(func=velocity_error),
                    "closure_error_m": MetricsTermCfg(func=constraint_error),
                    "success": MetricsTermCfg(func=success, reduce="last")}
    cfg.viewer = ViewerConfig(entity_name="robot", body_name="base_Link_del", distance=1.5,
                               azimuth=135, elevation=-18, width=640, height=360)
    return cfg


def runner_cfg(iterations=500):
    return RslRlOnPolicyRunnerCfg(max_iterations=iterations, num_steps_per_env=24, save_interval=50,
        experiment_name="infantry_balance", logger="tensorboard", upload_model=False, clip_actions=1.,
        actor=RslRlModelCfg(hidden_dims=(128, 128), obs_normalization=True,
                            distribution_cfg={"class_name": "GaussianDistribution", "init_std": .35, "std_type": "scalar"}),
        critic=RslRlModelCfg(hidden_dims=(128, 128), obs_normalization=True),
        algorithm=RslRlPpoAlgorithmCfg(learning_rate=3e-4, entropy_coef=.003, num_learning_epochs=5))
