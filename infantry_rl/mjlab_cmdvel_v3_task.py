"""Command tracking v3: preserve v2 behavior and add mild mechanical regularization."""
import torch
from mjlab.managers import RewardTermCfg
from infantry_rl import mjlab_cmdvel_task as v2
from infantry_rl import mjlab_task as base


def mechanical_reward(env, name):
    data = env.scene["robot"].data
    if name == "vertical_velocity":
        return data.root_link_lin_vel_b[:, 2].square()
    if name == "joint_acceleration":
        # Normalize to a useful scale before averaging over the four driven leg joints.
        return (data.joint_acc[:, base.indices(env)["legs"]] / 100.0).square().mean(dim=1)
    if name == "action_smooth":
        manager = env.action_manager
        second_difference = manager.action - 2.0 * manager.prev_action + manager.prev_prev_action
        return second_difference.square().mean(dim=1)
    if name == "joint_limits":
        idx = base.indices(env)["legs"]
        position = data.joint_pos[:, idx]
        limits = data.soft_joint_pos_limits[:, idx]
        below = -(position - limits[..., 0]).clip(max=0.0)
        above = (position - limits[..., 1]).clip(min=0.0)
        return (below + above).sum(dim=1)
    raise KeyError(name)


def make_cfg(num_envs=64, task="cmd_vel_v3", play=False):
    cfg = v2.make_cfg(num_envs, "cmd_vel", play)
    for name, weight in (
        ("vertical_velocity", -0.2),
        ("joint_acceleration", -0.03),
        ("action_smooth", -0.02),
        ("joint_limits", -1.0),
    ):
        cfg.rewards[name] = RewardTermCfg(
            func=mechanical_reward, weight=weight, params={"name": name}
        )
    return cfg


runner_cfg = base.runner_cfg
