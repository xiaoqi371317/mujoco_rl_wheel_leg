"""Flat-ground command tracking v2; keep v1 immutable for checkpoint replay."""
from dataclasses import dataclass
import torch
from mjlab.managers import RewardTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand, UniformVelocityCommandCfg
from infantry_rl import mjlab_task as base


class MixedCommand(UniformVelocityCommand):
    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        # Explicit coverage: 35% stop, 30% straight, 20% arc, 15% spin.
        bucket = torch.rand(len(env_ids), device=self.device)
        self.is_standing_env[env_ids] = bucket < .35
        straight = env_ids[(bucket >= .35) & (bucket < .65)]
        spin = env_ids[bucket >= .85]
        self.vel_command_b[straight, 2] = 0.
        self.vel_command_b[spin, 0] = 0.


@dataclass(kw_only=True)
class MixedCommandCfg(UniformVelocityCommandCfg):
    def build(self, env):
        return MixedCommand(self, env)


def yaw_error(env):
    return (env.scene['robot'].data.root_link_ang_vel_b[:, 2] - env.command_manager.get_command('twist')[:, 2]).abs()


def tracking_good(env):
    return ((base.velocity_error(env) < .10) & (yaw_error(env) < .20)
            & (base.tilt_angle(env) < .20) & ~env.termination_manager.terminated).float()


def tracking_reward(env, name):
    vx = base.velocity_error(env)
    wz = yaw_error(env)
    if name == 'joint_tracking':
        return base.reward_feature(env, 'posture') * torch.exp(-(vx / .25).square() - (wz / .5).square())
    # Linear errors retain a gradient in the reward even during rapid unwanted spin.
    if name == 'yaw_error':
        return wz
    if name == 'forward_error':
        return vx
    raise KeyError(name)


def make_cfg(num_envs=64, task='cmd_vel', play=False):
    cfg = base.make_cfg(num_envs, 'velocity', play)
    cfg.commands = {'twist': MixedCommandCfg(entity_name='robot',
        resampling_time_range=(3., 5.), rel_standing_envs=.35,
        rel_heading_envs=0., rel_forward_envs=0., heading_command=False, debug_vis=play,
        ranges=UniformVelocityCommandCfg.Ranges(lin_vel_x=(-.3, .5),
            lin_vel_y=(0., 0.), ang_vel_z=(-1.5, 1.5)))}
    cfg.rewards.pop('velocity')
    cfg.rewards.pop('yaw')
    cfg.rewards['alive'].weight = 1.
    for name, weight in [('joint_tracking', 4.), ('yaw_error', -1.5), ('forward_error', -2.)]:
        cfg.rewards[name] = RewardTermCfg(func=tracking_reward, weight=weight, params={'name': name})
    cfg.metrics.pop('success')
    cfg.metrics['survived_episode'] = MetricsTermCfg(func=base.success, reduce='last')
    cfg.metrics['yaw_mae_rad_s'] = MetricsTermCfg(func=yaw_error)
    cfg.metrics['tracking_fraction'] = MetricsTermCfg(func=tracking_good)
    return cfg


runner_cfg = base.runner_cfg
