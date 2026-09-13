"""Height-conditioned wheel-leg PPO with turn release and controlled spin."""
from dataclasses import dataclass
import torch
from mjlab.managers import ObservationTermCfg, RewardTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand, UniformVelocityCommandCfg
from infantry_rl import mjlab_task as base
from infantry_rl import mjlab_cmdvel_v3_task as v3

LOW_HEIGHT, HIGH_HEIGHT, SPIN_SPEED = .235, .28, 3.0

class HeightVelocityCommand(UniformVelocityCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.height_target = torch.full((self.num_envs,), LOW_HEIGHT, device=self.device)
        self.height_command = self.height_target.clone()
        self.manual_source = None

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        n = len(env_ids)
        progress = 1. if self.cfg.full_range else min(1., self._env.common_step_counter / 14400.)
        bucket = torch.rand(n, device=self.device)
        # 30% stop, 30% straight, 20% arc, 20% spin; every mode can be high.
        self.is_standing_env[env_ids] = bucket < .3
        self.vel_command_b[env_ids[(bucket >= .3) & (bucket < .6)], 2] = 0.
        spin = env_ids[bucket >= .8]
        self.vel_command_b[spin, 0] = 0.
        sign = torch.where(torch.rand(len(spin), device=self.device) < .5, -1., 1.)
        self.vel_command_b[spin, 2] = sign * (1.2 + 1.8 * progress)
        # Explicit release transitions preserve forward motion while dropping yaw.
        release = env_ids[(self.command_counter[env_ids] % 2 == 1) & (torch.rand(n,device=self.device)<.5)]
        self.vel_command_b[release, 2] = 0.
        high = torch.rand(n, device=self.device) < .45
        self.height_target[env_ids] = LOW_HEIGHT + high.float() * progress * (HIGH_HEIGHT-LOW_HEIGHT)
        fresh = env_ids[self.command_counter[env_ids] == 0]
        self.height_command[fresh] = LOW_HEIGHT

    def compute(self, dt):
        super().compute(dt)
        if self.manual_source is not None:
            vx, wz, h = self.manual_source()
            self.vel_command_b[:] = torch.tensor([vx, 0., wz], device=self.device)
            self.height_target[:] = h
        # 3 cm/s avoids an instantaneous leg extension impulse.
        self.height_command += (self.height_target-self.height_command).clamp(-.03*dt,.03*dt)

@dataclass(kw_only=True)
class HeightVelocityCommandCfg(UniformVelocityCommandCfg):
    full_range: bool = False
    def build(self, env):
        return HeightVelocityCommand(self, env)

def height_error(env):
    return (base.height(env)-env.command_manager.get_term('twist').height_command).abs()

def observations(env):
    h = env.command_manager.get_term('twist').height_command
    return torch.cat((base.observations(env), ((h-LOW_HEIGHT)*20.).unsqueeze(1)),1)

def reward(env, name):
    d = env.scene['robot'].data
    cmd = env.command_manager.get_command('twist')
    ex = (d.root_link_lin_vel_b[:,0]-cmd[:,0]).abs()
    ez = (d.root_link_ang_vel_b[:,2]-cmd[:,2]).abs()
    posture = torch.exp(-(base.tilt_angle(env)/.25).square())
    height_quality = torch.exp(-(height_error(env)/.025).square())
    if name == 'posture': return posture
    if name == 'height': return posture*height_quality
    if name == 'height_error': return height_error(env)
    if name == 'joint_tracking': return posture*torch.exp(-(ex/.25).square()-(ez/.5).square())
    if name == 'support': return posture*base.wheel_support(env).amin(1)
    if name == 'braking': return (cmd[:,2].abs()<.01).float()*d.root_link_ang_vel_b[:,2].square()
    raise KeyError(name)

def tracking_good(env):
    d=env.scene['robot'].data
    cmd=env.command_manager.get_command('twist')
    return ((base.velocity_error(env)<.1) & ((d.root_link_ang_vel_b[:,2]-cmd[:,2]).abs()<.2)
            & (height_error(env)<.015) & (base.tilt_angle(env)<.2)
            & ~env.termination_manager.terminated).float()

def make_cfg(num_envs=64, task='cmd_vel_v4', play=False):
    cfg = v3.make_cfg(num_envs, play=play)
    cfg.commands['twist'] = HeightVelocityCommandCfg(entity_name='robot',resampling_time_range=(2.,4.),
        rel_heading_envs=0.,heading_command=False,debug_vis=play,full_range=play,
        ranges=UniformVelocityCommandCfg.Ranges(lin_vel_x=(-.3,.5),lin_vel_y=(0.,0.),ang_vel_z=(-1.5,1.5)))
    cfg.actions['legs'].scale = .35
    for group in cfg.observations.values():
        group.terms['state'] = ObservationTermCfg(func=observations)
    cfg.rewards.pop('leg_deviation')
    for name,weight in [('posture',2.),('height',3.),('height_error',-8.),('joint_tracking',4.),('support',.5),('braking',-.3)]:
        cfg.rewards[name] = RewardTermCfg(func=reward,weight=weight,params={'name':name})
    cfg.metrics['height_mae_m'] = MetricsTermCfg(func=height_error)
    cfg.metrics['tracking_fraction'] = MetricsTermCfg(func=tracking_good)
    return cfg

runner_cfg = base.runner_cfg
