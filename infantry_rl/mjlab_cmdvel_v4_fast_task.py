"""V4 fast: user-forward = model -x, 1.3 m/s, fast static height changes."""
from dataclasses import dataclass
import torch
from mjlab.managers import RewardTermCfg
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand
from infantry_rl import mjlab_task as base
from infantry_rl import mjlab_cmdvel_v4_task as v4

FORWARD_SPEED, REVERSE_SPEED = 1.3, .5
HEIGHT_RATE = .12  # m/s: 45 mm target transition in 0.375 s instead of 1.5 s.
CURRICULUM_STEPS = 28800  # 1200 PPO iterations, independent of warm-start age.
LOW_HEIGHT, HIGH_HEIGHT, SPIN_SPEED = v4.LOW_HEIGHT, v4.HIGH_HEIGHT, v4.SPIN_SPEED
height_error = v4.height_error


class FastCommand(v4.HeightVelocityCommand):
    def _resample_command(self, env_ids):
        n=len(env_ids)
        previous=self.vel_command_b[env_ids].clone()
        progress=1. if self.cfg.full_range else min(1.,self._env.common_step_counter/CURRICULUM_STEPS)
        maximum=.3+(FORWARD_SPEED-.3)*progress
        bucket=torch.rand(n,device=self.device)
        self.vel_command_b[env_ids]=0.
        self.is_standing_env[env_ids]=bucket<.3
        self.height_target[env_ids]=LOW_HEIGHT
        self.height_target[env_ids[(bucket>=.15)&(bucket<.3)]]=HIGH_HEIGHT
        forward=env_ids[(bucket>=.3)&(bucket<.7)]
        # Half the forward samples are exactly the current maximum, including 1.3.
        r=torch.rand(len(forward),device=self.device)
        self.vel_command_b[forward,0]=-torch.where(r<.5,maximum,.1+(maximum-.1)*(r-.5)*2.)
        reverse=env_ids[(bucket>=.7)&(bucket<.8)]
        self.vel_command_b[reverse,0]=torch.empty(len(reverse),device=self.device).uniform_(.1,REVERSE_SPEED)
        arc=env_ids[(bucket>=.8)&(bucket<.9)]
        self.vel_command_b[arc,0]=-torch.empty(len(arc),device=self.device).uniform_(.15,min(.6,maximum))
        self.vel_command_b[arc,2]=torch.empty(len(arc),device=self.device).uniform_(-1.2,1.2)
        spin=env_ids[bucket>=.9]
        self.vel_command_b[spin,2]=torch.where(torch.rand(len(spin),device=self.device)<.5,-SPIN_SPEED,SPIN_SPEED)
        # Dedicated high-speed braking examples; do not require high-stance driving.
        release=(previous[:,0].abs()>.15)&(self.command_counter[env_ids]>0)&(torch.rand(n,device=self.device)<.25)
        stop=env_ids[release]
        self.vel_command_b[stop]=0.;self.height_target[stop]=LOW_HEIGHT
        self.is_standing_env[stop]=True
        fresh=env_ids[self.command_counter[env_ids]==0]
        self.height_command[fresh]=LOW_HEIGHT

    def compute(self, dt):
        UniformVelocityCommand.compute(self,dt)
        if self.manual_source is not None:
            vx,wz,h=self.manual_source()
            self.vel_command_b[:]=torch.tensor([vx,0.,wz],device=self.device)
            self.height_target[:]=h if abs(vx)<.01 and abs(wz)<.01 else LOW_HEIGHT
        self.height_command+=(self.height_target-self.height_command).clamp(-HEIGHT_RATE*dt,HEIGHT_RATE*dt)


@dataclass(kw_only=True)
class FastCommandCfg(v4.HeightVelocityCommandCfg):
    def build(self,env):return FastCommand(self,env)


def fast_reward(env,name):
    d=env.scene['robot'].data
    cmd=env.command_manager.get_command('twist')
    if name=='linear_braking':
        return (cmd[:,0].abs()<.01).float()*d.root_link_lin_vel_b[:,0].square()
    raise KeyError(name)


def make_cfg(num_envs=64,task='cmd_vel_v4_fast',play=False):
    cfg=v4.make_cfg(num_envs,play=play)
    cfg.commands['twist']=FastCommandCfg(entity_name='robot',resampling_time_range=(2.,4.),
        rel_heading_envs=0.,heading_command=False,debug_vis=play,full_range=play,
        ranges=v4.UniformVelocityCommandCfg.Ranges(lin_vel_x=(-FORWARD_SPEED,REVERSE_SPEED),lin_vel_y=(0.,0.),ang_vel_z=(-1.5,1.5)))
    cfg.rewards['joint_tracking'].weight=6.
    cfg.rewards['forward_error'].weight=-4.
    cfg.rewards['height'].weight=5.
    cfg.rewards['height_error'].weight=-15.
    cfg.rewards['linear_braking']=RewardTermCfg(func=fast_reward,weight=-.5,params={'name':'linear_braking'})
    return cfg


runner_cfg=base.runner_cfg
