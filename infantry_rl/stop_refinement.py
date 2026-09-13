"""Small continuation-only reward changes for low-stance stop oscillation."""
from dataclasses import dataclass
import hashlib
from pathlib import Path
import torch
from mjlab.managers import RewardTermCfg
from infantry_rl import mjlab_cmdvel_v4_fast_task as fast
from infantry_rl import mjlab_task as base

class StopCommand(fast.FastCommand):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        self.stop_age=torch.zeros(self.num_envs,device=self.device)
    def _resample_command(self,ids):
        super()._resample_command(ids)
        # More long stop windows, retaining full-speed and spin regression samples.
        quiet=ids[torch.rand(len(ids),device=self.device)<.25]
        self.vel_command_b[quiet]=0.
        self.height_target[quiet]=fast.LOW_HEIGHT
        self.is_standing_env[quiet]=True
        self.time_left[quiet]=torch.empty(len(quiet),device=self.device).uniform_(5.,8.)
        self.stop_age[ids]=0.
    def compute(self,dt):
        super().compute(dt)
        self.stop_age+=dt

@dataclass(kw_only=True)
class StopCommandCfg(fast.FastCommandCfg):
    def build(self,env):return StopCommand(self,env)

def reward(env,name):
    command=env.command_manager.get_term('twist')
    c=command.command
    active=((c[:,0].abs()<.01)&(c[:,2].abs()<.01)&(command.height_target<.25)
            &(command.stop_age>1.)&(env.episode_length_buf>50)).float()
    d=env.scene['robot'].data
    if name=='quiet_velocity':return active*(d.root_link_lin_vel_b[:,0]/.05).square()
    if name=='quiet_action_rate':
        a=env.action_manager
        return active*(a.action-a.prev_action).square().mean(1)
    if name=='quiet_tracking':
        return active*torch.exp(-(d.root_link_lin_vel_b[:,0]/.025).square()
            -(d.root_link_ang_vel_b[:,2]/.06).square()-(base.tilt_angle(env)/.2).square())
    raise KeyError(name)

def apply(cfg):
    # Preserve nested range/viz dataclass instances instead of asdict conversion.
    old=cfg.commands['twist']
    cfg.commands['twist']=StopCommandCfg(**vars(old))
    cfg.commands['twist'].full_range=True
    for name,weight in [('quiet_velocity',-.5),('quiet_action_rate',-.15),('quiet_tracking',1.5)]:
        cfg.rewards[name]=RewardTermCfg(func=reward,weight=weight,params={'name':name})
    return cfg

def apply_from_metadata(cfg,meta):
    if meta.get('refinement')=='stop':
        if hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!=meta['stop_refinement_sha256']:
            raise ValueError('Stop refinement code changed since training')
        return apply(cfg)
    return cfg
