"""Stronger terrain continuation, keeping the trained 56-dimensional policy."""
import torch
from mjlab.managers import RewardTermCfg
from infantry_rl import mjlab_rough_task as rough
from infantry_rl import mjlab_task as base
from infantry_rl.rugged_terrain import terrain_cfg

local_height=rough.local_height
metric=rough.metric


def height_reward(env,name):
    cmd=env.command_manager.get_term('twist')
    error=(local_height(env)-cmd.height_command).abs()
    idle=cmd.phase==0
    if name=='height':
        width=torch.where(env._rough_flat,.025,.04)
        return idle*torch.exp(-(base.tilt_angle(env)/.25).square()-(error/width).square())
    # Permit small passive compression over sharper bumps, without reducing flat tracking.
    return idle*torch.where(env._rough_flat,error,(error-.008).clamp_min(0))


def make_cfg(num_envs=1024,play=False,seed=43,flat_only=False):
    cfg=rough.make_cfg(num_envs,play=play,seed=seed,flat_only=flat_only)
    if not flat_only:cfg.scene.terrain=terrain_cfg(seed)
    for name,weight in [('height',5.),('height_error',-15.)]:
        cfg.rewards[name]=RewardTermCfg(func=height_reward,weight=weight,params={'name':name})
    return cfg
