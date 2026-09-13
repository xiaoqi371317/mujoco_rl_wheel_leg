"""Separate quiet stance refinement from airborne leg retraction."""
from dataclasses import dataclass
import torch
from mjlab.managers import RewardTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from infantry_rl import mjlab_jumpcarry_task as carry,mjlab_task as base
from infantry_rl.mjlab_v6_jump_task import stance_errors
from infantry_rl.v6_terrain import terrain_cfg

class RefineCommand(carry.JumpCarryCommand):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        self.auto_height=False
        self.extra_jumps=not cfg.full_range and cfg.stage=='tuck'
    def _resample_command(self,env_ids):
        super()._resample_command(env_ids)
        ids=env_ids[(self.command_counter[env_ids]==0)|(self.phase[env_ids]==0)]
        probability=.8 if self.cfg.stage=='stance' else .5
        stop=ids[self._env._rough_flat[ids]&(torch.rand(len(ids),device=self.device)<probability)]
        self.drive[stop]=0.
        self.drive_height[stop]=torch.where(torch.rand(len(stop),device=self.device)<.3,.28,.235)
    def begin_jump(self,ids):
        # Stance rehearsal must not be interrupted by automatic jumps.
        if not self.cfg.full_range:
            if self.cfg.stage=='stance':return
            ids=ids[self.drive[ids,0].abs()>.2]
        super().begin_jump(ids)
    def compute(self,dt):
        super().compute(dt)
        if self.cfg.stage=='tuck':
            flight=(self.phase==4)&self.air_seen
            clearance=carry.wheel_clearance(self._env)
            descending=self.robot.data.root_link_lin_vel_w[:,2]<0
            extend=descending&((clearance<.06)|(self.age>.18))
            tuck=.24-.08*(self.age/.06).clamp(0,1)
            self.reference_height=torch.where(flight&~extend,tuck,self.reference_height)
            self.height_command[flight]=self.reference_height[flight]

@dataclass(kw_only=True)
class RefineCommandCfg(carry.JumpCarryCommandCfg):
    stage:str='stance'
    def build(self,env):return RefineCommand(self,env)

def mask(env):
    cmd=env.command_manager.get_term('twist')
    return (cmd.phase==0)&(cmd.drive.abs().sum(1)<.02)&env._rough_flat

def reward(env,name):
    cmd=env.command_manager.get_term('twist');d=env.scene['robot'].data;quiet=mask(env)
    pair,dx,vel=stance_errors(env)
    if name=='stance_pair':return quiet*(pair/.12).square().mean(1).clamp_max(25)
    if name=='stance_align':return quiet*(dx/.03).square().clamp_max(25)
    if name=='stand_yaw_stop':return quiet*(d.root_link_ang_vel_b[:,2]/.1).square().clamp_max(25)
    if name=='stand_translation':return quiet*(d.root_link_lin_vel_b[:,:2]/.08).square().sum(1).clamp_max(25)
    if name=='stand_quality':
        return quiet*torch.exp(-(pair/.10).square().mean(1)-(dx/.025).square()-(d.root_link_ang_vel_b[:,2]/.08).square())
    if name=='teacher':
        teacher=getattr(env,'_motion_teacher',None)
        if teacher is None:return torch.zeros_like(cmd.age)
        with torch.no_grad():target=teacher(carry.rough.observations(env)).clamp(-1,1)
        return (cmd.phase==0)*~quiet*(env.action_manager.action-target).square().mean(1)
    if name=='air_retract':
        wheel=d.body_link_pos_w[:,env._v6_wheels]
        reach=d.root_link_pos_w[:,2]-wheel[:,:,2].amin(1)
        air=(base.wheel_support(env).sum(1)==0)&(carry.wheel_clearance(env)>.005)
        return (cmd.phase==4)*air*torch.exp(-((reach-.11)/.05).square())*torch.exp(-(base.tilt_angle(env)/.3).square())
    raise KeyError(name)

def metric(env,name):
    if name=='stand_samples':return mask(env).float()
    pair,dx,_=stance_errors(env)
    if name=='stand_pair_masked':return mask(env)*pair.abs().mean(1)
    if name=='stand_yaw_masked':return mask(env)*env.scene['robot'].data.root_link_ang_vel_b[:,2].abs()
    raise KeyError(name)

def make_cfg(num_envs=1024,play=False,seed=43,stage='stance'):
    cfg=carry.make_cfg(num_envs,play=play,seed=seed)
    cfg.scene.terrain=terrain_cfg(seed)
    for sub,ratio in zip(cfg.scene.terrain.terrain_generator.sub_terrains.values(),(.9,.05,.05)):sub.proportion=ratio
    cfg.commands['twist']=RefineCommandCfg(**vars(cfg.commands['twist']),stage=stage)
    cfg.commands['twist'].resampling_time_range=(12.,20.)
    cfg.episode_length_s=24.
    for name,w in [('stance_pair',-4.),('stance_align',-2.),('stand_yaw_stop',-2.),('stand_translation',-2.),('stand_quality',6.),('teacher',-2.)]:
        cfg.rewards[name]=RewardTermCfg(func=reward,weight=w,params={'name':name})
    # Distinct key: retain the jump horizontal-braking reward.
    if stage=='tuck':
        cfg.rewards['air_retract']=RewardTermCfg(func=reward,weight=14.,params={'name':'air_retract'})
        cfg.rewards['takeoff'].weight=8.
        cfg.rewards['flight'].weight=12.
    for name in ['stand_samples','stand_pair_masked','stand_yaw_masked']:
        cfg.metrics[name]=MetricsTermCfg(func=metric,params={'name':name})
    return cfg

local_height=carry.local_height
wheel_clearance=carry.wheel_clearance
