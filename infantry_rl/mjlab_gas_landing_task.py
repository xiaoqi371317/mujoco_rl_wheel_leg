"""Retract after takeoff; prepare landing using predicted extended-leg contact."""
from dataclasses import dataclass
import torch
from mjlab.managers import RewardTermCfg
from infantry_rl import mjlab_gas_task as gas,mjlab_v6_refine_task as refine
from infantry_rl import mjlab_jumpcarry_task as carry,mjlab_task as base

class LandingCommand(refine.RefineCommand):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        self.landing_ready=torch.zeros(self.num_envs,dtype=torch.bool,device=self.device)
    def _resample_command(self,ids):
        super()._resample_command(ids)
        self.landing_ready[ids[self.command_counter[ids]==0]]=False
    def begin_jump(self,ids):
        eligible=ids[self.phase[ids]==0]
        super().begin_jump(ids)
        self.landing_ready[eligible]=False
    def compute(self,dt):
        previous=self.reference_height.clone()
        # Keep velocity carry and phase detection, bypass the old fixed tuck timer.
        carry.JumpCarryCommand.compute(self,dt)
        flight=(self.phase==4)&self.air_seen
        vz=self.robot.data.root_link_lin_vel_w[:,2]
        distance=(carry.local_height(self._env)-.25).clamp_min(0.)
        ttc=(vz+torch.sqrt(vz.square()+2*9.81*distance))/9.81
        self.landing_ready|=flight&(self.age>.06)&(ttc<.14)
        self.landing_ready[self.phase==0]=False
        target=torch.full_like(self.age,.18-.02*self.progress(200))
        target=torch.where(self.landing_ready,.26,target)
        # A bounded reference rate avoids an instantaneous .16 -> .24 command.
        reference=previous+(target-previous).clamp(-1.0*dt,1.2*dt)
        self.reference_height=torch.where(flight,reference,self.reference_height)
        self.height_command[flight]=self.reference_height[flight]

@dataclass(kw_only=True)
class LandingCommandCfg(refine.RefineCommandCfg):
    def build(self,env):return LandingCommand(self,env)

def reward(env,name):
    cmd=env.command_manager.get_term('twist');d=env.scene['robot'].data
    if name=='air_retract':return refine.reward(env,'air_retract')*~cmd.landing_ready
    if name=='landing_prepare':
        mask=(cmd.phase==4)&cmd.landing_ready
        pair,dx,_=gas.stance_errors(env)
        return mask*torch.exp(-(base.tilt_angle(env)/.25).square()-(pair/.15).square().mean(1)-(dx/.04).square())
    if name=='teacher':
        teacher=getattr(env,'_motion_teacher',None)
        if teacher is None:return torch.zeros_like(cmd.age)
        with torch.no_grad():target=teacher(carry.rough.observations(env)).clamp(-1,1)
        return (cmd.phase==0)*(env.action_manager.action-target).square().mean(1)
    raise KeyError(name)

def make_cfg(num_envs=1024,play=False,seed=43,stage='tuck'):
    cfg=gas.make_cfg(num_envs,play=play,seed=seed,stage='tuck')
    cfg.commands['twist']=LandingCommandCfg(**vars(cfg.commands['twist']))
    for name,w in [('air_retract',10.),('landing_prepare',8.),('teacher',-4.)]:
        cfg.rewards[name]=RewardTermCfg(func=reward,weight=w,params={'name':name})
    cfg.rewards['landing'].weight=24.
    cfg.rewards['carry_error'].weight=-2.
    return cfg

local_height=gas.local_height
wheel_clearance=gas.wheel_clearance
stance_errors=gas.stance_errors
