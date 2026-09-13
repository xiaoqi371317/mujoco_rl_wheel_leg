"""Continue V5: retain measured pre-jump world-horizontal velocity throughout the jump."""
from dataclasses import dataclass
import torch
from mjlab.managers import RewardTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from infantry_rl import mjlab_motion_task as motion,mjlab_rough_task as rough,mjlab_task as base

def world_to_body(q,v):
    t=2*torch.cross(q[:,1:],v,dim=1)
    return v-q[:,:1]*t+torch.cross(q[:,1:],t,dim=1)

class JumpCarryCommand(motion.MotionCommand):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        self.carry_world=torch.zeros_like(self.drive)
        self.carry_valid=torch.zeros(self.num_envs,dtype=torch.bool,device=self.device)
        self.extra_jumps=not cfg.full_range
        self.auto_jump=False
    def _resample_command(self,env_ids):
        super()._resample_command(env_ids)
        ids=env_ids[self.command_counter[env_ids]==0]
        self.carry_world[ids]=0.;self.carry_valid[ids]=False
        if not self.cfg.full_range:
            # Keep normal/high/stop/spin examples; specialize 55% of flat starts to moving jumps.
            ids=ids[self._env._rough_flat[ids]&(torch.rand(len(ids),device=self.device)<.55)]
            maximum=.6+1.4*self.progress(300)
            self.drive[ids,0]=-torch.empty(len(ids),device=self.device).uniform_(.3,maximum)
            self.drive[ids,2]=0.;self.drive_height[ids]=.235
    def begin_jump(self,ids):
        ids=ids[(self.phase[ids]==0)&(base.wheel_support(self._env)[ids].sum(1)>0)]
        self.carry_world[ids]=self.robot.data.root_link_lin_vel_w[ids]
        self.carry_world[ids,2]=0.;self.carry_valid[ids]=True
        super().begin_jump(ids)
        self.jump_target[ids]=.06+.02*self.progress(300)
    def compute(self,dt):
        super().compute(dt)
        if self.extra_jumps:
            level=self._env.scene.terrain.terrain_levels
            terrain_ok=self._env._rough_flat|((level==0)&(motion.iteration(self._env)>200))
            vmax=.6+1.4*self.progress(300)
            err=(self.robot.data.root_link_lin_vel_b[:,0]-self.drive[:,0]).abs()
            eligible=(self.phase==0)&(self.age>1.8)&terrain_ok&(self.drive_height<.25)
            eligible&=(self.drive[:,2].abs()<.15)&(self.drive[:,0].abs()<=vmax)&(err<.20)
            # Requests happen after actual speed settles, not after braking to zero.
            self.begin_jump((eligible&(torch.rand(self.num_envs,device=self.device)<dt*.9)).nonzero().flatten())
        active=(self.phase>=2)&self.carry_valid
        # Explicit manual stop takes precedence; airborne horizontal speed cannot change instantly.
        if self.manual_source is not None:
            stopped=self.drive[:,:2].abs().sum(1)<.01
            self.carry_world[active&stopped]=0.
        target=world_to_body(self.robot.data.root_link_quat_w,self.carry_world)
        self.vel_command_b[active,:2]=target[active,:2]
        self.carry_valid[self.phase==0]=False

@dataclass(kw_only=True)
class JumpCarryCommandCfg(motion.MotionCommandCfg):
    def build(self,env):return JumpCarryCommand(self,env)

def errors(env):
    cmd=env.command_manager.get_term('twist');d=env.scene['robot'].data
    error=d.root_link_lin_vel_w[:,:2]-cmd.carry_world[:,:2]
    target=cmd.carry_world[:,:2];norm=target.norm(dim=1)
    projected=(d.root_link_lin_vel_w[:,:2]*target).sum(1)/norm.clamp_min(.1)
    loss=(norm-projected).clamp_min(0)
    return error,loss,norm

def reward(env,name):
    cmd=env.command_manager.get_term('twist');p=cmd.phase;d=env.scene['robot'].data
    active=(p>=2)&cmd.carry_valid;e,loss,speed=errors(env)
    quality=torch.exp(-(e/.25).square().sum(1));posture=torch.exp(-(base.tilt_angle(env)/.35).square())
    if name=='carry_quality':return active*quality
    if name=='carry_error':return active*(e/.30).square().sum(1).clamp_max(16.)
    if name=='braking':return ((p==2)|(p==3)|(p==5))*(loss/.25).square().clamp_max(16.)
    if name=='takeoff':return (p==3)*posture*torch.exp(-((d.root_link_lin_vel_w[:,2]-1.15)/.55).square())*(.3+.7*quality)
    if name=='flight':
        clearance=motion.wheel_clearance(env)
        air=(base.wheel_support(env).sum(1)==0)&(clearance>.005)
        return (p==4)*air*posture*(clearance/cmd.jump_target).clamp(0,1)*(.3+.7*quality)
    if name=='landing':
        return (p==5)*cmd.air_seen*posture*base.wheel_support(env).amin(1)*torch.exp(-(d.root_link_lin_vel_w[:,2]/.4).square())*quality
    if name=='teacher':
        teacher=getattr(env,'_motion_teacher',None)
        if teacher is None:return torch.zeros_like(speed)
        with torch.no_grad():target=teacher(rough.observations(env)).clamp(-1,1)
        return (p==0)*(env.action_manager.action-target).square().mean(1)
    raise KeyError(name)

def metric(env,name):
    cmd=env.command_manager.get_term('twist');e,loss,speed=errors(env);active=(cmd.phase>=2)&cmd.carry_valid
    if name=='carry_speed_error':return active*e.norm(dim=1)
    if name=='carry_speed_loss':return active*loss
    if name=='jump_active':return active.float()
    return motion.metric(env,name)

def make_cfg(num_envs=1024,play=False,seed=43):
    cfg=motion.make_cfg(num_envs,play=play,seed=seed)
    for sub,ratio in zip(cfg.scene.terrain.terrain_generator.sub_terrains.values(),(.8,.1,.1)):sub.proportion=ratio
    cfg.commands['twist']=JumpCarryCommandCfg(**vars(cfg.commands['twist']))
    for name,w in [('carry_quality',5.),('carry_error',-1.),('braking',-.8),('takeoff',10.),('flight',18.),('landing',10.),('teacher',-1.)]:
        cfg.rewards[name]=RewardTermCfg(func=reward,weight=w,params={'name':name})
    for name in ['carry_speed_error','carry_speed_loss','jump_active']:
        cfg.metrics[name]=MetricsTermCfg(func=metric,params={'name':name})
    return cfg

local_height=motion.local_height
wheel_clearance=motion.wheel_clearance
