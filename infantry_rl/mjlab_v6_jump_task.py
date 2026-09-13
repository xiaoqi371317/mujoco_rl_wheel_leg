"""V6: higher real jumps and quiet mirror-symmetric flat-ground stance."""
from dataclasses import dataclass
import torch
from mjlab.managers import RewardTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from infantry_rl import mjlab_jumpcarry_task as carry,mjlab_motion_task as motion,mjlab_task as base

class V6Command(carry.JumpCarryCommand):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        self.stance_hold=torch.zeros(self.num_envs,dtype=torch.bool,device=self.device)
    def _resample_command(self,env_ids):
        super()._resample_command(env_ids)
        ids=env_ids[(self.command_counter[env_ids]==0)|(self.phase[env_ids]==0)]
        self.stance_hold[ids]=False
        ids=ids[self._env._rough_flat[ids]&(torch.rand(len(ids),device=self.device)<.25)]
        self.stance_hold[ids]=True
        self.drive[ids]=0.
        self.drive_height[ids]=torch.where(torch.rand(len(ids),device=self.device)<.25,.28,.235)
    def begin_jump(self,ids):
        if not self.cfg.full_range:ids=ids[~self.stance_hold[ids]]
        super().begin_jump(ids)
        ids=ids[self.phase[ids]==2]
        self.jump_target[ids]=.10+.05*self.progress(300)

@dataclass(kw_only=True)
class V6CommandCfg(carry.JumpCarryCommandCfg):
    def build(self,env):return V6Command(self,env)

def stance_mask(env):
    cmd=env.command_manager.get_term('twist')
    return (cmd.phase==0)&(cmd.drive.abs().sum(1)<.02)&(cmd.age>1.)&env._rough_flat

def stance_errors(env):
    robot=env.scene['robot'];d=robot.data
    if not hasattr(env,'_v6_pair_ids'):
        env._v6_pair_ids=[robot.find_joints(names,preserve_order=True)[0] for names in
            [('lf0_Joint','l20_Joint'),('rf0_Joint','r20_Joint')]]
        env._v6_wheels=robot.find_bodies(('l_wheel_Link','r_wheel_Link'),preserve_order=True)[0]
    left,right=env._v6_pair_ids
    # Mirrored joint axes have opposite signs in the fix XML.
    pair=d.joint_pos[:,left]+d.joint_pos[:,right]
    wheel=d.body_link_pos_w[:,env._v6_wheels]
    delta=carry.world_to_body(d.root_link_quat_w,wheel[:,0]-wheel[:,1])
    return pair,delta[:,0],d.joint_vel[:,left+right]

def reward(env,name):
    cmd=env.command_manager.get_term('twist');d=env.scene['robot'].data
    if name=='takeoff':
        e,_,_=carry.errors(env);quality=torch.exp(-(e/.25).square().sum(1))
        # Ballistic scale is a curriculum target, not a guarantee of achieved clearance.
        target=torch.sqrt(2*9.81*cmd.jump_target)+.10
        return (cmd.phase==3)*torch.exp(-(base.tilt_angle(env)/.35).square())*torch.exp(-((d.root_link_lin_vel_w[:,2]-target)/.50).square())*(.3+.7*quality)
    if name=='teacher':
        teacher=getattr(env,'_motion_teacher',None)
        if teacher is None:return torch.zeros_like(cmd.age)
        with torch.no_grad():target=teacher(carry.rough.observations(env)).clamp(-1,1)
        return (cmd.phase==0)*~stance_mask(env)*(env.action_manager.action-target).square().mean(1)
    pair,dx,vel=stance_errors(env);mask=stance_mask(env)
    if name=='stance_mirror':return mask*(pair/.12).square().mean(1).clamp_max(16.)
    if name=='stance_wheel_alignment':return mask*(dx/.03).square().clamp_max(16.)
    if name=='stance_quiet':return mask*(vel/2.).square().mean(1).clamp_max(16.)
    raise KeyError(name)

def metric(env,name):
    pair,dx,vel=stance_errors(env);mask=stance_mask(env)
    if name=='stance_pair_rad':return mask*pair.abs().mean(1)
    if name=='stance_wheel_dx_m':return mask*dx.abs()
    if name=='stance_samples':return mask.float()
    if name=='jump_target_m':return env.command_manager.get_term('twist').jump_target
    return carry.metric(env,name)

def make_cfg(num_envs=1024,play=False,seed=43):
    cfg=carry.make_cfg(num_envs,play=play,seed=seed)
    from infantry_rl.v6_terrain import terrain_cfg
    cfg.scene.terrain=terrain_cfg(seed)
    for sub,ratio in zip(cfg.scene.terrain.terrain_generator.sub_terrains.values(),(.8,.1,.1)):sub.proportion=ratio
    cfg.commands['twist']=V6CommandCfg(**vars(cfg.commands['twist']))
    for name,w in [('takeoff',16.),('stance_mirror',-2.),('stance_wheel_alignment',-1.),('stance_quiet',-.15),('teacher',-1.)]:
        cfg.rewards[name]=RewardTermCfg(func=reward,weight=w,params={'name':name})
    cfg.rewards['flight'].weight=28.
    cfg.rewards['landing'].weight=14.
    for name in ['stance_pair_rad','stance_wheel_dx_m','stance_samples','jump_target_m']:
        cfg.metrics[name]=MetricsTermCfg(func=metric,params={'name':name})
    return cfg

local_height=carry.local_height
wheel_clearance=carry.wheel_clearance
