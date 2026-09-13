"""V5: closed-chain crouch recovery and phase-conditioned, torque-limited jumping."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import torch
from mjlab.envs import mdp
from mjlab.envs.mdp.actions.actions import JointPositionAction, JointPositionActionCfg
from mjlab.managers import EventTermCfg, ObservationTermCfg, RewardTermCfg, TerminationTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from infantry_rl import mjlab_task as base
from infantry_rl import mjlab_cmdvel_v4_task as v4
from infantry_rl import mjlab_cmdvel_v4_fast_task as fast

POSE_FILE=Path(__file__).with_name('v5_poses.json')
OBS_DIM=56
# 0 drive/stand, 1 rise, 2 crouch, 3 push, 4 flight, 5 landing/settle.

def pose_bank():
    bank=json.loads(POSE_FILE.read_text())
    assert bank['model_sha256']==hashlib.sha256(base.MODEL.read_bytes()).hexdigest()
    return bank

def reset_pose(env, env_ids):
    mdp.reset_scene_to_default(env,env_ids)
    if env_ids is None:env_ids=torch.arange(env.num_envs,device=env.device)
    robot=env.scene['robot'];n=len(env_ids)
    if not hasattr(env,'_v5_pose_bank'):
        bank=pose_bank();rows=sorted(bank['poses'],key=lambda x:x['height'])
        order=[bank['joint_names'].index(n) for n in robot.joint_names]
        env._v5_pose_bank=torch.tensor([[r['qpos'][7+i] for i in order] for r in rows],device=env.device)
        env._v5_pose_heights=torch.tensor([r['height'] for r in rows],device=env.device)
        env._v5_initial_height=torch.full((env.num_envs,),.24,device=env.device)
        env._v5_recover=torch.zeros(env.num_envs,dtype=torch.bool,device=env.device)
    progress=min(1.,env.common_step_counter/(24*400))
    minimum=.20-.04*progress
    forced=getattr(env,'_v5_reset_height',None)
    requested=torch.empty(n,device=env.device).uniform_(minimum,.221)
    recovery=torch.rand(n,device=env.device)<.60
    if forced is not None:
        if isinstance(forced,torch.Tensor):forced=forced[env_ids]
        requested[:]=forced;recovery[:]=forced<.23
    requested=torch.where(recovery,requested,.24)
    index=(requested[:,None]-env._v5_pose_heights[None,:]).abs().argmin(1)
    h=env._v5_pose_heights[index]
    env._v5_initial_height[env_ids]=h;env._v5_recover[env_ids]=recovery
    state=robot.data.default_root_state[env_ids].clone()
    state[:,:3]+=env.scene.env_origins[env_ids];state[:,2]=h+env.scene.env_origins[env_ids,2]
    robot.write_root_state_to_sim(state,env_ids=env_ids)
    robot.write_joint_state_to_sim(env._v5_pose_bank[index],torch.zeros_like(env._v5_pose_bank[index]),env_ids=env_ids)


class V5Command(fast.FastCommand):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        self.phase=torch.zeros(self.num_envs,dtype=torch.long,device=self.device)
        self.age=torch.zeros(self.num_envs,device=self.device)
        self.reference_height=torch.full_like(self.age,.24)
        self.jump_target=torch.full_like(self.age,.03)
        self.air_seen=torch.zeros(self.num_envs,dtype=torch.bool,device=self.device)
        self.request_jump=False
        self.auto_jump=not cfg.full_range

    def _resample_command(self,env_ids):
        super()._resample_command(env_ids)
        fresh=self.command_counter[env_ids]==0
        ids=env_ids[fresh]
        self.phase[ids]=torch.where(self._env._v5_recover[ids],1,0)
        self.age[ids]=0.;self.air_seen[ids]=False
        self.reference_height[ids]=self._env._v5_initial_height[ids]
        self.height_command[ids]=self.reference_height[ids]
        # Keep existing drive, reverse and spin examples. No high-stance driving.
        self.vel_command_b[ids]=0.
        self.height_target[ids]=.235

    def begin_jump(self,ids):
        self.phase[ids]=2;self.age[ids]=0.;self.air_seen[ids]=False
        self.reference_height[ids]=.24
        progress=1. if self.cfg.full_range else max(0.,min(1.,(self._env.common_step_counter-24*150)/(24*650)))
        self.jump_target[ids]=.02+.04*progress

    def compute(self,dt):
        super().compute(dt)
        self.age+=dt
        old=self.phase.clone()
        if self.request_jump:
            ids=((old==0)&(self.robot.data.root_link_lin_vel_b[:,0].abs()<.12)).nonzero().flatten()
            self.begin_jump(ids);self.request_jump=False
        if self.auto_jump and self._env.common_step_counter>24*150:
            eligible=(old==0)&(self.age>2.)
            ids=(eligible&(torch.rand(self.num_envs,device=self.device)<dt*.35)).nonzero().flatten()
            self.begin_jump(ids)
        support=base.wheel_support(self._env)
        airborne=(support.sum(1)==0)&(wheel_clearance(self._env)>.005)
        self.air_seen|=airborne&((old==3)|(old==4))
        changes=[((old==1)&(self.age>1.8),0),((old==2)&(self.age>=.40),3),
                 ((old==3)&((self.age>=.22)|airborne),4),
                 ((old==4)&(((support.sum(1)>0)&self.air_seen&(self.age>.08))|(self.age>.65)),5),
                 ((old==5)&(self.age>1.2),0)]
        for mask,p in changes:
            self.phase[mask]=p;self.age[mask]=0.
        active=self.phase!=0
        self.vel_command_b[active]=0.;self.height_target[active]=.235
        initial=self._env._v5_initial_height
        self.reference_height=torch.where(self.phase==1,initial+(.24-initial)*(self.age/.9).clamp(0,1),self.reference_height)
        self.reference_height=torch.where(self.phase==2,.24-.06*(self.age/.4).clamp(0,1),self.reference_height)
        self.reference_height=torch.where(self.phase==3,.30,self.reference_height)
        self.reference_height=torch.where((self.phase==4)|(self.phase==5),.24,self.reference_height)
        self.reference_height=torch.where(self.phase==0,.24,self.reference_height)
        self.height_command[active]=self.reference_height[active]


@dataclass(kw_only=True)
class V5CommandCfg(fast.FastCommandCfg):
    def build(self,env):return V5Command(self,env)


class V5LegAction(JointPositionAction):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        rows=sorted(pose_bank()['poses'],key=lambda x:x['height'])
        self.heights=torch.tensor([r['height'] for r in rows],device=self.device)
        self.poses=torch.tensor([[r['drive'][n] for n in self.target_names] for r in rows],device=self.device)

    def process_actions(self,actions):
        super().process_actions(actions)
        cmd=self._env.command_manager.get_term('twist')
        h=cmd.reference_height.clamp(self.heights[0],self.heights[-1])
        hi=torch.searchsorted(self.heights,h).clamp(1,len(self.heights)-1);lo=hi-1
        mix=((h-self.heights[lo])/(self.heights[hi]-self.heights[lo]))[:,None]
        reference=self.poses[lo]*(1-mix)+self.poses[hi]*mix
        self._processed_actions+=((cmd.phase!=0)[:,None]*(reference-self._offset))


@dataclass(kw_only=True)
class V5LegActionCfg(JointPositionActionCfg):
    def build(self,env):return V5LegAction(self,env)


def wheel_clearance(env):
    robot=env.scene['robot']
    if not hasattr(env,'_v5_wheel_bodies'):
        env._v5_wheel_bodies=robot.find_bodies(('l_wheel_Link','r_wheel_Link'),preserve_order=True)[0]
    return (robot.data.body_link_pos_w[:,env._v5_wheel_bodies,2]-env.scene.env_origins[:,None,2]-.06).amin(1)

def observations(env):
    cmd=env.command_manager.get_term('twist')
    return torch.cat((v4.observations(env),torch.nn.functional.one_hot(cmd.phase,6).float(),
        (cmd.age/2).clamp(0,1)[:,None],(cmd.jump_target*10)[:,None]),1)

def locomotion_reward(env,func,params):
    return func(env,**params)*(env.command_manager.get_term('twist').phase==0)

def reward(env,name):
    cmd=env.command_manager.get_term('twist');p=cmd.phase;d=env.scene['robot'].data
    posture=torch.exp(-(base.tilt_angle(env)/.25).square())
    active=(p!=0).float();ground=((p==1)|(p==2)|(p==5)).float()
    vz=d.root_link_lin_vel_w[:,2];speed=d.root_link_lin_vel_b[:,0]
    if name=='maneuver_posture':return active*posture
    if name=='reference_height':return ground*posture*torch.exp(-((base.height(env)-cmd.reference_height)/.035).square())
    if name=='takeoff':return (p==3)*posture*torch.exp(-((vz-1.0)/.65).square())
    if name=='flight':
        air=(base.wheel_support(env).sum(1)==0)&(wheel_clearance(env)>.005)
        return (p==4)*air*posture*(wheel_clearance(env)/cmd.jump_target).clamp(0,1)
    if name=='landing':return (p==5)*cmd.air_seen*posture*base.wheel_support(env).amin(1)*torch.exp(-(vz/.3).square()-(speed/.12).square())
    if name=='maneuver_drift':return active*(speed.square()+d.root_link_ang_vel_b[:,2].square()*.2)
    if name=='quiet':
        quiet=(p==0)&(cmd.vel_command_b.abs().sum(1)<.01)&(cmd.height_target<.25)&(cmd.age>1.)
        return quiet*(speed/.05).square()
    if name=='teacher':
        teacher=getattr(env,'_v5_teacher',None)
        if teacher is None:return torch.zeros_like(vz)
        with torch.no_grad():target=teacher(v4.observations(env)).clamp(-1,1)
        return (p==0)*(env.action_manager.action-target).square().mean(1)
    raise KeyError(name)

def bad_pose(env):
    return (base.tilt_angle(env)>.65)|(base.height(env)<.10)|(base.height(env)>.8)

def bad_contact(env):
    cmd=env.command_manager.get_term('twist')
    grace=(cmd.phase==1)&(cmd.age<1.2)
    return base.bad_contact(env)&~grace

def metric(env,name):
    cmd=env.command_manager.get_term('twist')
    if name=='airborne':return ((base.wheel_support(env).sum(1)==0)&(wheel_clearance(env)>.005)).float()
    if name=='wheel_clearance_m':return wheel_clearance(env).clamp_min(0)
    if name=='recovery_fraction':return (cmd.phase==1).float()
    raise KeyError(name)

def make_cfg(num_envs=1024,task='v5',play=False):
    cfg=fast.make_cfg(num_envs,play=play)
    cfg.commands['twist']=V5CommandCfg(**vars(cfg.commands['twist']))
    cfg.commands['twist'].resampling_time_range=(4.,6.)
    cfg.actions['legs']=V5LegActionCfg(**vars(cfg.actions['legs']))
    cfg.events={'reset':EventTermCfg(func=reset_pose,mode='reset')}
    for group in cfg.observations.values():group.terms['state']=ObservationTermCfg(func=observations)
    # Keep safety/effort penalties in every phase; gate ground tracking during flight.
    for name,term in list(cfg.rewards.items()):
        if name in ('alive','failure','effort','joint_limits','angular_rate','lateral_velocity'):continue
        cfg.rewards[name]=RewardTermCfg(func=locomotion_reward,weight=term.weight,params={'func':term.func,'params':term.params})
    for name,weight in [('maneuver_posture',3.),('reference_height',5.),('takeoff',8.),('flight',12.),
                        ('landing',8.),('maneuver_drift',-3.),('quiet',-.3),('teacher',-2.)]:
        cfg.rewards[name]=RewardTermCfg(func=reward,weight=weight,params={'name':name})
    cfg.terminations['bad_pose']=TerminationTermCfg(func=bad_pose)
    cfg.terminations['nonwheel_contact']=TerminationTermCfg(func=bad_contact)
    for name in ['airborne','wheel_clearance_m','recovery_fraction']:
        cfg.metrics[name]=MetricsTermCfg(func=metric,params={'name':name})
    cfg.episode_length_s=8.
    return cfg
