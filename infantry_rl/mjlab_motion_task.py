"""Continue V5 with moving height changes, moving jumps and 2 m/s commands."""
from dataclasses import dataclass
import torch
from mjlab.managers import RewardTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.sensor import TerrainHeightSensorCfg,ObjRef,GridPatternCfg
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand
from infantry_rl import mjlab_task as base
from infantry_rl import mjlab_v5_task as v5
from infantry_rl import mjlab_rough_task as rough
from infantry_rl import mjlab_rugged_task as rugged

local_height=rough.local_height


def iteration(env):
    return max(0,env.common_step_counter-getattr(env,'_motion_start_step',env.common_step_counter))/24


def wheel_clearance(env):
    robot=env.scene['robot']
    if not hasattr(env,'_motion_wheels'):
        env._motion_wheels=robot.find_bodies(('l_wheel_Link','r_wheel_Link'),preserve_order=True)[0]
    z=robot.data.body_link_pos_w[:,env._motion_wheels,2]
    sensors=[env.scene[name].data for name in ('ground_left','ground_right')]
    ground=torch.stack([s.hit_pos_w[:,0,2] for s in sensors],1)
    valid=torch.stack([s.distances[:,0]>=0 for s in sensors],1).all(1)
    return torch.where(valid,(z-ground-.06).amin(1),torch.zeros_like(z[:,0]))


class MotionCommand(v5.V5Command):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        self.drive=torch.zeros_like(self.vel_command_b)
        self.drive_height=torch.full_like(self.age,.235)
        self.jump_start_height=torch.full_like(self.age,.24)
        self.auto_jump=not cfg.full_range
        self.auto_height=not cfg.full_range
        self.external_source=None

    def progress(self,rounds):
        return 1. if self.cfg.full_range else min(1.,iteration(self._env)/rounds)

    def _resample_command(self,env_ids):
        fresh=self.command_counter[env_ids]==0
        ids=env_ids[fresh|(self.phase[env_ids]==0)];n=len(ids)
        bucket=torch.rand(n,device=self.device)
        maximum=torch.full((n,),1.3+.7*self.progress(400),device=self.device)
        flat=self._env._rough_flat[ids];level=self._env.scene.terrain.terrain_levels[ids]
        rough_cap=torch.where(level==0,2.,torch.where(level==1,1.6,1.3))
        maximum=torch.where(flat,maximum,torch.minimum(maximum,rough_cap))
        self.drive[ids]=0.;self.drive_height[ids]=.235
        forward=(bucket>=.15)&(bucket<.75)
        speed=.25+(maximum-.25)*torch.rand(n,device=self.device)
        speed=torch.where(torch.rand(n,device=self.device)<.35,maximum,speed)
        high=((bucket>=.50)&(bucket<.75))&(flat|(level==0))
        high_max=.5+1.5*self.progress(600)
        speed=torch.where(high,speed.clamp_max(high_max),speed)
        self.drive[ids[forward],0]=-speed[forward]
        self.drive_height[ids[high|(bucket<.075)]]=.28
        reverse=(bucket>=.75)&(bucket<.85)
        self.drive[ids[reverse],0]=torch.empty(int(reverse.sum()),device=self.device).uniform_(.1,.5)
        arc=(bucket>=.85)&(bucket<.95)
        self.drive[ids[arc],0]=-torch.empty(int(arc.sum()),device=self.device).uniform_(.3,.8)
        self.drive[ids[arc],2]=torch.empty(int(arc.sum()),device=self.device).uniform_(-1.,1.)
        spin=bucket>=.95
        spin_speed=torch.where(flat[spin],3.,1.2)
        self.drive[ids[spin],2]=torch.where(torch.rand(int(spin.sum()),device=self.device)<.5,-spin_speed,spin_speed)
        self.is_standing_env[ids]=False
        fresh_ids=env_ids[fresh]
        self.phase[fresh_ids]=torch.where(self._env._v5_recover[fresh_ids],1,0)
        self.age[fresh_ids]=0.;self.air_seen[fresh_ids]=False
        self.reference_height[fresh_ids]=self._env._v5_initial_height[fresh_ids]
        self.height_command[fresh_ids]=self.reference_height[fresh_ids]
        self.vel_command_b[ids]=self.drive[ids]
        self.height_target[ids]=self.drive_height[ids]

    def begin_jump(self,ids):
        ids=ids[(self.phase[ids]==0)&(base.wheel_support(self._env)[ids].sum(1)>0)]
        self.jump_start_height[ids]=self.height_command[ids]
        self.phase[ids]=2;self.age[ids]=0.;self.air_seen[ids]=False
        self.jump_target[ids]=.04+.02*self.progress(600)

    def compute(self,dt):
        # Bypass FastCommand's moving-height restriction and V5's forced stopping.
        UniformVelocityCommand.compute(self,dt)
        if self.external_source is not None:
            drive,height=self.external_source();self.drive[:]=drive;self.drive_height[:]=height
        elif self.manual_source is not None:
            vx,wz,h=self.manual_source()
            self.drive[:]=torch.tensor([vx,0.,wz],device=self.device);self.drive_height[:]=h
        self.age+=dt
        if self.auto_height:
            eligible=(self.phase==0)&(self.age>1.)&self._env._rough_flat&(self.drive[:,0].abs()>.1)
            eligible&=self.drive[:,0].abs()<=.5+1.5*self.progress(600)
            change=eligible&(torch.rand(self.num_envs,device=self.device)<dt*.22)
            self.drive_height[change]=torch.where(self.drive_height[change]>.25,.235,.28)
        if self.request_jump:
            self.begin_jump((self.phase==0).nonzero().flatten());self.request_jump=False
        if self.auto_jump:
            eligible=(self.phase==0)&(self.age>1.5)&(self.drive[:,2].abs()<.2)&(self.drive_height<.25)
            terrain_ok=self._env._rough_flat|((self._env.scene.terrain.terrain_levels==0)&(iteration(self._env)>400))
            jump_max=.3+1.7*max(0.,min(1.,(iteration(self._env)-100)/600))
            eligible&=terrain_ok&(self.drive[:,0].abs()<=jump_max)
            self.begin_jump((eligible&(torch.rand(self.num_envs,device=self.device)<dt*.35)).nonzero().flatten())
        old=self.phase.clone();support=base.wheel_support(self._env)
        airborne=(support.sum(1)==0)&(wheel_clearance(self._env)>.005)
        self.air_seen|=airborne&((old==3)|(old==4))
        for mask,phase in [((old==1)&(self.age>1.8),0),((old==2)&(self.age>=.40),3),
            ((old==3)&((self.age>=.22)|airborne),4),
            ((old==4)&(((support.sum(1)>0)&self.air_seen&(self.age>.08))|(self.age>.65)),5),
            ((old==5)&(self.age>1.2),0)]:
            self.phase[mask]=phase;self.age[mask]=0.
        self.vel_command_b[:]=self.drive
        self.vel_command_b[self.phase==1]=0. # Only recovery from a crouched reset starts stationary.
        self.height_target[:]=self.drive_height
        self.height_command+=(self.height_target-self.height_command).clamp(-.18*dt,.18*dt)
        initial=self._env._v5_initial_height
        self.reference_height=torch.where(self.phase==1,initial+(.24-initial)*(self.age/.9).clamp(0,1),self.reference_height)
        crouch=self.jump_start_height+(.18-self.jump_start_height)*(self.age/.4).clamp(0,1)
        self.reference_height=torch.where(self.phase==2,crouch,self.reference_height)
        self.reference_height=torch.where(self.phase==3,.30,self.reference_height)
        self.reference_height=torch.where((self.phase==4)|(self.phase==5),.24,self.reference_height)
        self.reference_height=torch.where(self.phase==0,.24,self.reference_height)
        active=self.phase!=0;self.height_command[active]=self.reference_height[active]


@dataclass(kw_only=True)
class MotionCommandCfg(v5.V5CommandCfg):
    def build(self,env):return MotionCommand(self,env)


def reward(env,name):
    cmd=env.command_manager.get_term('twist');p=cmd.phase;d=env.scene['robot'].data
    active=p!=0;posture=torch.exp(-(base.tilt_angle(env)/.25).square())
    ex=d.root_link_lin_vel_b[:,0]-cmd.vel_command_b[:,0]
    ez=d.root_link_ang_vel_b[:,2]-cmd.vel_command_b[:,2]
    vz=d.root_link_lin_vel_w[:,2]
    if name=='reference_height':
        return ((p==1)|(p==2)|(p==5))*posture*torch.exp(-((local_height(env)-cmd.reference_height)/.035).square())
    if name=='flight':
        air=(base.wheel_support(env).sum(1)==0)&(wheel_clearance(env)>.005)
        return (p==4)*air*posture*(wheel_clearance(env)/cmd.jump_target).clamp(0,1)
    if name=='landing':
        return (p==5)*cmd.air_seen*posture*base.wheel_support(env).amin(1)*torch.exp(-(vz/.4).square()-(ex/.3).square())
    if name=='maneuver_drift':return active*(ex.square()+.2*ez.square()+.3*d.root_link_lin_vel_b[:,1].square())
    if name=='moving_tracking':return active*posture*torch.exp(-(ex/.3).square()-(ez/.5).square())
    if name=='teacher':
        teacher=getattr(env,'_motion_teacher',None)
        if teacher is None:return torch.zeros_like(ex)
        # Old policy only supervises its established low-stance, <=1.3 m/s driving region.
        eligible=(p==0)&(cmd.height_target<.25)&(cmd.vel_command_b[:,0].abs()<=1.3)&env._rough_flat
        with torch.no_grad():target=teacher(rough.observations(env)).clamp(-1,1)
        return eligible*(env.action_manager.action-target).square().mean(1)
    raise KeyError(name)


def metric(env,name):
    cmd=env.command_manager.get_term('twist')
    if name=='airborne':return ((base.wheel_support(env).sum(1)==0)&(wheel_clearance(env)>.005)).float()
    if name=='wheel_clearance_m':return wheel_clearance(env).clamp_min(0)
    if name=='moving_jump_fraction':return ((cmd.phase>=2)&(cmd.vel_command_b[:,0].abs()>.1)).float()
    if name=='moving_high_fraction':return ((cmd.phase==0)&(cmd.height_target>.25)&(cmd.vel_command_b[:,0].abs()>.1)).float()
    if name=='forward_target_mps':return (-cmd.vel_command_b[:,0]).clamp_min(0)
    return rough.metric(env,name)


def make_cfg(num_envs=1024,play=False,seed=43):
    cfg=rugged.make_cfg(num_envs,play=play,seed=seed)
    for sub,ratio in zip(cfg.scene.terrain.terrain_generator.sub_terrains.values(),(.6,.25,.15)):sub.proportion=ratio
    cfg.commands['twist']=MotionCommandCfg(**vars(cfg.commands['twist']))
    cfg.commands['twist'].ranges.lin_vel_x=(-2.,.5)
    for side in ('left','right'):
        cfg.scene.sensors=(*cfg.scene.sensors,TerrainHeightSensorCfg(name='ground_'+side,
            frame=ObjRef(type='body',name=side[0]+'_wheel_Link',entity='robot'),
            pattern=GridPatternCfg(size=(0.,0.),resolution=.1),ray_alignment='world',
            include_geom_groups=(5,),max_distance=2.,debug_vis=False))
    for name,weight in [('reference_height',5.),('flight',12.),('landing',8.),('maneuver_drift',-3.),
                        ('moving_tracking',6.),('teacher',-1.)]:
        cfg.rewards[name]=RewardTermCfg(func=reward,weight=weight,params={'name':name})
    for name in ['airborne','wheel_clearance_m','moving_jump_fraction','moving_high_fraction','forward_target_mps']:
        cfg.metrics[name]=MetricsTermCfg(func=metric,params={'name':name})
    return cfg
