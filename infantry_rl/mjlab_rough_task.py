"""V5 continuation for mild uneven ground; same 56 observations and six actions."""
from dataclasses import dataclass
import torch
from mjlab.managers import EventTermCfg,ObservationTermCfg,RewardTermCfg,TerminationTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.sensor import TerrainHeightSensorCfg,ObjRef,GridPatternCfg
from infantry_rl import mjlab_task as base
from infantry_rl import mjlab_v5_task as v5
from infantry_rl import mjlab_cmdvel_v4_task as v4
from infantry_rl.rough_terrain import terrain_cfg

def run_steps(env):return max(0,env.common_step_counter-getattr(env,'_rough_start_step',env.common_step_counter))

def reset_pose(env,env_ids):
    if env_ids is None:env_ids=torch.arange(env.num_envs,device=env.device)
    terrain=env.scene.terrain
    if terrain.terrain_origins is not None:
        cap=min(terrain.terrain_origins.shape[0]-1,run_steps(env)//(24*150))
        level=getattr(env,'_rough_force_level',None)
        if isinstance(level,torch.Tensor):level=level[env_ids]
        terrain.terrain_levels[env_ids]=torch.randint(cap+1,(len(env_ids),),device=env.device) if level is None else level
        typ=getattr(env,'_rough_force_type',None)
        if isinstance(typ,torch.Tensor):typ=typ[env_ids]
        if typ is not None:terrain.terrain_types[env_ids]=typ
        terrain.env_origins[env_ids]=terrain.terrain_origins[terrain.terrain_levels[env_ids],terrain.terrain_types[env_ids]]
        flat=terrain.terrain_types[env_ids]==0
    else:flat=torch.ones(len(env_ids),dtype=torch.bool,device=env.device)
    if not hasattr(env,'_rough_flat'):env._rough_flat=torch.ones(env.num_envs,dtype=torch.bool,device=env.device)
    env._rough_flat[env_ids]=flat
    # V5's solved joint bank is retained; only flat examples start in a crouch.
    forced=getattr(env,'_rough_initial_height',None)
    if isinstance(forced,torch.Tensor):forced=forced[env_ids]
    requested=torch.where(flat&(torch.rand(len(env_ids),device=env.device)<.40),.16,.24)
    if forced is not None:requested[:]=forced
    if not hasattr(env,'_v5_reset_height') or not isinstance(env._v5_reset_height,torch.Tensor):
        env._v5_reset_height=torch.full((env.num_envs,),.24,device=env.device)
    env._v5_reset_height[env_ids]=requested
    v5.reset_pose(env,env_ids)

class RoughCommand(v5.V5Command):
    def __init__(self,cfg,env):
        super().__init__(cfg,env);self.auto_jump=False
        self.preserve_jumps=not cfg.full_range
    def _resample_command(self,env_ids):
        super()._resample_command(env_ids)
        rough=env_ids[~self._env._rough_flat[env_ids]];n=len(rough)
        progress=1. if self.cfg.full_range else min(1.,run_steps(self._env)/(24*400))
        vmax=.6+.7*progress
        bucket=torch.rand(n,device=self.device)
        self.vel_command_b[rough]=0.;self.height_target[rough]=.235
        self.is_standing_env[rough]=bucket<.15
        forward=rough[(bucket>=.15)&(bucket<.65)]
        self.vel_command_b[forward,0]=-torch.empty(len(forward),device=self.device).uniform_(.25,vmax)
        reverse=rough[(bucket>=.65)&(bucket<.75)]
        self.vel_command_b[reverse,0]=torch.empty(len(reverse),device=self.device).uniform_(.1,.35)
        arc=rough[(bucket>=.75)&(bucket<.95)]
        self.vel_command_b[arc,0]=-torch.empty(len(arc),device=self.device).uniform_(.2,min(.65,vmax))
        self.vel_command_b[arc,2]=torch.empty(len(arc),device=self.device).uniform_(-1.,1.)
        spin=rough[bucket>=.95]
        self.vel_command_b[spin,2]=torch.empty(len(spin),device=self.device).uniform_(-1.2,1.2)
    def begin_jump(self,ids):
        # This course trains rough-ground mobility, and retains jumps on flat patches.
        super().begin_jump(ids[self._env._rough_flat[ids]])
    def compute(self,dt):
        super().compute(dt)
        if self.preserve_jumps:
            eligible=self._env._rough_flat&(self.phase==0)&(self.age>2.)
            ids=(eligible&(torch.rand(self.num_envs,device=self.device)<dt*.25)).nonzero().flatten()
            self.begin_jump(ids)

@dataclass(kw_only=True)
class RoughCommandCfg(v5.V5CommandCfg):
    def build(self,env):return RoughCommand(self,env)

def local_height(env):
    # TerrainHeightData.heights is a computed cached tensor; reward evaluation can
    # cache it before sense() refreshes the ray buffers. Read the live hit buffer.
    data=env.scene['local_ground'].data
    height=env.scene['robot'].data.root_link_pos_w[:,2]-data.hit_pos_w[:,0,2]
    return torch.where(data.distances[:,0]>=0,height,base.height(env))

def observations(env):
    obs=v5.observations(env)
    # Base observation #41 is the pre-existing scalar root height, not a new scan.
    obs[:,41]=(local_height(env)-.24)*5.
    return obs

def reward(env,name):
    cmd=env.command_manager.get_term('twist');idle=cmd.phase==0
    tilt=base.tilt_angle(env);d=env.scene['robot'].data
    error=(local_height(env)-cmd.height_command).abs()
    if name=='height':return idle*torch.exp(-(tilt/.25).square()-(error/.025).square())
    if name=='height_error':return idle*error
    if name=='teacher':
        teacher=getattr(env,'_rough_teacher',None)
        if teacher is None:return torch.zeros_like(error)
        with torch.no_grad():target=teacher(observations(env)).clamp(-1,1)
        return env._rough_flat*(env.action_manager.action-target).square().mean(1)
    if name=='rough_posture':return idle*(~env._rough_flat)*(d.root_link_ang_vel_b[:,:2].square().sum(1)+4*tilt.square())
    if name=='rough_action_rate':return idle*(~env._rough_flat)*(env.action_manager.action-env.action_manager.prev_action).square().mean(1)
    raise KeyError(name)

def bad_pose(env):return (base.tilt_angle(env)>.65)|(local_height(env)<.10)|(local_height(env)>.8)

def outside_patch(env):
    if env.scene.terrain.terrain_origins is None:return torch.zeros(env.num_envs,dtype=torch.bool,device=env.device)
    delta=env.scene['robot'].data.root_link_pos_w[:,:2]-env.scene.env_origins[:,:2]
    return (delta[:,0].abs()>13.5)|(delta[:,1].abs()>2.5)

def metric(env,name):
    if name=='height_mae_m':return (local_height(env)-env.command_manager.get_term('twist').height_command).abs()
    if name=='terrain_level':return env.scene.terrain.terrain_levels.float()
    if name=='flat_fraction':return env._rough_flat.float()
    if name=='ground_hit_fraction':return (env.scene['local_ground'].data.distances>=0).all(1).float()
    raise KeyError(name)

def make_cfg(num_envs=1024,play=False,seed=43,flat_only=False):
    cfg=v5.make_cfg(num_envs,play=play)
    if not flat_only:cfg.scene.terrain=terrain_cfg(seed)
    else:
        # Put only terrain in group 5, keeping ray casts clear of robot geometry.
        from mjlab.terrains import TerrainEntityCfg
        from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
        from infantry_rl.rough_terrain import GentleTerrainCfg
        cfg.scene.terrain=TerrainEntityCfg(terrain_type='generator',textures=(),materials=(),
            terrain_generator=TerrainGeneratorCfg(seed=seed,curriculum=True,size=(28.,6.),num_rows=1,
                color_scheme='none',sub_terrains={'flat':GentleTerrainCfg(kind='flat')}))
    cfg.scene.sensors=(*cfg.scene.sensors,TerrainHeightSensorCfg(name='local_ground',
        frame=ObjRef(type='body',name='base_Link_del',entity='robot'),pattern=GridPatternCfg(size=(0.,0.),resolution=.1),
        ray_alignment='world',include_geom_groups=(5,),max_distance=2.,debug_vis=False))
    cfg.commands['twist']=RoughCommandCfg(**vars(cfg.commands['twist']))
    cfg.commands['twist'].full_range=play
    cfg.events={'reset':EventTermCfg(func=reset_pose,mode='reset')}
    for group in cfg.observations.values():group.terms['state']=ObservationTermCfg(func=observations)
    for name,weight in [('height',5.),('height_error',-15.),('teacher',-3.),('rough_posture',-.3),('rough_action_rate',-.05)]:
        cfg.rewards[name]=RewardTermCfg(func=reward,weight=weight,params={'name':name})
    cfg.terminations['bad_pose']=TerminationTermCfg(func=bad_pose)
    cfg.terminations['patch_boundary']=TerminationTermCfg(func=outside_patch,time_out=True)
    for name in ['height_mae_m','terrain_level','flat_fraction','ground_hit_fraction']:
        cfg.metrics[name]=MetricsTermCfg(func=metric,params={'name':name})
    cfg.episode_length_s=8.
    return cfg
