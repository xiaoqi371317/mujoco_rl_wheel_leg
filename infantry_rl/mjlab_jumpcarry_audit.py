"""Matched 2 m/s, moving-height and moving-jump evaluation with real ground clearance."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import torch
from mjlab.rl import MjlabOnPolicyRunner,RslRlVecEnvWrapper
from infantry_rl.mjlab_train import CheckedEnv
from infantry_rl.mjlab_rugged_audit import verify
from infantry_rl.mjlab_v5_audit import latest
from infantry_rl.mjlab_motion_train import agent
from infantry_rl import mjlab_jumpcarry_task as task
from infantry_rl import mjlab_task as base


def cases():
    rows=[]
    def add(name,typ=0,level=0,speed=0.,yaw=0.,mode='drive'):
        rows.append(dict(name=name,typ=typ,level=level,speed=speed,yaw=yaw,mode=mode))
    for speed in [0.,.8,1.3,1.6,2.]:add(f'flat_drive_{speed}',speed=speed)
    for speed in [0.,.5,1.,1.5,2.]:add(f'flat_high_{speed}',speed=speed,mode='high')
    for speed in [0.,.5,1.,1.5,2.]:add(f'flat_jump_{speed}',speed=speed,mode='jump')
    add('flat_reverse',speed=-.3);add('flat_brake_2',speed=2.,mode='brake')
    add('flat_spin_left',yaw=3.);add('flat_spin_right',yaw=-3.);add('flat_rise',mode='rise')
    for typ,label in [(1,'bumps'),(2,'waves')]:
        for level in [0,2]:
            for speed in [.5,1.3]+([2.] if level==0 else []):
                add(f'{label}_{level}_drive_{speed}',typ,level,speed)
            add(f'{label}_{level}_brake',typ,level,1.,mode='brake')
        for mode in ['high','jump']:
            for speed in [.5,1.]:add(f'{label}_0_{mode}_{speed}',typ,0,speed,mode=mode)
    return rows


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--checkpoint',type=Path);ap.add_argument('--out',type=Path);ap.add_argument('--repeats',type=int,default=4)
    args=ap.parse_args();verify(args.run);torch.set_num_threads(4)
    rows=cases();r=args.repeats;n=len(rows)*r;checkpoint=args.checkpoint or latest(args.run)
    cfg=task.make_cfg(n,play=True,seed=1043);cfg.seed=2046;cfg.episode_length_s=10.
    cfg.scene.terrain.terrain_generator.size=(44.,12.);cfg.terminations.pop('patch_boundary')
    env=CheckedEnv(cfg,device='cuda:0');vec=RslRlVecEnvWrapper(env,clip_actions=1.)
    try:
        runner=MjlabOnPolicyRunner(vec,asdict(agent()),device='cuda:0')
        runner.load(str(checkpoint),load_cfg={'actor':True},map_location='cuda:0')
        policy=runner.get_inference_policy(device='cuda:0')
        def values(key):return torch.tensor([row[key] for row in rows],device=env.device).repeat_interleave(r)
        def modes(mode):return torch.tensor([row['mode']==mode for row in rows],device=env.device).repeat_interleave(r)
        jump,high,brake,rise=[modes(mode) for mode in ['jump','high','brake','rise']]
        env._rough_force_type=values('typ');env._rough_force_level=values('level')
        env._rough_initial_height=torch.where(rise,.16,.24)
        desired=torch.stack((-values('speed'),torch.zeros(n,device=env.device),values('yaw')),1)
        height=torch.full((n,),.235,device=env.device)
        cmd=env.command_manager.get_term('twist');cmd.auto_jump=False;cmd.auto_height=False;cmd.extra_jumps=False
        cmd.external_source=lambda:(desired,height)
        vec.reset();robot=env.scene['robot']
        pose=robot.data.root_link_pose_w.clone();pitch=torch.linspace(-.012,.012,r,device=env.device).repeat(len(rows))
        pose[:,3]=torch.cos(pitch/2);pose[:,5]=torch.sin(pitch/2);robot.write_root_link_pose_to_sim(pose)
        alive=torch.ones(n,dtype=torch.bool,device=env.device)
        count=torch.zeros(n,device=env.device);sums=torch.zeros(n,4,device=env.device);good=count.clone()
        peak=count.clone();air_run=count.clone();airtime=count.clone();upward=count.clone()
        jump_error=count.clone();jump_count=count.clone();trigger_speed=count.clone();started=alive.clone()&False
        high_latency=torch.full_like(count,-1.);high_peak=count.clone()
        ground_min=torch.full_like(count,float('inf'));ground_max=-ground_min
        carry_sum=torch.zeros(n,4,device=env.device);carry_count=carry_sum.clone();carry_min=torch.full_like(carry_sum,1.)
        before_world=torch.zeros(n,2,device=env.device)
        failed={name:torch.zeros_like(alive) for name in env.termination_manager.active_terms}
        with torch.no_grad():
            for step in range(450):
                if step==100:height[high]=.28
                if step==150:
                    trigger_speed[:]=robot.data.root_link_lin_vel_b[:,0]
                    before_world[:]=robot.data.root_link_lin_vel_w[:,:2]
                    cmd.begin_jump((jump&alive).nonzero().flatten());started=jump&(cmd.phase==2)
                if step==250:desired[brake]=0.
                if step==300:height[high]=.235
                cmd.compute(0.)
                _,_,done,_=vec.step(policy(vec.get_observations()))
                ended=alive&done.bool()
                for name in failed:failed[name]|=ended&env.termination_manager.get_term(name)
                alive&=~done.bool();d=robot.data
                ex=(d.root_link_lin_vel_b[:,0]-desired[:,0]).abs()
                ez=(d.root_link_ang_vel_b[:,2]-desired[:,2]).abs()
                tilt=base.tilt_angle(env);h=task.local_height(env);eh=(h-height).abs()
                stable=alive&(step>=100)&(~jump|(step>=350))&(~brake|(step>=350))
                stable&=~high|((step>=125)&(step<300))|(step>=325)
                quality=(ex<.2)&(ez<.35)&(tilt<.25)&(eh<.03)
                stop=(desired.abs().sum(1)<.01)&~jump
                quality&=~stop|(d.root_link_lin_vel_b[:,0].abs()<.04)
                count+=stable;good+=stable&quality
                sums+=torch.stack((ex,ez,tilt,eh),1)*stable[:,None]
                ground=env.scene['local_ground'].data.hit_pos_w[:,0,2]
                ground_min=torch.minimum(ground_min,torch.where(stable,ground,ground_min))
                ground_max=torch.maximum(ground_max,torch.where(stable,ground,ground_max))
                if 100<=step<300:
                    high_peak=torch.maximum(high_peak,torch.where(high&alive,h,0.))
                    reached=high&alive&(high_latency<0)&(h>=.270)
                    high_latency[reached]=(step-100)*env.step_dt
                if 150<=step<350:
                    pre_norm=before_world.norm(dim=1)
                    projection=(d.root_link_lin_vel_w[:,:2]*before_world).sum(1)/pre_norm.clamp_min(.1)
                    relative=projection/pre_norm.clamp_min(.1)
                    measured_error=(d.root_link_lin_vel_w[:,:2]-before_world).norm(dim=1)
                    for j,phase in enumerate([2,3,4,5]):
                        valid=alive&jump&(cmd.phase==phase)
                        carry_sum[:,j]+=valid*measured_error;carry_count[:,j]+=valid
                        carry_min[:,j]=torch.where(valid&(pre_norm>.2),torch.minimum(carry_min[:,j],relative),carry_min[:,j])
                    clearance=task.wheel_clearance(env)
                    air=alive&jump&(base.wheel_support(env).sum(1)==0)&(clearance>.005)
                    peak=torch.maximum(peak,torch.where(air,clearance,0.))
                    air_run=torch.where(air,air_run+env.step_dt,0.);airtime=torch.maximum(airtime,air_run)
                    upward=torch.maximum(upward,torch.where(alive&jump,d.root_link_lin_vel_w[:,2],0.))
                    jump_error+=alive*jump*ex;jump_count+=alive&jump
        mean=sums/count.clamp_min(1)[:,None];tracking=good/count.clamp_min(1)
        report=dict(checkpoint=str(checkpoint),repeats=r,seconds=9.,terrain_seed=1043,patch_size_m=[44,12],cases={})
        for k,row in enumerate(rows):
            sl=slice(k*r,(k+1)*r);passed=alive[sl]&(tracking[sl]>.8)
            if row['mode']=='high':passed&=(high_latency[sl]>=0)&(high_latency[sl]<=.8)
            if row['mode']=='jump':
                passed&=started[sl]&(peak[sl]>.04)&(airtime[sl]>=.08)&(upward[sl]>.7)
                passed&=(jump_error/jump_count.clamp_min(1))[sl]<.3
                if row['speed']>.1:passed&=-trigger_speed[sl]>.65*row['speed']
            original_passed=passed.clone()
            if row['mode']=='jump' and row['speed']>.1:
                measured=(carry_count[sl]>0).all(1)
                passed&=measured&(carry_min[sl].amin(1)>.8)&((carry_sum[sl].sum(1)/carry_count[sl].sum(1).clamp_min(1))<.2)
            report['cases'][row['name']]=dict(mode=row['mode'],speed=row['speed'],terrain=row['typ'],
                passed=bool(passed.all()),original_criteria_passed=bool(original_passed.all()),
                phase_speed_mae=(carry_sum[sl]/carry_count[sl].clamp_min(1)).mean(0).tolist(),
                phase_min_speed_retention=carry_min[sl].amin(0).tolist(),
                phase_samples=carry_count[sl].sum(0).tolist(),success_fraction=float(passed.float().mean()),
                survival_fraction=float(alive[sl].float().mean()),tracking_fraction=float(tracking[sl].mean()),
                speed_mae=float(mean[sl,0].mean()),yaw_mae=float(mean[sl,1].mean()),
                tilt_deg=float(torch.rad2deg(mean[sl,2]).mean()),height_mae=float(mean[sl,3].mean()),
                high_latency_s=high_latency[sl].tolist(),high_peak_m=float(high_peak[sl].mean()),
                jump_clearance_m=float(peak[sl].mean()),airtime_s=float(airtime[sl].mean()),
                takeoff_vz=float(upward[sl].mean()),jump_trigger_forward_mps=float(-trigger_speed[sl].mean()),
                jump_speed_mae=float((jump_error/jump_count.clamp_min(1))[sl].mean()),
                measured_seconds=float((count[sl]*env.step_dt).mean()),
                ground_variation_m=float(torch.where(count[sl]>0,ground_max[sl]-ground_min[sl],0.).mean()),
                failure_counts={key:int(value[sl].sum()) for key,value in failed.items() if value[sl].any()})
        report['passed']=sum(v['passed'] for v in report['cases'].values())
        out=args.out or args.run/(checkpoint.stem+'_jumpcarry_audit.json');out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text(json.dumps(report,indent=2));print('MOTION_AUDIT',out,'passed',report['passed'],'/',len(rows),flush=True)
    finally:vec.close()


if __name__=='__main__':main()
