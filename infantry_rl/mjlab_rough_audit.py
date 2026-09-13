"""Matched held-out terrain evaluation, with flat skill retention checks."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import torch
from mjlab.rl import MjlabOnPolicyRunner,RslRlVecEnvWrapper
from infantry_rl.model_selection import select_model
from infantry_rl.mjlab_train import CheckedEnv
from infantry_rl.mjlab_rough_train import agent
from infantry_rl.mjlab_v5_audit import latest
from infantry_rl import mjlab_rough_task as task
from infantry_rl import mjlab_task as base

def verify(run):
    meta=json.loads((run/'run.json').read_text());model=select_model('fix')
    assert meta['model_sha256']==hashlib.sha256(model.read_bytes()).hexdigest()
    for name,digest in meta['source_hashes'].items():
        if name.endswith(('_train.py','_audit.py','_play.py')):continue
        assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()==digest,name
    return meta

def cases():
    rows=[]
    for typ,label in [(0,'flat'),(1,'bumps'),(2,'waves')]:
        for level in ([0] if typ==0 else [0,2]):
            for mode,x,z in [('stop',0.,0.),('forward_0_4',-.4,0.),('forward_0_8',-.8,0.),
                             ('forward_1_3',-1.3,0.),('reverse',.3,0.),('left_arc',-.5,.8),
                             ('right_arc',-.5,-.8),('brake',-1.,0.)]:
                rows.append((f'{label}_{level}_{mode}',typ,level,x,z,.24,False,mode=='brake'))
    rows.extend([('flat_rise',0,0,0.,0.,.16,False,False),('flat_jump',0,0,0.,0.,.24,True,False),
                 ('flat_rise_jump',0,0,0.,0.,.16,True,False),('flat_spin_left',0,0,0.,3.,.24,False,False),
                 ('flat_spin_right',0,0,0.,-3.,.24,False,False)])
    return rows

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--checkpoint',type=Path)
    ap.add_argument('--out',type=Path);ap.add_argument('--repeats',type=int,default=4)
    args=ap.parse_args();meta=verify(args.run);torch.set_num_threads(4)
    rows=cases();r=args.repeats;n=len(rows)*r;checkpoint=args.checkpoint or latest(args.run)
    cfg=task.make_cfg(n,play=True,seed=1043);cfg.seed=2043;cfg.episode_length_s=10.
    cfg.scene.terrain.terrain_generator.size=(28.,12.)
    cfg.terminations.pop('patch_boundary')
    env=CheckedEnv(cfg,device='cuda:0');vec=RslRlVecEnvWrapper(env,clip_actions=1.)
    try:
        runner=MjlabOnPolicyRunner(vec,asdict(agent()),device='cuda:0')
        runner.load(str(checkpoint),load_cfg={'actor':True},map_location='cuda:0')
        policy=runner.get_inference_policy(device='cuda:0')
        def values(index,dtype=None):return torch.tensor([row[index] for row in rows],device=env.device,dtype=dtype).repeat_interleave(r)
        env._rough_force_type=values(1);env._rough_force_level=values(2);env._rough_initial_height=values(5)
        jump=values(6);brake=values(7);desired=torch.stack((values(3),torch.zeros(n,device=env.device),values(4)),1)
        cmd=env.command_manager.get_term('twist');cmd.preserve_jumps=False;original=cmd.compute
        def fixed(dt):
            original(dt);idle=cmd.phase==0
            cmd.vel_command_b[idle]=desired[idle];cmd.height_target[:]=.235;cmd.height_command[idle]=.235
        cmd.compute=fixed;vec.reset()
        # Perturb the common root orientation, retaining linkage closure.
        robot=env.scene['robot'];pose=robot.data.root_link_pose_w.clone()
        pitch=torch.linspace(-.012,.012,r,device=env.device).repeat(len(rows))
        pose[:,3]=torch.cos(pitch/2);pose[:,5]=torch.sin(pitch/2);robot.write_root_link_pose_to_sim(pose)
        alive=torch.ones(n,device=env.device,dtype=torch.bool)
        counts=torch.zeros(n,device=env.device);sums=torch.zeros(n,5,device=env.device)
        good_count=counts.clone();peak=counts.clone();air_run=counts.clone();longest=counts.clone()
        total_ground=counts.clone();ground_min=torch.full_like(counts,float('inf'));ground_max=-ground_min
        failed={name:torch.zeros(n,device=env.device,dtype=torch.bool) for name in env.termination_manager.active_terms}
        with torch.inference_mode():
            for step in range(450):
                if step==150:cmd.begin_jump(jump.nonzero().flatten())
                if step==250:desired[brake]=0.
                fixed(0.);obs=vec.get_observations();_,_,done,_=vec.step(policy(obs))
                ended=alive&done.bool()
                for name in failed:failed[name]|=ended&env.termination_manager.get_term(name)
                alive&=~done.bool();d=robot.data
                vx=d.root_link_lin_vel_b[:,0];yaw=d.root_link_ang_vel_b[:,2];tilt=base.tilt_angle(env)
                ex=(vx-desired[:,0]).abs();ez=(yaw-desired[:,2]).abs();eh=(task.local_height(env)-.235).abs()
                active=alive&(step>=100)&(~(brake|jump)|(step>=350))
                counts+=active
                sums+=torch.stack((ex,ez,tilt,eh,vx.square()),1)*active[:,None]
                good=(ex<.15)&(ez<.3)&(tilt<.25)&(eh<.025)
                strict_stop=(desired.abs().sum(1)<.01)&~jump
                good&=~strict_stop|(vx.abs()<.03)
                good_count+=active&good
                ground=env.scene['local_ground'].data.hit_pos_w[:,0,2]
                if step==100:
                    assert ground[env._rough_flat&alive].abs().max().item()<1e-5,'Flat ray hit height is not zero'
                ground_min=torch.minimum(ground_min,torch.where(active,ground,ground_min))
                ground_max=torch.maximum(ground_max,torch.where(active,ground,ground_max))
                total_ground+=active*(env.scene['local_ground'].data.distances>=0).all(1)
                if step>=150:
                    clearance=v5_clearance(env)
                    air=alive&(base.wheel_support(env).sum(1)==0)&(clearance>.005)
                    peak=torch.maximum(peak,torch.where(air,clearance,0.))
                    air_run=torch.where(air,air_run+env.step_dt,0.);longest=torch.maximum(longest,air_run)
        mean=sums/counts.clamp_min(1)[:,None];quality=good_count/counts.clamp_min(1)
        report={'checkpoint':str(checkpoint),'terrain_seed':1043,'seed':2043,'repeats':r,'seconds':9.,
            'patch_size_m':[28,12],'cases':{}}
        for k,row in enumerate(rows):
            name,typ,level,x,z,h,j,b=row;sl=slice(k*r,(k+1)*r)
            passed=alive[sl]&(quality[sl]>.8)
            if j:passed&=(peak[sl]>.02)&(longest[sl]>=.06)
            report['cases'][name]=dict(passed=bool(passed.all()),success_fraction=float(passed.float().mean()),
                survival_fraction=float(alive[sl].float().mean()),speed_mae=float(mean[sl,0].mean()),
                yaw_mae=float(mean[sl,1].mean()),tilt_deg=float(torch.rad2deg(mean[sl,2]).mean()),
                local_height_mae_m=float(mean[sl,3].mean()),speed_rms=float(torch.sqrt(mean[sl,4]).mean()),
                tracking_fraction=float(quality[sl].mean()),ground_variation_m=float((ground_max[sl]-ground_min[sl]).nan_to_num().mean()),
                ground_hit_fraction=float((total_ground/counts.clamp_min(1))[sl].mean()),
                jump_clearance_m=float(peak[sl].mean()),airtime_s=float(longest[sl].mean()),
                failure_counts={key:int(value[sl].sum()) for key,value in failed.items() if value[sl].any()})
        report['all_passed']=all(v['passed'] for v in report['cases'].values())
        out=args.out or args.run/(checkpoint.stem+'_rough_audit.json');out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text(json.dumps(report,indent=2));print('ROUGH_AUDIT',json.dumps(report),flush=True)
    finally:vec.close()

def v5_clearance(env):
    # Jump acceptance is tested only on the zero-elevation flat column.
    from infantry_rl.mjlab_v5_task import wheel_clearance
    return wheel_clearance(env)

if __name__=='__main__':main()
