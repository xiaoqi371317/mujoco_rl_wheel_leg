"""Thirty-second low/high stance and stop-transition validation."""
import argparse,json
from pathlib import Path
from dataclasses import asdict
import torch
from mjlab.rl import MjlabOnPolicyRunner,RslRlVecEnvWrapper
from infantry_rl.mjlab_train import CheckedEnv
from infantry_rl.mjlab_rugged_audit import verify
from infantry_rl.mjlab_motion_train import agent
from infantry_rl import mjlab_v6_refine_task as task
from infantry_rl import mjlab_task as base

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();meta=verify(args.checkpoint.parent);torch.set_num_threads(4)
    cfg=task.make_cfg(16,play=True,stage=meta.get('stage','stance'));cfg.seed=2091;cfg.episode_length_s=32.
    env=CheckedEnv(cfg,device='cuda:0');vec=RslRlVecEnvWrapper(env,clip_actions=1.)
    try:
        runner=MjlabOnPolicyRunner(vec,asdict(agent()),device='cuda:0')
        runner.load(str(args.checkpoint),load_cfg={'actor':True},map_location='cuda:0');policy=runner.get_inference_policy(device='cuda:0')
        cmd=env.command_manager.get_term('twist');cmd.auto_jump=cmd.auto_height=cmd.extra_jumps=False
        desired=torch.zeros(16,3,device=env.device);desired[8:,0]=-1.3
        height=torch.tensor([.235]*4+[.28]*4+[.235]*4+[.28]*4,device=env.device)
        cmd.external_source=lambda:(desired,height)
        env._rough_force_type=0;env._rough_force_level=0;env._rough_initial_height=.24
        with torch.no_grad():
            vec.reset();robot=env.scene['robot'];pose=robot.data.root_link_pose_w.clone()
            pitch=torch.linspace(-.012,.012,4,device=env.device).repeat(4)
            pose[:,3]=torch.cos(pitch/2);pose[:,5]=torch.sin(pitch/2);robot.write_root_link_pose_to_sim(pose)
            alive=torch.ones(16,dtype=torch.bool,device=env.device);sums=torch.zeros(16,6,device=env.device)
            count=torch.zeros(16,device=env.device);angle=count.clone();peak=count.clone();previous=count.clone()
            for step in range(1500):
                if step==150:desired[8:]=0.
                _,_,done,_=vec.step(policy(vec.get_observations()));alive&=~done.bool()
                d=robot.data;pair,dx,_=task.stance_errors(env);q=d.root_link_quat_w
                yaw=torch.atan2(2*(q[:,0]*q[:,3]+q[:,1]*q[:,2]),1-2*(q[:,2].square()+q[:,3].square()))
                valid=alive&(step>=250)
                delta=yaw-previous;delta=torch.atan2(torch.sin(delta),torch.cos(delta));previous=yaw.clone()
                angle+=valid*delta;peak=torch.maximum(peak,angle.abs())
                values=torch.stack((pair.abs().mean(1),dx.abs(),d.root_link_ang_vel_b[:,2].abs(),d.root_link_lin_vel_b[:,:2].norm(dim=1),(task.local_height(env)-height).abs(),base.tilt_angle(env)),1)
                sums+=valid[:,None]*values;count+=valid
            mean=sums/count.clamp_min(1)[:,None]
            passed=alive&(count>=1200)&(mean[:,0]<.05)&(mean[:,1]<.01)&(mean[:,2]<.02)&(mean[:,3]<.02)&(mean[:,4]<.02)&(mean[:,5]<.1)&(peak<torch.deg2rad(torch.tensor(5.,device=env.device)))
            report={'checkpoint':str(args.checkpoint.resolve()),'seconds':30,'measured_after_s':5,'cases':{}}
            for i,name in enumerate(['low','high','brake_low','brake_high']):
                sl=slice(4*i,4*i+4)
                report['cases'][name]=dict(zip(['pair_rad','wheel_dx_m','yaw_abs_rad_s','speed_m_s','height_error_m','tilt_rad'],mean[sl].mean(0).tolist()))
                report['cases'][name].update(passed=bool(passed[sl].all()),pass_fraction=float(passed[sl].float().mean()),survival=float(alive[sl].float().mean()),max_heading_excursion_deg=torch.rad2deg(peak[sl]).tolist())
            report['passed']=sum(v['passed'] for v in report['cases'].values());report['gate_passed']=bool(passed.all())
            args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(report,indent=2));print('STANCE_AUDIT',json.dumps(report),flush=True)
    finally:vec.close()

if __name__=='__main__':main()
