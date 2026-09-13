"""Measure actual two-wheel clearance, uninterrupted airtime and stable landing."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import torch
from mjlab.rl import MjlabOnPolicyRunner,RslRlVecEnvWrapper
from infantry_rl.model_selection import select_model
from infantry_rl.mjlab_train import CheckedEnv
from infantry_rl.mjlab_v5_train import agent_cfg
from infantry_rl import mjlab_v5_task as task
from infantry_rl import mjlab_task as base

CASES=[('stop',.24,False,0.,0.),('rise_20cm',.20,False,0.,0.),
       ('rise_18cm',.18,False,0.,0.),('rise_16cm',.16,False,0.,0.),
       ('jump',.24,True,0.,0.),('rise_then_jump',.16,True,0.,0.),
       ('forward_1_3',.24,False,-1.3,0.),('spin_left',.24,False,0.,3.),
       ('spin_right',.24,False,0.,-3.)]

def verify(run):
    meta=json.loads((run/'run.json').read_text());model=select_model('fix')
    assert meta['model_sha256']==hashlib.sha256(model.read_bytes()).hexdigest()
    for name,digest in meta['source_hashes'].items():
        # Execution wrapper improvements do not alter a checkpoint's dynamics.
        if name in ('mjlab_v5_train.py','mjlab_v5_audit.py','mjlab_v5_play.py'):continue
        assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()==digest,name
    return meta

def latest(run):return max(run.glob('model_*.pt'),key=lambda p:int(p.stem.split('_')[1]))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--checkpoint',type=Path);ap.add_argument('--repeats',type=int,default=8)
    args=ap.parse_args();verify(args.run);torch.set_num_threads(4)
    checkpoint=args.checkpoint or latest(args.run);reps=args.repeats;n=len(CASES)*reps
    cfg=task.make_cfg(n,play=True);cfg.seed=2027;cfg.episode_length_s=10.
    env=CheckedEnv(cfg,device='cuda:0');vec=RslRlVecEnvWrapper(env,clip_actions=1.)
    try:
        runner=MjlabOnPolicyRunner(vec,asdict(agent_cfg()),device='cuda:0')
        runner.load(str(checkpoint),load_cfg={'actor':True},map_location='cuda:0')
        policy=runner.get_inference_policy(device='cuda:0');cmd=env.command_manager.get_term('twist')
        env._v5_reset_height=torch.tensor([c[1] for c in CASES],device=env.device).repeat_interleave(reps)
        jump=torch.tensor([c[2] for c in CASES],device=env.device).repeat_interleave(reps)
        desired=torch.tensor([[c[3],0.,c[4]] for c in CASES],device=env.device).repeat_interleave(reps,0)
        original=cmd.compute
        def controlled(dt):
            original(dt);idle=cmd.phase==0
            cmd.vel_command_b[idle]=desired[idle];cmd.height_target[:]=.235;cmd.height_command[idle]=.235
        cmd.compute=controlled;cmd.auto_jump=False;vec.reset()
        # Slight common-body pitch perturbations preserve all loop closures.
        state=env.scene['robot'].data.root_link_pose_w.clone()
        pitch=torch.linspace(-.015,.015,reps,device=env.device).repeat(len(CASES))
        state[:,3]=torch.cos(pitch/2);state[:,5]=torch.sin(pitch/2)
        env.scene['robot'].write_root_link_pose_to_sim(state)
        alive=torch.ones(n,dtype=torch.bool,device=env.device)
        peak=torch.zeros(n,device=env.device);air_run=peak.clone();longest=peak.clone()
        final_good=peak.clone();final_count=peak.clone();speed_sum=peak.clone();speed_error=peak.clone()
        rise_good=peak.clone();rise_count=peak.clone();drift=peak.clone()
        origin=env.scene['robot'].data.root_link_pos_w[:,:2].clone()
        with torch.inference_mode():
            for step in range(450):
                if step==150:cmd.begin_jump(jump.nonzero().flatten())
                controlled(0.)
                obs=vec.get_observations();obs,_,done,_=vec.step(policy(obs))
                alive&=~done.bool();d=env.scene['robot'].data
                speed=d.root_link_lin_vel_b[:,0]
                good=(base.tilt_angle(env)<.2)&((base.height(env)-.235).abs()<.015)&((speed-desired[:,0]).abs()<.1)&((d.root_link_ang_vel_b[:,2]-desired[:,2]).abs()<.2)&alive
                if 100<=step<145:
                    rise_good+=good;rise_count+=1
                if step>=150:
                    clearance=task.wheel_clearance(env)
                    air=(base.wheel_support(env).sum(1)==0)&(clearance>.005)&alive
                    air_run=torch.where(air,air_run+env.step_dt,0.)
                    longest=torch.maximum(longest,air_run)
                    peak=torch.maximum(peak,torch.where(air,clearance,0.))
                if step>=350:
                    final_good+=good;final_count+=1;speed_sum+=speed*alive
                    speed_error+=(speed-desired[:,0]).abs()*alive
                drift=torch.maximum(drift,torch.linalg.vector_norm(d.root_link_pos_w[:,:2]-origin,dim=1)*alive)
        report={'checkpoint':str(checkpoint),'seed':2027,'repeats':reps,'seconds':9.,'cases':{}}
        for k,(name,h,j,x,z) in enumerate(CASES):
            sl=slice(k*reps,(k+1)*reps)
            settled=(final_good/final_count)[sl];recovered=(rise_good/rise_count)[sl]
            passed=alive[sl]&(settled>.9)
            if h<.23:passed&=recovered>.8
            if j:passed&=(peak[sl]>=.02)&(longest[sl]>=.06)&(drift[sl]<.30)
            report['cases'][name]=dict(passed=bool(passed.all()),success_fraction=float(passed.float().mean()),
                survival_fraction=float(alive[sl].float().mean()),peak_two_wheel_clearance_m=peak[sl].tolist(),
                longest_airtime_s=longest[sl].tolist(),final_tracking_fraction=settled.tolist(),
                recovery_tracking_fraction=recovered.tolist(),max_xy_displacement_m=drift[sl].tolist(),
                final_forward_speed_m_s=float((-speed_sum/final_count)[sl].mean()),
                final_speed_mae=float((speed_error/final_count)[sl].mean()))
        report['all_passed']=all(c['passed'] for c in report['cases'].values())
        (args.run/(checkpoint.stem+'_v5_audit.json')).write_text(json.dumps(report,indent=2))
        print('V5_AUDIT',json.dumps(report),flush=True)
    finally:vec.close()

if __name__=='__main__':main()
