"""Fast-policy acceptance: corrected forward, 1.3 m/s, braking and fast lift."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner,RslRlVecEnvWrapper
from infantry_rl import mjlab_task as base
from infantry_rl import mjlab_cmdvel_v4_fast_task as task
from infantry_rl.model_selection import select_model

# name, model vx, yaw, initial height, switch second, final vx/yaw/height, settle limit.
CASES=[
    ('stop',0.,0.,.235,None,None,None),
    ('forward_0_3',-.3,0.,.235,None,None,None),
    ('forward_0_8',-.8,0.,.235,None,None,None),
    ('forward_1_3',-1.3,0.,.235,None,None,None),
    ('reverse_0_5',.5,0.,.235,None,None,None),
    ('left_arc',-.5,1.2,.235,None,None,None),
    ('right_arc',-.5,-1.2,.235,None,None,None),
    ('spin_left',0.,3.,.235,None,None,None),
    ('spin_right',0.,-3.,.235,None,None,None),
    ('high_stop',0.,0.,.28,None,None,None),
    ('fast_raise',0.,0.,.235,2.,(0.,0.,.28),1.),
    ('fast_lower',0.,0.,.28,5.,(0.,0.,.235),1.),
    ('fast_drive_then_stop',-1.3,0.,.235,5.,(0.,0.,.235),1.5),
    ('turn_release',-.5,1.2,.235,5.,(-.5,0.,.235),1.5),
    ('spin_then_stop',0.,3.,.235,5.,(0.,0.,.235),1.5),
]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--seconds',type=float,default=10.)
    ap.add_argument('--only-stop',action='store_true')
    args=ap.parse_args()
    if args.seconds<10.:ap.error('--seconds must be at least 10')
    cases=[row for row in CASES if row[0]=='stop'] if args.only_stop else CASES
    torch.set_num_threads(4)
    meta=json.loads((args.run/'run.json').read_text())
    assert meta['task']=='cmd_vel_v4_fast' and meta['model_variant']=='fix'
    model=select_model('fix')
    assert hashlib.sha256(model.read_bytes()).hexdigest()==meta['model_sha256']
    for key,name in [('task_sha256','mjlab_task.py'),('model_selection_sha256','model_selection.py'),
        ('cmdvel_sha256','mjlab_cmdvel_task.py'),('cmdvel_v3_sha256','mjlab_cmdvel_v3_task.py'),
        ('cmdvel_v4_sha256','mjlab_cmdvel_v4_task.py'),('cmdvel_v4_fast_sha256','mjlab_cmdvel_v4_fast_task.py')]:
        assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()==meta[key],name
    repeats=4;n=len(cases)*repeats
    from infantry_rl.stop_refinement import apply_from_metadata
    cfg=apply_from_metadata(task.make_cfg(n),meta);cfg.seed=2026
    cfg.episode_length_s=args.seconds
    env=ManagerBasedRlEnv(cfg,device='cuda:0');vec=RslRlVecEnvWrapper(env,clip_actions=1.)
    try:
        runner=MjlabOnPolicyRunner(vec,asdict(base.runner_cfg()),device='cuda:0')
        runner.load(str(args.checkpoint),load_cfg={'actor':True},map_location='cuda:0')
        policy=runner.get_inference_policy(device='cuda:0')
        cmd=env.command_manager.get_term('twist')
        desired=torch.tensor([[x,0.,z] for _,x,z,_,_,_,_ in cases],device='cuda:0').repeat_interleave(repeats,0)
        heights=torch.tensor([h for _,_,_,h,_,_,_ in cases],device='cuda:0').repeat_interleave(repeats)
        def fixed(dt):
            cmd.vel_command_b.copy_(desired);cmd.height_target.copy_(heights)
            cmd.height_command+=(heights-cmd.height_command).clamp(-task.HEIGHT_RATE*dt,task.HEIGHT_RATE*dt)
        cmd.compute=fixed
        vec.reset()
        alive=torch.ones(n,dtype=torch.bool,device='cuda:0')
        counts=torch.zeros(n,device='cuda:0');sums=torch.zeros(n,9,device='cuda:0')
        minimum=torch.full((n,),float('inf'),device='cuda:0')
        maximum=torch.full((n,),float('-inf'),device='cuda:0')
        streak=torch.zeros(n,device='cuda:0');settled=torch.full((n,),float('nan'),device='cuda:0')
        with torch.inference_mode():
            for step in range(round(args.seconds/env.step_dt)):
                t=step*env.step_dt
                for i,(_,_,_,_,switch,final,_) in enumerate(cases):
                    if switch is not None and step==round(switch/env.step_dt):
                        g=slice(i*repeats,(i+1)*repeats)
                        desired[g,0]=final[0];desired[g,2]=final[1];heights[g]=final[2]
                fixed(0.)
                d=env.scene['robot'].data
                vx=d.root_link_lin_vel_b[:,0];yaw=d.root_link_ang_vel_b[:,2]
                height=base.height(env);tilt=base.tilt_angle(env)
                ex=(vx-desired[:,0]).abs();ez=(yaw-desired[:,2]).abs()
                eh=(height-heights).abs() # Final requested height, not the ramped reference.
                good=(ex<.1)&(ez<.2)&(eh<.015)&(tilt<.2)
                active=alive & (t>=2.)
                for i,(name,_,_,_,switch,_,limit) in enumerate(cases):
                    if switch is None:continue
                    g=slice(i*repeats,(i+1)*repeats)
                    active[g] &= t>=switch+limit
                    if t>=switch:
                        ready=good[g]&alive[g]
                        if name in ('fast_raise','fast_lower'):ready &= eh[g]<.01
                        streak[g]=torch.where(ready,streak[g]+1,0.)
                        reached=(streak[g]>=10)&settled[g].isnan()
                        # Report completion of a continuous 0.2 s settled window.
                        settled[g]=torch.where(reached,torch.full_like(settled[g],t-switch),settled[g])
                sums+=active[:,None]*torch.stack((-vx,yaw,ex,ez,eh,torch.rad2deg(tilt),good.float(),height,vx.square()),1)
                minimum=torch.where(active,torch.minimum(minimum,-vx),minimum)
                maximum=torch.where(active,torch.maximum(maximum,-vx),maximum)
                counts+=active
                obs=env.observation_manager.compute(update_history=True)
                vec.step(policy(obs));alive &= ~env.reset_terminated
        values=sums/counts.clamp_min(1)[:,None]
        report={'checkpoint':str(args.checkpoint),'seed':2026,'seconds':args.seconds,'repeats':repeats,
            'forward_convention':'user forward = model -x','height_ramp_m_s':task.HEIGHT_RATE,
            'excluded':['high_forward','high_spin'],'cases':{}}
        names=['forward_speed_m_s','yaw_rad_s','speed_mae','yaw_mae','height_mae_m','tilt_deg','tracking_fraction','height_m','speed_mean_square']
        for i,(name,x,z,h,switch,final,limit) in enumerate(cases):
            g=slice(i*repeats,(i+1)*repeats);valid=counts[g]>0
            means=values[g][valid].mean(0).tolist() if valid.any() else [None]*len(names)
            row=dict(zip(names,means));row['survival_fraction']=alive[g].float().mean().item()
            row['scored_steps']=counts[g].tolist()
            row['speed_rms']=values[g,8][valid].sqrt().mean().item() if valid.any() else None
            row['speed_std']=(values[g,8][valid]-values[g,0][valid].square()).clamp_min(0).sqrt().mean().item() if valid.any() else None
            row['speed_peak_to_peak']=(maximum[g][valid]-minimum[g][valid]).mean().item() if valid.any() else None
            row['passed']=bool(alive[g].all() and valid.all() and (values[g,2]<.1).all()
                and (values[g,3]<.2).all() and (values[g,4]<.015).all() and (values[g,6]>.8).all())
            row['target_initial']={'forward_speed':-x,'yaw':z,'height':h}
            if switch is not None:
                row['switch_at_s']=switch;row['settle_limit_s']=limit
                row['settle_seconds']=[v if v==v else None for v in settled[g].tolist()]
                row['passed'] &= bool((settled[g]<=limit+1e-6).all())
                row['target_final']={'forward_speed':-final[0],'yaw':final[1],'height':final[2]}
            report['cases'][name]=row
        report['all_passed']=all(v['passed'] for v in report['cases'].values())
        suffix='_stop_audit.json' if args.only_stop else '_fast_audit.json'
        path=args.run/(args.checkpoint.stem+suffix)
        path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
        print('FAST_AUDIT '+json.dumps(report,allow_nan=False),flush=True)
    finally:vec.close()


if __name__=='__main__':main()
