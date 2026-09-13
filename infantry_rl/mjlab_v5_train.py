"""V4 weight transfer to V5, with padded phase inputs and auditable snapshots."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import time
import torch
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.utils.os import dump_yaml
from infantry_rl.model_selection import select_model
from infantry_rl.mjlab_train import CheckedEnv,TrackedRunner
from infantry_rl import mjlab_task as base
from infantry_rl import mjlab_v5_task as task

FILES=['mjlab_task.py','mjlab_cmdvel_task.py','mjlab_cmdvel_v3_task.py','mjlab_cmdvel_v4_task.py',
       'mjlab_cmdvel_v4_fast_task.py','mjlab_v5_task.py','v5_poses.json','mjlab_v5_train.py','mjlab_v5_audit.py',
       'mjlab_v5_play.py','model_selection.py']

def hashes():
    return {name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in FILES if Path(__file__).with_name(name).exists()}

def agent_cfg(iterations=1200):
    cfg=base.runner_cfg(iterations)
    cfg.algorithm.learning_rate=1e-4;cfg.algorithm.schedule='fixed';cfg.algorithm.entropy_coef=.004
    cfg.actor.distribution_cfg['init_std']=.15
    return cfg

class FrozenTeacher:
    def __init__(self,state):self.state=state
    def __call__(self,obs):
        s=self.state;x=(obs-s['obs_normalizer._mean'])/(s['obs_normalizer._std']+.01)
        for i in (0,2,4):
            x=torch.nn.functional.linear(x,s[f'mlp.{i}.weight'],s[f'mlp.{i}.bias'])
            if i<4:x=torch.nn.functional.elu(x)
        return x

def transfer(runner,path):
    saved=torch.load(path,map_location='cuda:0',weights_only=False)
    for label,module in [('actor',runner.alg.actor),('critic',runner.alg.critic)]:
        old=saved[label+'_state_dict'];new=module.state_dict()
        for key,value in old.items():
            if key.startswith('distribution.'):continue # Reopen exploration for the new maneuvers.
            if new[key].shape==value.shape:new[key]=value.clone()
            elif key=='mlp.0.weight':
                assert value.shape[1]==48 and new[key].shape[1]==task.OBS_DIM
                new[key].zero_();new[key][:,:48]=value
            elif key.startswith('obs_normalizer.'):
                new[key][:,:48]=value
            else:raise ValueError('Unrecognized transfer shape: '+key)
        # Preserve normalization moments but allow the changed state distribution to adapt.
        new['obs_normalizer.count'].fill_(100_000)
        module.load_state_dict(new)
    teacher=FrozenTeacher(saved['actor_state_dict'])
    # Zero padded inputs must preserve the old mean action before any training.
    x=torch.randn(16,48,device='cuda:0')*.01+saved['actor_state_dict']['obs_normalizer._mean']
    padded=torch.cat((x,torch.zeros(16,task.OBS_DIM-48,device='cuda:0')),1)
    actor=runner.alg.actor
    with torch.no_grad():pred=actor.mlp(actor.obs_normalizer(padded));error=(pred-teacher(x)).abs().max().item()
    assert error<1e-5,error
    print('V4_TRANSFER_MAX_ACTION_ERROR',error,flush=True)
    return teacher

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--source',type=Path,required=True);ap.add_argument('--resume',type=Path)
    ap.add_argument('--envs',type=int,default=1024);ap.add_argument('--iterations',type=int,default=1200)
    ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
    torch.set_num_threads(4);torch.manual_seed(42);model=select_model('fix')
    source_meta=json.loads((args.source.parent/'run.json').read_text())
    assert source_meta['task']=='cmd_vel_v4_fast' and source_meta['observation_dim']==48
    assert source_meta['model_sha256']==hashlib.sha256(model.read_bytes()).hexdigest()
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'run.json').exists():raise FileExistsError(out)
    cfg=task.make_cfg(args.envs);cfg.seed=42
    agent=agent_cfg(args.iterations)
    metadata=dict(task='v5',model_variant='fix',model_sha256=source_meta['model_sha256'],
                  observation_dim=task.OBS_DIM,action_dim=6,envs=args.envs,iterations=args.iterations,
                  source=str(args.source.resolve()),source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
                  source_hashes=hashes(),transfer='V4 actor and critic; 8 zero-initialized phase inputs; fresh Adam',
                  resume=str(args.resume) if args.resume else None)
    (out/'run.json').write_text(json.dumps(metadata,indent=2))
    for name in metadata['source_hashes']:shutil.copy2(Path(__file__).with_name(name),out/name)
    shutil.copy2(model,out/model.name)
    dump_yaml(out/'params/env.yaml',asdict(cfg));dump_yaml(out/'params/agent.yaml',asdict(agent))
    env=None;t=time.time()
    try:
        env=CheckedEnv(cfg,device='cuda:0');vec=RslRlVecEnvWrapper(env,clip_actions=1.)
        runner=TrackedRunner(vec,asdict(agent),str(out),device='cuda:0')
        env._v5_teacher=transfer(runner,args.source)
        if args.resume:
            prior=json.loads((args.resume.parent/'run.json').read_text())
            assert prior['source_hashes']['mjlab_v5_task.py']==metadata['source_hashes']['mjlab_v5_task.py']
            runner.load(str(args.resume))
        if args.smoke:env.common_step_counter=24*900
        vec.reset();assert vec.get_observations()['actor'].shape==(args.envs,task.OBS_DIM)
        env._infantry_run_start_step=env.common_step_counter
        runner.learn(num_learning_iterations=args.iterations,init_at_random_ep_len=False)
        runner.export_policy_to_onnx(str(out),'policy.onnx')
        (out/'status.json').write_text(json.dumps(dict(state='complete',iterations=args.iterations,
            elapsed_seconds=time.time()-t,transitions=(env.common_step_counter-env._infantry_run_start_step)*args.envs),indent=2))
    except BaseException as exc:
        (out/'status.json').write_text(json.dumps(dict(state='failed',error=repr(exc)),indent=2));raise
    finally:
        if env is not None:env.close()

if __name__=='__main__':main()
