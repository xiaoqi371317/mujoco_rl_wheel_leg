"""Continue a V5 checkpoint on stronger rough ground, retaining Adam state."""
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
from infantry_rl.mjlab_train import CheckedEnv,TrackedRunner
from infantry_rl.mjlab_v5_train import agent_cfg,FrozenTeacher,hashes as v5_hashes
from infantry_rl.mjlab_v5_audit import verify as verify_v5
from infantry_rl import mjlab_rugged_task as task
from infantry_rl.mjlab_rough_train import hashes as rough_hashes
from infantry_rl.mjlab_rough_audit import verify as verify_rough

def agent(iterations=600):
    cfg=agent_cfg(iterations);cfg.algorithm.learning_rate=5e-5
    cfg.algorithm.entropy_coef=.0015
    return cfg

def hashes():
    h=rough_hashes()
    for name in ['rugged_terrain.py','mjlab_rugged_task.py','mjlab_rugged_train.py','mjlab_rugged_audit.py','mjlab_rugged_play.py']:
        p=Path(__file__).with_name(name)
        if p.exists():h[name]=hashlib.sha256(p.read_bytes()).hexdigest()
    return h

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--envs',type=int,default=1024);ap.add_argument('--iterations',type=int,default=600)
    ap.add_argument('--resume',type=Path)
    ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
    torch.set_num_threads(4);torch.manual_seed(44);meta=verify_v5(args.source.parent)
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'run.json').exists():raise FileExistsError(out)
    cfg=task.make_cfg(args.envs);cfg.seed=44;runner_cfg=agent(args.iterations)
    info=dict(task='v5_rugged',model_variant='fix',model_sha256=meta['model_sha256'],observation_dim=56,action_dim=6,
        source=str(args.source.resolve()),source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
        source_hashes=hashes(),envs=args.envs,additional_iterations=args.iterations,terrain_seed=43,
        transfer='Full V5 actor, critic, observation normalization and Adam state; LR overridden to 5e-5',
        resume=str(args.resume.resolve()) if args.resume else None)
    if args.resume:
        previous=verify_rough(args.resume.parent)
        for key in ['model_sha256','observation_dim','action_dim','source_sha256']:
            assert previous[key]==info[key],key
        info['resume_sha256']=hashlib.sha256(args.resume.read_bytes()).hexdigest()
    (out/'run.json').write_text(json.dumps(info,indent=2))
    for name in info['source_hashes']:shutil.copy2(Path(__file__).with_name(name),out/name)
    from infantry_rl import mjlab_task as base
    shutil.copy2(base.MODEL,out/base.MODEL.name)
    dump_yaml(out/'params/env.yaml',asdict(cfg));dump_yaml(out/'params/agent.yaml',asdict(runner_cfg))
    env=None;t=time.time()
    try:
        env=CheckedEnv(cfg,device='cuda:0');vec=RslRlVecEnvWrapper(env,clip_actions=1.)
        runner=TrackedRunner(vec,asdict(runner_cfg),str(out),device='cuda:0')
        runner.load(str(args.resume or args.source),map_location='cuda:0')
        saved=torch.load(args.source,map_location='cuda:0',weights_only=False)
        env._rough_teacher=FrozenTeacher(saved['actor_state_dict'])
        runner.alg.learning_rate=5e-5
        for group in runner.alg.optimizer.param_groups:group['lr']=5e-5
        env._rough_start_step=env.common_step_counter-(24*500 if args.smoke else 0)
        info['curriculum_start_step']=env._rough_start_step
        (out/'run.json').write_text(json.dumps(info,indent=2))
        env._infantry_run_start_step=env.common_step_counter
        initial=runner.current_learning_iteration
        vec.reset()
        assert vec.get_observations()['actor'].shape==(args.envs,56)
        hits=task.metric(env,'ground_hit_fraction').mean().item();assert hits==1.,hits
        assert float((task.local_height(env)-env._v5_initial_height).abs().max())<.002
        print('RUGGED_RESUMED',args.resume or args.source,'iteration',initial,'Adam retained','terrain_hits',hits,flush=True)
        env.scene.write(out/'scene')
        runner.learn(num_learning_iterations=args.iterations,init_at_random_ep_len=False)
        runner.export_policy_to_onnx(str(out),'policy.onnx')
        (out/'status.json').write_text(json.dumps(dict(state='complete',additional_iterations=args.iterations,
            elapsed_seconds=time.time()-t,transitions=(env.common_step_counter-env._infantry_run_start_step)*args.envs,
            final_checkpoint=f'model_{initial+args.iterations-1}.pt'),indent=2))
    except BaseException as exc:
        (out/'status.json').write_text(json.dumps(dict(state='failed',error=repr(exc)),indent=2));raise
    finally:
        if env is not None:env.close()

if __name__=='__main__':main()
