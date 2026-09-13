"""Resume the 56-input V5 policy for moving maneuvers, retaining network/Adam state."""
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
from infantry_rl.mjlab_rugged_train import hashes as rugged_hashes,agent as rugged_agent
from infantry_rl.mjlab_rugged_audit import verify
from infantry_rl.mjlab_v5_train import FrozenTeacher
from infantry_rl import mjlab_gas_task as task


def agent(iterations=400):
    cfg=rugged_agent(iterations);cfg.algorithm.learning_rate=3e-5;cfg.algorithm.entropy_coef=.001;cfg.save_interval=50
    return cfg


def hashes():
    result=rugged_hashes()
    for name in ['mjlab_gas_task.py','mjlab_gas_train.py','mjlab_gas_audit.py','mjlab_gas_stance_audit.py']:
        path=Path(__file__).with_name(name)
        if path.exists():result[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    for name in ['mjlab_v6_refine_task.py','mjlab_v6_refine_train.py','mjlab_v6_refine_play.py','mjlab_v6_stance_audit.py','mjlab_v6_refine_audit.py']:
        path=Path(__file__).with_name(name)
        if path.exists():result[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    terrain=Path(__file__).with_name('v6_terrain.py')
    result[terrain.name]=hashlib.sha256(terrain.read_bytes()).hexdigest()
    for name in ['mjlab_motion_task.py','motion_control.py','mjlab_motion_train.py','mjlab_motion_play.py','mjlab_motion_audit.py','mjlab_jumpcarry_task.py','mjlab_jumpcarry_train.py','mjlab_jumpcarry_play.py','mjlab_jumpcarry_audit.py','mjlab_v6_jump_task.py','mjlab_v6_jump_train.py','mjlab_v6_jump_play.py','mjlab_v6_jump_audit.py']:
        path=Path(__file__).with_name(name)
        if path.exists():result[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--teacher',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--envs',type=int,default=1024);ap.add_argument('--iterations',type=int,default=400)
    ap.add_argument('--stage',choices=['stance','tuck'],default='stance')
    ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
    torch.set_num_threads(4);torch.manual_seed(46)
    meta=verify(args.source.parent);teacher_meta=verify(args.teacher.parent)
    assert meta['model_sha256']==teacher_meta['model_sha256']
    assert meta['observation_dim']==teacher_meta['observation_dim']==56
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'run.json').exists():raise FileExistsError(out)
    cfg=task.make_cfg(args.envs,stage=args.stage);cfg.seed=46;runner_cfg=agent(args.iterations)
    info=dict(task='gas_adaptation_trial',stage=args.stage,model_variant='fix_with_gas_trial',model_sha256=meta['model_sha256'],observation_dim=56,action_dim=6,
        source=str(args.source.resolve()),source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
        teacher=str(args.teacher.resolve()),teacher_sha256=hashlib.sha256(args.teacher.read_bytes()).hexdigest(),
        source_hashes=hashes(),envs=args.envs,additional_iterations=args.iterations,terrain_seed=43,
        transfer='Full actor/critic and Adam continuation; LR 3e-5; std floor .08; normalization count 100000, moments retained')
    physics=Path(__file__).parent/'robot/xmls/training_scene_fix_gas_trial.xml'
    info.update(physics_model_sha256=hashlib.sha256(physics.read_bytes()).hexdigest(),physics_model_file=physics.name,gas_force_range_N=[200,300],gas_ramp_iterations=80,hardware_force_calibrated=False)
    shutil.copy2(physics,out/physics.name)
    (out/'run.json').write_text(json.dumps(info,indent=2))
    for name in info['source_hashes']:shutil.copy2(Path(__file__).with_name(name),out/name)
    from infantry_rl import mjlab_task as base
    shutil.copy2(base.MODEL,out/base.MODEL.name)
    dump_yaml(out/'params/env.yaml',asdict(cfg));dump_yaml(out/'params/agent.yaml',asdict(runner_cfg))
    env=None;start=time.time()
    try:
        env=CheckedEnv(cfg,device='cuda:0');vec=RslRlVecEnvWrapper(env,clip_actions=1.)
        env._gas_force=torch.empty(args.envs,device=env.device).uniform_(200.,300.)
        env._gas_ramp_steps=24*80
        runner=TrackedRunner(vec,asdict(runner_cfg),str(out),device='cuda:0')
        runner.load(str(args.source),map_location='cuda:0')
        teacher=torch.load(args.teacher,map_location='cuda:0',weights_only=False)
        env._motion_teacher=FrozenTeacher(teacher['actor_state_dict'])
        runner.alg.learning_rate=3e-5
        for group in runner.alg.optimizer.param_groups:group['lr']=3e-5
        assert runner.alg.optimizer.state,'Adam state was not restored'
        with torch.no_grad():
            for module in (runner.alg.actor,runner.alg.critic):module.obs_normalizer.count.fill_(100_000)
            dist=runner.alg.actor.distribution
            if dist.std_type=='scalar':dist.std_param.clamp_(min=.08)
            else:dist.log_std_param.clamp_(min=float(torch.log(torch.tensor(.08))))
        env._motion_start_step=env.common_step_counter-(24*800 if args.smoke else 0)
        env._rough_start_step=env._motion_start_step
        env._infantry_run_start_step=env.common_step_counter
        info['curriculum_start_step']=env._motion_start_step
        (out/'run.json').write_text(json.dumps(info,indent=2))
        initial=runner.current_learning_iteration;vec.reset()
        assert vec.get_observations()['actor'].shape==(args.envs,56)
        assert (env.scene['local_ground'].data.distances>=0).all()
        print('MOTION_RESUMED',args.source,'iteration',initial,'Adam retained','envs',args.envs,flush=True)
        env.scene.write(out/'scene')
        runner.learn(num_learning_iterations=args.iterations,init_at_random_ep_len=False)
        runner.export_policy_to_onnx(str(out),'policy.onnx')
        (out/'status.json').write_text(json.dumps(dict(state='complete',additional_iterations=args.iterations,
            elapsed_seconds=time.time()-start,transitions=(env.common_step_counter-env._infantry_run_start_step)*args.envs,
            final_checkpoint=f'model_{initial+args.iterations-1}.pt'),indent=2))
    except BaseException as exc:
        (out/'status.json').write_text(json.dumps(dict(state='failed',error=repr(exc)),indent=2));raise
    finally:
        if env is not None:env.close()


if __name__=='__main__':main()
