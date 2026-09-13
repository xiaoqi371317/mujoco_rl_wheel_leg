"""Train native mjlab + MuJoCo-Warp + RSL-RL PPO inside conda mujoco_rl."""
import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import time
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.utils.os import dump_yaml
from infantry_rl.mjlab_task import make_cfg, runner_cfg, MODEL


class CheckedEnv(ManagerBasedRlEnv):
    """Stop a run rather than letting reset hide a numerical failure."""
    def step(self, action):
        result = super().step(action)
        if self.termination_manager.get_term("nan").any().item():
            raise FloatingPointError("MuJoCo-Warp produced a non-finite state")
        if not torch.isfinite(result[1]).all().item():
            raise FloatingPointError("Non-finite reward")
        return result


class TrackedRunner(MjlabOnPolicyRunner):
    def save(self, path, infos=None):
        pending = Path(path).with_suffix(".pt.tmp")
        super().save(str(pending), infos)
        pending.replace(path)
        folder = Path(path).parent
        record = dict(state="running", framework="mjlab", device=str(self.device),
                      iteration=int(self.current_learning_iteration), checkpoint=Path(path).name,
                      env_steps=int(self.env.unwrapped.common_step_counter),
                      transitions=int((self.env.unwrapped.common_step_counter - getattr(self.env.unwrapped,'_infantry_run_start_step',0)) * self.env.num_envs))
        (folder / "status.json").write_text(json.dumps(record, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--iterations", type=int, default=1000)
    ap.add_argument("--task", choices=["balance", "velocity", "cmd_vel", "cmd_vel_v3", "cmd_vel_v4", "cmd_vel_v4_fast"], default="cmd_vel")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--resume", type=Path)
    ap.add_argument("--warm-start", type=Path, help='Load compatible actor/critic weights with fresh optimizer and curriculum')
    ap.add_argument("--model", choices=['original','fix'], default='original')
    ap.add_argument('--refine-stop',action='store_true',help='Continue fast policy with low-stance stop regularization')
    args = ap.parse_args()
    from infantry_rl.model_selection import select_model
    model_path = select_model(args.model)
    if args.refine_stop and (args.task!='cmd_vel_v4_fast' or not args.resume):
        ap.error('--refine-stop requires --task cmd_vel_v4_fast and --resume')
    if args.resume and args.warm_start:
        ap.error('--resume and --warm-start are mutually exclusive')
    if args.task == 'cmd_vel_v4_fast' and args.model != 'fix':
        ap.error('cmd_vel_v4_fast requires --model fix')
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required: run with the GPU-enabled mujoco_rl conda environment")
    torch.set_num_threads(4)
    torch.manual_seed(42)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "run.json").exists():
        raise FileExistsError("Use a new run directory")
    task_factory = make_cfg
    if args.task == 'cmd_vel':
        from infantry_rl.mjlab_cmdvel_task import make_cfg as task_factory
    elif args.task == 'cmd_vel_v3':
        from infantry_rl.mjlab_cmdvel_v3_task import make_cfg as task_factory
    elif args.task == "cmd_vel_v4":
        from infantry_rl.mjlab_cmdvel_v4_task import make_cfg as task_factory
    elif args.task == "cmd_vel_v4_fast":
        from infantry_rl.mjlab_cmdvel_v4_fast_task import make_cfg as task_factory
    cfg = task_factory(args.envs, args.task)
    if args.refine_stop:
        from infantry_rl.stop_refinement import apply
        cfg=apply(cfg)
    cfg.seed = 42
    agent = runner_cfg(args.iterations)
    if args.refine_stop:
        agent.algorithm.learning_rate=5e-5
        agent.algorithm.schedule='fixed'
        agent.algorithm.entropy_coef=.001
    metadata = dict(framework="mjlab", task=args.task, python=sys.executable,
                    device="cuda:0", gpu=torch.cuda.get_device_name(0), envs=args.envs,
                    iterations=args.iterations, observation_dim=48 if args.task in ("cmd_vel_v4", "cmd_vel_v4_fast") else 47, action_dim=6,
                    model_variant=args.model, model_file=model_path.name,
                    model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
                    model_selection_sha256=hashlib.sha256(Path(__file__).with_name('model_selection.py').read_bytes()).hexdigest(),
                    task_sha256=hashlib.sha256(Path(__file__).with_name("mjlab_task.py").read_bytes()).hexdigest(),
                    versions={n: importlib.metadata.version(n) for n in
                        ["mjlab", "mujoco", "mujoco-warp", "warp-lang", "torch", "rsl-rl-lib"]})
    if args.task in ('cmd_vel', 'cmd_vel_v3', 'cmd_vel_v4', 'cmd_vel_v4_fast'):
        metadata['cmdvel_sha256'] = hashlib.sha256(Path(__file__).with_name('mjlab_cmdvel_task.py').read_bytes()).hexdigest()
    if args.task in ('cmd_vel_v3', 'cmd_vel_v4', 'cmd_vel_v4_fast'):
        metadata['cmdvel_v3_sha256'] = hashlib.sha256(Path(__file__).with_name('mjlab_cmdvel_v3_task.py').read_bytes()).hexdigest()
    if args.task in ("cmd_vel_v4", "cmd_vel_v4_fast"):
        metadata["cmdvel_v4_sha256"] = hashlib.sha256(Path(__file__).with_name("mjlab_cmdvel_v4_task.py").read_bytes()).hexdigest()
    if args.task == 'cmd_vel_v4_fast':
        metadata['cmdvel_v4_fast_sha256']=hashlib.sha256(Path(__file__).with_name('mjlab_cmdvel_v4_fast_task.py').read_bytes()).hexdigest()
        metadata['forward_convention']='user forward = model -x'
    if args.refine_stop:
        metadata['refinement']='stop'
        metadata['stop_refinement_sha256']=hashlib.sha256(Path(__file__).with_name('stop_refinement.py').read_bytes()).hexdigest()
    if args.resume:
        previous=json.loads((args.resume.parent/'run.json').read_text())
        for key in ['model_sha256','observation_dim','action_dim','task_sha256','cmdvel_v4_fast_sha256']:
            if previous.get(key)!=metadata.get(key):raise ValueError('Incompatible resume: '+key)
        metadata['resume']=str(args.resume.resolve())
        metadata['resume_sha256']=hashlib.sha256(args.resume.read_bytes()).hexdigest()
    if args.warm_start:
        source_meta=json.loads((args.warm_start.parent/'run.json').read_text())
        for key in ['model_sha256','observation_dim','action_dim','task_sha256','cmdvel_v4_sha256']:
            if source_meta.get(key)!=metadata.get(key):raise ValueError('Incompatible warm-start: '+key)
        if source_meta['task'] not in ('cmd_vel_v4','cmd_vel_v4_fast'):raise ValueError('Unsupported warm-start task')
        metadata['warm_start']=str(args.warm_start.resolve())
        metadata['warm_start_sha256']=hashlib.sha256(args.warm_start.read_bytes()).hexdigest()
    (out / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    shutil.copy2(model_path, out / model_path.name)
    for name in ("mjlab_task.py", "mjlab_cmdvel_task.py", "mjlab_cmdvel_v3_task.py", "mjlab_cmdvel_v4_task.py", "mjlab_cmdvel_v4_fast_task.py", "stop_refinement.py", "model_selection.py", "v4_controls.py", "mjlab_v4_audit.py", "mjlab_fast_audit.py", "fast_control.py", "v4_keyboard.py", "mjlab_train.py", "mjlab_play.py"):
        source = Path(__file__).with_name(name)
        if source.exists():
            shutil.copy2(source, out / name)
    dump_yaml(out / "params/env.yaml", asdict(cfg))
    dump_yaml(out / "params/agent.yaml", asdict(agent))
    print("RUN " + json.dumps(metadata), flush=True)
    start = time.time()
    env = None
    try:
        env = CheckedEnv(cfg, device="cuda:0")
        env.scene.write(out / "scene")
        vec = RslRlVecEnvWrapper(env, clip_actions=1.)
        obs = vec.get_observations()
        assert obs["actor"].shape == (args.envs, metadata["observation_dim"]), obs["actor"].shape
        assert vec.num_actions == 6
        for key, term in env.action_manager._terms.items():
            print("ACTION_ORDER", key, term.target_names, flush=True)
        runner = TrackedRunner(vec, asdict(agent), str(out), device="cuda:0")
        if args.resume:
            runner.load(str(args.resume))
            if args.refine_stop:
                runner.alg.learning_rate=5e-5
                for group in runner.alg.optimizer.param_groups:group['lr']=5e-5
                vec.reset()
            print('RESUMED',args.resume,'iteration',runner.current_learning_iteration,'optimizer state retained',flush=True)
        elif args.warm_start:
            runner.load(str(args.warm_start),load_cfg={'actor':True,'critic':True,'optimizer':False,'iteration':False},map_location='cuda:0')
            runner.current_learning_iteration=0
            env.common_step_counter=0
            vec.reset()
            print('WARM_START: compatible weights loaded; optimizer, iteration and curriculum reset',flush=True)
        env._infantry_run_start_step=env.common_step_counter
        runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=False)
        runner.export_policy_to_onnx(str(out), "policy.onnx")
        (out / "status.json").write_text(json.dumps(dict(state="complete", framework="mjlab",
            device="cuda:0", iterations=args.iterations, elapsed_seconds=time.time() - start,
            transitions=(env.common_step_counter-env._infantry_run_start_step) * env.num_envs), indent=2), encoding="utf-8")
    except BaseException as exc:
        (out / "status.json").write_text(json.dumps(dict(state="failed", error=repr(exc)), indent=2), encoding="utf-8")
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
