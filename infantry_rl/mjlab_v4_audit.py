"""Fixed-command acceptance tests, including stop after spin. Never equate survival with tracking."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from infantry_rl import mjlab_task as base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--wait', action='store_true', help='Wait up to two hours for this checkpoint before allocating GPU resources')
    args = ap.parse_args()
    if args.wait:
        deadline = time.monotonic() + 7200
        while not args.checkpoint.exists():
            status_path = args.run / 'status.json'
            if status_path.exists():
                try:
                    state = json.loads(status_path.read_text()).get('state')
                except json.JSONDecodeError:
                    state = None  # A status write may be in progress.
                if state in ['failed', 'stopped_by_user', 'complete']:
                    raise RuntimeError(f'Training ended ({state}) without requested checkpoint')
            if time.monotonic() > deadline:
                raise TimeoutError('Checkpoint wait exceeded two hours')
            time.sleep(5)
    torch.set_num_threads(4)
    meta = json.loads((args.run / 'run.json').read_text())
    from infantry_rl.model_selection import select_model
    select_model(meta.get('model_variant','original'))
    if 'model_selection_sha256' in meta:
        assert hashlib.sha256(Path(__file__).with_name('model_selection.py').read_bytes()).hexdigest() == meta['model_selection_sha256']
    assert hashlib.sha256(base.MODEL.read_bytes()).hexdigest() == meta['model_sha256']
    assert hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest() == meta['task_sha256']
    factory = base.make_cfg
    if meta['task'] == 'cmd_vel':
        from infantry_rl import mjlab_cmdvel_task as task
        assert hashlib.sha256(Path(task.__file__).read_bytes()).hexdigest() == meta['cmdvel_sha256']
        factory = task.make_cfg
    elif meta['task'] == 'cmd_vel_v3':
        from infantry_rl import mjlab_cmdvel_task as v2_task
        from infantry_rl import mjlab_cmdvel_v3_task as task
        assert hashlib.sha256(Path(v2_task.__file__).read_bytes()).hexdigest() == meta['cmdvel_sha256']
        assert hashlib.sha256(Path(task.__file__).read_bytes()).hexdigest() == meta['cmdvel_v3_sha256']
        factory = task.make_cfg
    from infantry_rl import mjlab_cmdvel_v4_task as task
    assert meta['task'] == 'cmd_vel_v4'
    for key, filename in [('cmdvel_sha256','mjlab_cmdvel_task.py'),('cmdvel_v3_sha256','mjlab_cmdvel_v3_task.py'),('cmdvel_v4_sha256','mjlab_cmdvel_v4_task.py')]:
        assert hashlib.sha256(Path(__file__).with_name(filename).read_bytes()).hexdigest() == meta[key]
    factory = task.make_cfg
    cases = [('stop',0.,0.,.235),('forward',.3,0.,.235),('reverse',-.2,0.,.235),
             ('left_arc',.3,1.2,.235),('right_arc',.3,-1.2,.235),
             ('spin_left',0.,3.,.235),('spin_right',0.,-3.,.235),
             ('high_stop',0.,0.,.28),('high_forward',.3,0.,.28),('high_spin',0.,3.,.28),
             ('raise_then_lower',0.,0.,.28),('turn_release',.3,1.2,.235),('spin_then_stop',0.,3.,.235)]
    repeats = 4
    cfg = factory(len(cases) * repeats, meta['task'])
    cfg.seed = 2026
    env = ManagerBasedRlEnv(cfg, device='cuda:0')
    vec = RslRlVecEnvWrapper(env, clip_actions=1.)
    runner = MjlabOnPolicyRunner(vec, asdict(base.runner_cfg()), device='cuda:0')
    runner.load(str(args.checkpoint), load_cfg={'actor': True}, map_location='cuda:0')
    policy = runner.get_inference_policy(device='cuda:0')
    command = env.command_manager.get_term('twist')
    desired = torch.tensor([[x, 0., z] for _, x, z, h in cases], device='cuda:0').repeat_interleave(repeats, 0)
    # Replace only command sampling, not physics/reset/reward or the policy.
    heights = torch.tensor([h for _,x,z,h in cases],device='cuda:0').repeat_interleave(repeats)
    def fixed_command(dt):
        command.vel_command_b.copy_(desired)
        command.height_target.copy_(heights)
        command.height_command += (heights-command.height_command).clamp(-.03*dt,.03*dt)
    command.compute = fixed_command
    alive = torch.ones(env.num_envs, dtype=torch.bool, device='cuda:0')
    sums = torch.zeros(env.num_envs, 11, device='cuda:0')
    counts = torch.zeros(env.num_envs, device='cuda:0')
    try:
        vec.reset()
        with torch.inference_mode():
            for step in range(500):
                if step == 250:
                    desired[-2*repeats:, 2] = 0.
                    heights[-3*repeats:-2*repeats] = .235
                command.compute(0.)  # Refresh transitions; ramp only once in env.step.
                obs = env.observation_manager.compute(update_history=True)
                d = env.scene['robot'].data
                vx, wz = d.root_link_lin_vel_b[:, 0], d.root_link_ang_vel_b[:, 2]
                ex, ez = (vx-desired[:, 0]).abs(), (wz-desired[:, 2]).abs()
                tilt = base.tilt_angle(env)
                good = ((ex < .1) & (ez < .2) & (tilt < .2) & (task.height_error(env)<.015)).float()
                action = env.action_manager
                action_smooth = (action.action - 2*action.prev_action + action.prev_prev_action).square().mean(1)
                leg_idx = base.indices(env)['legs']
                joint_acceleration = (d.joint_acc[:, leg_idx] / 100.).square().mean(1)
                position = d.joint_pos[:, leg_idx]
                limits = d.soft_joint_pos_limits[:, leg_idx]
                joint_limits = (-(position-limits[..., 0]).clip(max=0.)
                                + (position-limits[..., 1]).clip(min=0.)).sum(1)
                active = alive & (step >= 100)
                # Transition case is scored only once the stop has had 2s to settle.
                if step < 350:
                    active[-3*repeats:] = False
                sums += active[:, None] * torch.stack((vx, wz, ex, ez, torch.rad2deg(tilt), good,
                    d.root_link_lin_vel_b[:, 2].square(), joint_acceleration, action_smooth, joint_limits, task.height_error(env)), 1)
                counts += active
                _, _, dones, _ = vec.step(policy(obs))
                alive &= ~env.reset_terminated
        values = sums / counts.clamp_min(1)[:, None]
        report = {'checkpoint': str(args.checkpoint), 'seed': 2026, 'seconds': 10, 'cases': {}}
        for i, (name, x, z, h) in enumerate(cases):
            group = slice(i*repeats, (i+1)*repeats)
            valid = counts[group] > 0
            mean = values[group][valid].mean(0).tolist() if valid.any() else [None]*11
            metric_names = ['vx', 'wz', 'vx_mae', 'yaw_mae', 'tilt_deg', 'tracking_fraction',
                'vertical_velocity_sq', 'joint_acceleration_norm_sq', 'action_smooth', 'joint_limit_violation', 'height_mae_m']
            report['cases'][name] = dict(zip(metric_names, mean))
            report['cases'][name]['survival_fraction'] = alive[group].float().mean().item()
            report['cases'][name]['scored_steps_per_episode'] = counts[group].tolist()
            if not (counts[group] > 0).any():
                for key in metric_names:
                    report['cases'][name][key] = None
            report['cases'][name]['passed'] = bool(alive[group].all() and (counts[group] > 0).all() and
                (values[group, 2] < .1).all() and (values[group, 3] < .2).all() and (values[group, 5] > .8).all() and (values[group,10] < .015).all())
        report['all_passed'] = all(case['passed'] for case in report['cases'].values())
        dest = args.run / (args.checkpoint.stem + '_audit.json')
        dest.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print('AUDIT ' + json.dumps(report), flush=True)
    finally:
        vec.close()


if __name__ == '__main__':
    main()
