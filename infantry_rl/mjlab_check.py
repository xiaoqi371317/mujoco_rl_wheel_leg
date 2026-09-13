"""GPU gates for reset geometry, control layout, costs, and timeouts."""
import torch
from infantry_rl.mjlab_task import make_cfg, constraint_error, bad_contact, reward_feature, WEIGHTS
from infantry_rl.mjlab_train import CheckedEnv


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', choices=['balance', 'cmd_vel', 'cmd_vel_v3'], default='balance')
    args = ap.parse_args()
    factory = make_cfg
    if args.task == 'cmd_vel':
        from infantry_rl.mjlab_cmdvel_task import make_cfg as factory
    elif args.task == 'cmd_vel_v3':
        from infantry_rl.mjlab_cmdvel_v3_task import make_cfg as factory
    cfg = factory(64)
    cfg.seed = 123
    env = CheckedEnv(cfg, device="cuda:0")
    try:
        obs, _ = env.reset(seed=123)
        assert obs["actor"].shape == (64, 47)
        assert constraint_error(env).max().item() < 1e-5
        assert not bad_contact(env).any().item()
        assert env.action_manager.get_term("legs").target_names == ["rf0_Joint", "lf0_Joint", "r20_Joint", "l20_Joint"]
        assert env.action_manager.get_term("wheels").target_names == ["r_wheel_Joint", "l_wheel_Joint"]
        if args.task in ('cmd_vel', 'cmd_vel_v3'):
            command = env.command_manager.get_term('twist')
            sampled = []
            for _ in range(32):
                command._resample_command(torch.arange(64, device=env.device))
                command._update_command()
                sampled.append(command.command.clone())
            commands = torch.cat(sampled)
            assert (commands[:, 1] == 0).all()
            assert ((commands[:, 0] >= -.3) & (commands[:, 0] <= .5)).all()
            assert (commands[:, 2].abs() <= 1.5).all()
            stopped = (commands[:, 0] == 0) & (commands[:, 2] == 0)
            straight = (commands[:, 0] != 0) & (commands[:, 2] == 0)
            spin = (commands[:, 0] == 0) & (commands[:, 2] != 0)
            arc = (commands[:, 0] != 0) & (commands[:, 2] != 0)
            for observed, expected in [(stopped, .35), (straight, .30), (spin, .15), (arc, .20)]:
                assert abs(observed.float().mean().item() - expected) < .05
        max_error = 0.
        for _ in range(96):
            obs, reward, _, _, _ = env.step(torch.rand(64, 6, device="cuda:0") * 2 - 1)
            assert torch.isfinite(obs["actor"]).all() and torch.isfinite(reward).all()
            max_error = max(max_error, constraint_error(env).max().item())
            for name, weight in WEIGHTS.items():
                if weight < 0:
                    assert (weight * reward_feature(env, name) <= 0).all()
        assert max_error < .005, max_error
        env.reset(seed=123)
        env.episode_length_buf[:] = env.max_episode_length - 1
        _, _, terminated, truncated, _ = env.step(torch.zeros(64, 6, device="cuda:0"))
        assert truncated.all() and not terminated.any()
        # Drop the whole closed-chain robot into the floor: every non-wheel body
        # remains collision enabled and its ground contact must be detected.
        robot = env.scene["robot"]
        state = robot.data.default_root_state.clone()
        state[:, :3] += env.scene.env_origins
        state[:, 2] = .11
        robot.write_root_state_to_sim(state)
        env.scene.write_data_to_sim()
        env.sim.forward()
        env.sim.sense()
        assert bad_contact(env).all()
        print("MJLAB_GPU_GATES_OK", {"max_closure_m": max_error, "observations": 47, "actions": 6}, flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
