"""Regression: unauthorized spin must lose reward; commanded spin must win."""
from types import SimpleNamespace
import torch
from infantry_rl.mjlab_cmdvel_task import tracking_reward, MixedCommand
from infantry_rl.mjlab_cmdvel_v3_task import mechanical_reward


def test_reward_ranking():
    command = torch.zeros(1, 3)
    data = SimpleNamespace(root_link_lin_vel_b=torch.zeros(1, 3),
        root_link_ang_vel_b=torch.zeros(1, 3), projected_gravity_b=torch.tensor([[0., 0., -1.]]),
        root_link_pos_w=torch.tensor([[0., 0., .235]]))
    class Scene(dict):
        env_origins = torch.zeros(1, 3)
    env = SimpleNamespace(scene=Scene(robot=SimpleNamespace(data=data)),
        command_manager=SimpleNamespace(get_command=lambda name: command))
    def score():
        return (4*tracking_reward(env, 'joint_tracking')-1.5*tracking_reward(env, 'yaw_error')
                -2*tracking_reward(env, 'forward_error')).item()
    stopped = score()
    data.root_link_ang_vel_b[0, 2] = 3.25
    unwanted = score()
    data.root_link_ang_vel_b[0, 2] = 6.5
    assert score() < unwanted < stopped
    for sign in [-1., 1.]:
        command[0, 2] = sign*1.2
        data.root_link_ang_vel_b[0, 2] = 0
        wrong = score()
        data.root_link_ang_vel_b[0, 2] = command[0, 2]
        assert score() > wrong
    print('REWARD_RANKING_OK')


def test_v3_mechanical_regularizers():
    data = SimpleNamespace(
        root_link_lin_vel_b=torch.zeros(1, 3),
        joint_acc=torch.zeros(1, 4),
        joint_pos=torch.zeros(1, 4),
        soft_joint_pos_limits=torch.tensor([[[-1., 1.]] * 4]),
    )
    action = SimpleNamespace(
        action=torch.ones(1, 6),
        prev_action=torch.full((1, 6), .5),
        prev_prev_action=torch.zeros(1, 6),
    )
    env = SimpleNamespace(
        scene={'robot': SimpleNamespace(data=data)},
        action_manager=action,
        _infantry_indices={'legs': torch.arange(4)},
    )
    assert mechanical_reward(env, 'vertical_velocity').item() == 0.
    assert mechanical_reward(env, 'joint_acceleration').item() == 0.
    assert mechanical_reward(env, 'action_smooth').item() == 0.
    assert mechanical_reward(env, 'joint_limits').item() == 0.
    data.root_link_lin_vel_b[0, 2] = .2
    data.joint_acc[0, 0] = 100.
    action.prev_action.zero_()
    data.joint_pos[0, 0] = 1.2
    assert mechanical_reward(env, 'vertical_velocity').item() > 0.
    assert mechanical_reward(env, 'joint_acceleration').item() > 0.
    assert mechanical_reward(env, 'action_smooth').item() > 0.
    assert abs(mechanical_reward(env, 'joint_limits').item() - .2) < 1e-6
    print('V3_MECHANICAL_REWARDS_OK')


if __name__ == '__main__':
    test_reward_ranking()
    test_v3_mechanical_regularizers()
