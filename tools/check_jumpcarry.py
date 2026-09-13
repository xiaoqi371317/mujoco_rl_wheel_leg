"""Check measured-velocity capture, body-frame conversion, reward sign and stop priority."""
from types import SimpleNamespace as NS
import torch
from infantry_rl import mjlab_jumpcarry_task as task
from infantry_rl.model_selection import select_model
from infantry_rl.mjlab_train import CheckedEnv
from infantry_rl.mjlab_rugged_play import configure_manual_reset
from mjlab.rl import RslRlVecEnvWrapper

def main():
    q=torch.tensor([[2**-.5,0.,0.,2**-.5]])
    assert torch.allclose(task.world_to_body(q,torch.tensor([[1.,0.,0.]])),torch.tensor([[0.,-1.,0.]]),atol=1e-6)
    d=NS(root_link_lin_vel_w=torch.tensor([[-1.,0.,0.]]),projected_gravity_b=torch.tensor([[0.,0.,-1.]]))
    cmd=NS(carry_world=torch.tensor([[-1.,0.,0.]]),phase=torch.tensor([3]),carry_valid=torch.tensor([True]))
    fake=NS(scene={'robot':NS(data=d)},command_manager=NS(get_term=lambda name:cmd))
    assert task.reward(fake,'carry_quality').item()==1. and task.reward(fake,'braking').item()==0.
    d.root_link_lin_vel_w.zero_();assert task.reward(fake,'carry_error').item()>0. and task.reward(fake,'braking').item()>0.
    select_model('fix');torch.set_num_threads(4)
    cfg=configure_manual_reset(task.make_cfg(4,play=True));env=CheckedEnv(cfg,device='cuda:0');vec=RslRlVecEnvWrapper(env,clip_actions=1.)
    try:
        env._rough_force_type=0;env._rough_force_level=0;env._rough_initial_height=.24
        cmd=env.command_manager.get_term('twist');cmd.extra_jumps=False;cmd.auto_height=False
        drive=torch.zeros(4,3,device=env.device);drive[:,0]=-1.3
        h=torch.full((4,),.235,device=env.device);cmd.external_source=lambda:(drive,h);vec.reset()
        with torch.no_grad():
            for _ in range(20):vec.step(torch.zeros(4,6,device=env.device))
        robot=env.scene['robot'];vel=torch.zeros(4,6,device=env.device);vel[:,0]=-.9;vel[:,1]=.1
        robot.write_root_link_velocity_to_sim(vel);env.sim.forward();env.sim.sense()
        measured=robot.data.root_link_lin_vel_w.clone();measured[:,2]=0.
        cmd.request_jump=True;cmd.compute(0.)
        assert (cmd.phase==2).all() and torch.allclose(cmd.carry_world,measured)
        assert not torch.allclose(cmd.carry_world,drive),'Captured requested speed instead of measured speed'
        for phase in [2,3,4,5]:
            cmd.phase[:]=phase;cmd.age[:]=0.;cmd.compute(0.)
            target=task.world_to_body(robot.data.root_link_quat_w,measured)
            assert torch.allclose(cmd.vel_command_b[:,:2],target[:,:2],atol=1e-5)
            assert torch.allclose(cmd.carry_world,measured)
        cmd.external_source=None;cmd.manual_source=lambda:(0.,0.,.235);cmd.phase[:]=3;cmd.compute(0.)
        assert (cmd.carry_world==0).all() and (cmd.vel_command_b[:,:2]==0).all()
        assert vec.get_observations()['actor'].shape==(4,56)
        print('JUMPCARRY_PASS actual_speed_snapshot world_frame phase_2_3_4_5 reward_sign stop_priority obs56',flush=True)
    finally:vec.close()
if __name__=='__main__':main()
