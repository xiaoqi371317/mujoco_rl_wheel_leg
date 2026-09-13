"""Verify command preservation and reward direction before a moving-maneuver run."""
from contextlib import nullcontext
from types import SimpleNamespace as NS
import torch
from infantry_rl.model_selection import select_model
from infantry_rl import mjlab_motion_task as task
from infantry_rl.mjlab_motion_play import controls
from infantry_rl.mjlab_train import CheckedEnv
from infantry_rl.mjlab_rugged_play import configure_manual_reset
from infantry_rl.motion_control import resolve_command
from mjlab.rl import RslRlVecEnvWrapper


def main():
    assert resolve_command(2.,0.,True)==(-2.,0.,.28)
    assert resolve_command(2.,0.,True,True)==(0.,3.,.235)
    # Exercise the actual GUI callback with simple handles, without a browser client.
    handles={}
    class Handle:
        def __init__(self,value=None):self.value=value
        def on_click(self,fn):self.click=fn;return fn
    class Gui:
        def add_folder(self,*a,**kw):return nullcontext()
        def add(self,label,**kw):
            handles[label]=Handle(kw.get('initial_value'));return handles[label]
        add_slider=add_checkbox=add_button=add_text=add
    server=NS(gui=Gui(),get_clients=lambda:{1:True},on_client_disconnect=lambda f:f)
    cmd=NS(phase=torch.zeros(1,dtype=torch.long))
    controls(cmd);cmd.create_gui('twist',server,lambda:0)
    handles['前进速度 m/s'].value=2.;handles['撑高 F（支持行进中）'].value=True
    handles['跳跃一次（保持前进指令）'].click(None)
    assert cmd.request_jump and cmd.manual_source()==(-2.,0.,.28)
    # Moving maneuver penalties must prefer matching forward speed to braking.
    d=NS(projected_gravity_b=torch.tensor([[0.,0.,-1.]]),root_link_lin_vel_b=torch.tensor([[-1.,0.,0.]]),
        root_link_ang_vel_b=torch.zeros(1,3),root_link_lin_vel_w=torch.zeros(1,3))
    c=NS(phase=torch.tensor([5]),vel_command_b=torch.tensor([[-1.,0.,0.]]))
    fake=NS(scene={'robot':NS(data=d)},command_manager=NS(get_term=lambda name:c))
    assert task.reward(fake,'maneuver_drift').item()==0
    d.root_link_lin_vel_b.zero_();assert task.reward(fake,'maneuver_drift').item()==1
    select_model('fix');torch.set_num_threads(4)
    cfg=configure_manual_reset(task.make_cfg(4,play=True))
    env=CheckedEnv(cfg,device='cuda:0');vec=RslRlVecEnvWrapper(env,clip_actions=1.)
    try:
        env._rough_force_type=0;env._rough_force_level=0;env._rough_initial_height=.24
        cmd=env.command_manager.get_term('twist');cmd.auto_jump=False;cmd.auto_height=False
        desired=torch.zeros(4,3,device=env.device);height=torch.full((4,),.235,device=env.device)
        cmd.external_source=lambda:(desired,height);vec.reset()
        with torch.no_grad():
            for _ in range(20):vec.step(torch.zeros(4,6,device=env.device))
        desired[:,0]=torch.tensor([-.5,-1.,-2.,0.],device=env.device);height[:]=.28
        for _ in range(30):cmd.compute(.02)
        assert torch.allclose(cmd.height_command,torch.full_like(height,.28))
        cmd.request_jump=True;cmd.compute(0.)
        assert (cmd.phase==2).all() and torch.equal(cmd.vel_command_b,desired)
        for _ in range(110):
            cmd.compute(.02);assert torch.equal(cmd.vel_command_b,desired)
        assert vec.get_observations()['actor'].shape==(4,56)
        assert all((env.scene[name].data.distances>=0).all() for name in ('local_ground','ground_left','ground_right'))
        print('MOTION_CONTROL_PASS GUI_preserves_speed F_at_2mps moving_jump_command reward_direction ground_rays obs56',flush=True)
    finally:vec.close()


if __name__=='__main__':main()
