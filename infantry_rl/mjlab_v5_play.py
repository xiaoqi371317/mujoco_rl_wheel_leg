"""V5 web viewer: drive, one-shot jump, and natural crouch reset/recovery."""
import argparse
from dataclasses import asdict
import torch
from mjlab.rl import MjlabOnPolicyRunner,RslRlVecEnvWrapper
from infantry_rl.mjlab_train import CheckedEnv
from infantry_rl.mjlab_v5_train import agent_cfg
from infantry_rl.mjlab_v5_audit import verify,latest
from infantry_rl import mjlab_v5_task as task

def controls(cmd):
    cmd.manual_source=lambda:(0.,0.,.235);cmd.auto_jump=False
    def create_gui(name,server,get_env_idx,**kwargs):
        with server.gui.add_folder('V5 跳跃与起立'):
            server.gui.add_markdown('先播放仿真。跳跃按钮触发一次下蹲、蹬地、腾空、落地；请在停稳时触发。')
            vx=server.gui.add_slider('前进速度 m/s',min=-.5,max=1.3,step=.05,initial_value=0.)
            wz=server.gui.add_slider('转向 rad/s',min=-3.,max=3.,step=.1,initial_value=0.)
            high=server.gui.add_checkbox('撑高 F（静止）',initial_value=False)
            spin=server.gui.add_checkbox('自转 R',initial_value=False)
            stop=server.gui.add_button('停止运动')
            jump=server.gui.add_button('跳跃一次')
            natural=server.gui.add_button('重置为 16 cm 收腿姿态并自主起立')
            @stop.on_click
            def clear(_):vx.value=wz.value=0.;high.value=spin.value=False
            @jump.on_click
            def trigger(_):clear(None);cmd.request_jump=True
            @natural.on_click
            def recover(_):
                clear(None);cmd._v5_reset_requested=True
            @server.on_client_disconnect
            def disconnect(_):clear(None)
            status=server.gui.add_text('阶段',initial_value='等待播放',disabled=True)
        def read():
            status.value=['静止/行驶','起立','下蹲','蹬地','腾空','落地站稳'][int(cmd.phase[0])]
            if not server.get_clients():return 0.,0.,.235
            from infantry_rl.fast_control import resolve_command
            return resolve_command(vx.value,wz.value,high.value,spin.value)
        cmd.manual_source=read
    cmd.create_gui=create_gui

def main():
    from pathlib import Path
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--checkpoint',type=Path)
    args=ap.parse_args();verify(args.run);torch.set_num_threads(4)
    cfg=task.make_cfg(1,play=True);cfg.episode_length_s=120.
    env=CheckedEnv(cfg,device='cuda:0');vec=RslRlVecEnvWrapper(env,clip_actions=1.)
    try:
        runner=MjlabOnPolicyRunner(vec,asdict(agent_cfg()),device='cuda:0')
        checkpoint=args.checkpoint or latest(args.run)
        runner.load(str(checkpoint),load_cfg={'actor':True},map_location='cuda:0')
        policy=runner.get_inference_policy(device='cuda:0');cmd=env.command_manager.get_term('twist')
        controls(cmd);env._v5_reset_height=.16;vec.reset()
        def control_policy(obs):
            if getattr(cmd,'_v5_reset_requested',False):
                cmd._v5_reset_requested=False;obs,_=vec.reset()
            return policy(obs)
        import viser
        from mjlab.viewer import ViserPlayViewer
        from infantry_rl.mjlab_play import _install_zh_gui,_apply_viewer_palette
        _apply_viewer_palette(env)
        server=viser.ViserServer(host='127.0.0.1',port=8080);_install_zh_gui(server)
        print('V5_CHECKPOINT',checkpoint,flush=True)
        ViserPlayViewer(vec,control_policy,viser_server=server).run()
    finally:vec.close()

if __name__=='__main__':main()
