"""V5 rough viewer with selectable terrain and the existing V5 controls."""
import argparse
import json
from dataclasses import asdict
from pathlib import Path
import torch
from mjlab.rl import MjlabOnPolicyRunner,RslRlVecEnvWrapper
from infantry_rl.mjlab_train import CheckedEnv
from infantry_rl.mjlab_rugged_train import agent
from infantry_rl.mjlab_rugged_audit import verify
from infantry_rl.mjlab_v5_audit import latest
from infantry_rl.mjlab_v5_play import controls
from infantry_rl import mjlab_rugged_task as task
from mjlab.viewer import ViserPlayViewer


from infantry_rl.mjlab_rough_play import RoughPlayViewer


def configure_manual_reset(cfg):
    """Keep playback running through flight, falls and time limits until reset is requested."""
    # auto_reset=False alone leaves a terminated environment unable to step again.
    # Remove training episode endings as well, so flight/landing can continue.
    # CheckedEnv still raises on non-finite physics instead of hiding it with a reset.
    cfg.terminations={'nan':cfg.terminations['nan']}
    cfg.auto_reset=False
    return cfg


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--checkpoint',type=Path)
    ap.add_argument('--terrain',choices=['rugged','v4','v4-steps'],default='rugged')
    ap.add_argument('--original-friction',action='store_true',help='Use original V4 contact friction in course playback')
    args=ap.parse_args();verify(args.run);torch.set_num_threads(4)
    cfg=task.make_cfg(1,play=True);cfg.episode_length_s=120.
    course=args.terrain!='rugged'
    if course:
        from infantry_rl.v4_course import terrain_cfg
        cfg.scene.terrain=terrain_cfg(args.terrain,training_friction=not args.original_friction)
    configure_manual_reset(cfg)
    env=CheckedEnv(cfg,device='cuda:0');vec=RslRlVecEnvWrapper(env,clip_actions=1.)
    try:
        runner=MjlabOnPolicyRunner(vec,asdict(agent()),device='cuda:0')
        selection=args.run/'rugged_comparison.json'
        recommended=args.run/json.loads(selection.read_text())['recommended_checkpoint'] if selection.exists() else latest(args.run)
        checkpoint=args.checkpoint or recommended
        runner.load(str(checkpoint),load_cfg={'actor':True},map_location='cuda:0');policy=runner.get_inference_policy(device='cuda:0')
        cmd=env.command_manager.get_term('twist');controls(cmd);cmd.preserve_jumps=False
        env._rough_force_type=0 if course else 1;env._rough_force_level=0 if course else 2;env._rough_initial_height=.24
        original=cmd.create_gui
        def gui(name,server,get_env_idx,**kwargs):
            original(name,server,get_env_idx,**kwargs)
            with server.gui.add_folder('手动复位'):
                server.gui.add_markdown('腾空、落地、摔倒和超时均不自动复位。需要重新开始时，请点击查看器的“重置环境”按钮。')
            if course:
                with server.gui.add_folder('原 V4 场地'):
                    server.gui.add_markdown('当前场地：'+args.terrain+'。出生点在平地；前进沿 -X 方向驶向台阶/斜坡。高台阶尚未专项训练。')
                    server.gui.add_markdown('地面摩擦：'+('原 V4 参数' if args.original_friction else '训练参数'))
                return
            with server.gui.add_folder('地形选择'):
                kind=server.gui.add_dropdown('路面',options=['平地','密集坑洼','较大起伏'],initial_value='密集坑洼')
                level=server.gui.add_slider('起伏等级',min=0,max=2,step=1,initial_value=2)
                reset=server.gui.add_button('切换地形并复位')
                server.gui.add_markdown('凹凸路面用于移动稳定性测试；跳跃只在平地区域启用。驶出地形后可手动重置环境。')
                @reset.on_click
                def change(_):
                    env._rough_force_type=['平地','密集坑洼','较大起伏'].index(kind.value)
                    env._rough_force_level=int(level.value);cmd._rough_reset=True
        cmd.create_gui=gui;vec.reset()
        def control_policy(obs):
            if getattr(cmd,'_rough_reset',False) or getattr(cmd,'_v5_reset_requested',False):
                env._rough_initial_height=.16 if getattr(cmd,'_v5_reset_requested',False) and env._rough_force_type==0 else .24
                cmd._rough_reset=False;cmd._v5_reset_requested=False;obs,_=vec.reset()
            return policy(obs)
        import viser
        from infantry_rl.mjlab_play import _install_zh_gui,_apply_viewer_palette
        _apply_viewer_palette(env);server=viser.ViserServer(host='127.0.0.1',port=8080);_install_zh_gui(server)
        print('RUGGED_CHECKPOINT',checkpoint,flush=True);RoughPlayViewer(vec,control_policy,viser_server=server).run()
    finally:vec.close()

if __name__=='__main__':main()
