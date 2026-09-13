"""V5 rough viewer with selectable terrain and the existing V5 controls."""
import argparse
import json
import hashlib
from dataclasses import asdict
from pathlib import Path
import torch
from mjlab.rl import MjlabOnPolicyRunner,RslRlVecEnvWrapper
from infantry_rl.mjlab_train import CheckedEnv
from infantry_rl.mjlab_motion_train import agent
from infantry_rl.mjlab_rugged_audit import verify
from infantry_rl.mjlab_v5_audit import latest
from infantry_rl.motion_control import resolve_command
from infantry_rl import mjlab_gas_task as task
from mjlab.viewer import ViserPlayViewer


from infantry_rl.mjlab_rough_play import RoughPlayViewer


from infantry_rl.mjlab_rugged_play import configure_manual_reset


def controls(cmd):
    cmd.manual_source=lambda:(0.,0.,.235);cmd.auto_jump=False;cmd.auto_height=False
    def create_gui(name,server,get_env_idx,**kwargs):
        with server.gui.add_folder('行进跳跃保速'):
            speed=server.gui.add_slider('前进速度 m/s',min=-.5,max=2.,step=.05,initial_value=0.)
            yaw=server.gui.add_slider('转向 rad/s',min=-1.5,max=1.5,step=.1,initial_value=0.)
            high=server.gui.add_checkbox('撑高 F（支持行进中）',initial_value=False)
            spin=server.gui.add_checkbox('自转 R',initial_value=False)
            stop=server.gui.add_button('停止运动')
            jump=server.gui.add_button('跳跃一次（保持起跳前速度）')
            natural=server.gui.add_button('收腿复位并起立')
            @stop.on_click
            def clear(_):
                speed.value=yaw.value=0.;high.value=spin.value=False;cmd.request_jump=False
            @jump.on_click
            def trigger(_):
                # Preserve the user's speed and height controls on a jump request.
                cmd.request_jump=True
            @natural.on_click
            def recover(_):clear(None);cmd._v5_reset_requested=True
            @server.on_client_disconnect
            def disconnect(_):clear(None)
            status=server.gui.add_text('阶段',initial_value='等待播放',disabled=True)
        def read():
            status.value=['静止/行驶','起立','下蹲','蹬地','腾空','落地'][int(cmd.phase[0])]
            if not server.get_clients():return 0.,0.,.235
            return resolve_command(speed.value,yaw.value,high.value,spin.value)
        cmd.manual_source=read
    cmd.create_gui=create_gui


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--checkpoint',type=Path)
    ap.add_argument('--terrain',choices=['rugged','v4','v4-steps'],default='rugged')
    ap.add_argument('--original-friction',action='store_true',help='Use original V4 contact friction in course playback')
    ap.add_argument('--force',type=float,default=200.,help='Constant force per side in N; experimental, not hardware calibrated')
    ap.add_argument('--tuck',action='store_true',help='Enable the experimental airborne retraction used in dynamic evaluation')
    args=ap.parse_args();
    if not 0<=args.force<=300:raise ValueError('force must be within 0..300 N')
    verify(args.run);torch.set_num_threads(4)
    meta=json.loads((args.run/'run.json').read_text())
    physics=Path(__file__).parent/'robot/xmls/training_scene_fix_gas_trial.xml'
    if meta.get('physics_model_sha256'):assert hashlib.sha256(physics.read_bytes()).hexdigest()==meta['physics_model_sha256']
    cfg=task.make_cfg(1,play=True,stage='tuck' if args.tuck else meta.get('stage','stance'));cfg.episode_length_s=120.
    course=args.terrain!='rugged'
    if course:
        from infantry_rl.v4_course import terrain_cfg
        cfg.scene.terrain=terrain_cfg(args.terrain,training_friction=not args.original_friction)
    configure_manual_reset(cfg)
    env=CheckedEnv(cfg,device='cuda:0');vec=RslRlVecEnvWrapper(env,clip_actions=1.)
    try:
        runner=MjlabOnPolicyRunner(vec,asdict(agent()),device='cuda:0')
        selection=args.run/'refine_selection.json'
        recommended=args.run/json.loads(selection.read_text())['recommended_checkpoint'] if selection.exists() else latest(args.run)
        checkpoint=args.checkpoint or recommended
        print('PLAYBACK CHECKPOINT:',checkpoint.resolve(),flush=True)
        runner.load(str(checkpoint),load_cfg={'actor':True},map_location='cuda:0');policy=runner.get_inference_policy(device='cuda:0')
        cmd=env.command_manager.get_term('twist');controls(cmd);cmd.preserve_jumps=False;cmd.extra_jumps=False
        env._gas_force=torch.full((1,),args.force,device=env.device)
        print('GAS FORCE PER SIDE:',args.force,'N; TUCK:',args.tuck,flush=True)
        env._rough_force_type=0;env._rough_force_level=0;env._rough_initial_height=.24
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
                kind=server.gui.add_dropdown('路面',options=['平地','密集坑洼','较大起伏'],initial_value='平地')
                level=server.gui.add_slider('起伏等级',min=0,max=2,step=1,initial_value=0)
                reset=server.gui.add_button('切换地形并复位')
                server.gui.add_markdown('可行进中撑高和跳跃；最高难度路面仍需逐步测试。驶出地形后可手动重置环境。')
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
        print('MOTION_CHECKPOINT',checkpoint,flush=True);RoughPlayViewer(vec,control_policy,viser_server=server).run()
    finally:vec.close()

if __name__=='__main__':main()
