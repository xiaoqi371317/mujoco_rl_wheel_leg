"""V5 rough viewer with selectable terrain and the existing V5 controls."""
import argparse
from dataclasses import asdict
from pathlib import Path
import torch
from mjlab.rl import MjlabOnPolicyRunner,RslRlVecEnvWrapper
from infantry_rl.mjlab_train import CheckedEnv
from infantry_rl.mjlab_rough_train import agent
from infantry_rl.mjlab_rough_audit import verify
from infantry_rl.mjlab_v5_audit import latest
from infantry_rl.mjlab_v5_play import controls
from infantry_rl import mjlab_rough_task as task
from mjlab.viewer import ViserPlayViewer


class RoughPlayViewer(ViserPlayViewer):
    """Show the terrain-only raycast group without changing physics geom groups."""
    def setup(self):
        checkbox=self._server.gui.add_checkbox
        def terrain_checkbox(label,*args,**kwargs):
            if label=='G5':kwargs['initial_value']=True
            return checkbox(label,*args,**kwargs)
        self._server.gui.add_checkbox=terrain_checkbox
        try:
            super().setup()
        finally:
            self._server.gui.add_checkbox=checkbox
        self._scene.geom_groups_visible[5]=True
        self._scene._sync_visibilities()
        self._scene.request_update()
        print('ROUGH_TERRAIN_VISIBLE group=5',flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--checkpoint',type=Path)
    args=ap.parse_args();verify(args.run);torch.set_num_threads(4)
    cfg=task.make_cfg(1,play=True);cfg.episode_length_s=120.
    env=CheckedEnv(cfg,device='cuda:0');vec=RslRlVecEnvWrapper(env,clip_actions=1.)
    try:
        runner=MjlabOnPolicyRunner(vec,asdict(agent()),device='cuda:0');checkpoint=args.checkpoint or latest(args.run)
        runner.load(str(checkpoint),load_cfg={'actor':True},map_location='cuda:0');policy=runner.get_inference_policy(device='cuda:0')
        cmd=env.command_manager.get_term('twist');controls(cmd);cmd.preserve_jumps=False
        env._rough_force_type=1;env._rough_force_level=1;env._rough_initial_height=.24
        original=cmd.create_gui
        def gui(name,server,get_env_idx,**kwargs):
            original(name,server,get_env_idx,**kwargs)
            with server.gui.add_folder('地形选择'):
                kind=server.gui.add_dropdown('路面',options=['平地','小坑和凸起','缓波起伏'],initial_value='小坑和凸起')
                level=server.gui.add_slider('起伏等级',min=0,max=2,step=1,initial_value=1)
                reset=server.gui.add_button('切换地形并复位')
                server.gui.add_markdown('凹凸路面用于移动稳定性测试；跳跃只在平地区域启用。接近地形边界时自动复位。')
                @reset.on_click
                def change(_):
                    env._rough_force_type=['平地','小坑和凸起','缓波起伏'].index(kind.value)
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
        print('ROUGH_CHECKPOINT',checkpoint,flush=True);RoughPlayViewer(vec,control_policy,viser_server=server).run()
    finally:vec.close()

if __name__=='__main__':main()
