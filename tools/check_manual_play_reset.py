"""GPU regression: flight, falls, timeout and boundary cannot reset playback; the viewer can."""
import json
from pathlib import Path
import torch
import viser
from infantry_rl.model_selection import select_model
from infantry_rl import mjlab_rugged_task as task
from infantry_rl.v4_course import terrain_cfg
from infantry_rl.mjlab_rugged_play import configure_manual_reset,RoughPlayViewer
from infantry_rl.mjlab_v5_play import controls
from infantry_rl.mjlab_train import CheckedEnv
from mjlab.rl import RslRlVecEnvWrapper


def main():
    select_model('fix');torch.set_num_threads(4)
    result={}
    for manual in [False,True]:
        cfg=task.make_cfg(1,play=True);cfg.scene.terrain=terrain_cfg('v4')
        cfg.episode_length_s=120.
        original_terms=set(cfg.terminations)
        if manual:configure_manual_reset(cfg)
        env=CheckedEnv(cfg,device='cuda:0');vec=RslRlVecEnvWrapper(env,clip_actions=1.)
        server=viewer=None
        try:
            env._rough_force_type=0;env._rough_force_level=0;env._rough_initial_height=.24
            cmd=env.command_manager.get_term('twist');controls(cmd);cmd.preserve_jumps=False
            vec.reset();robot=env.scene['robot'];resets=[];original=env._reset_idx
            def track(ids=None):
                resets.append(True);return original(ids)
            env._reset_idx=track
            def pose(x,z,roll=0.):
                state=robot.data.root_link_pose_w.clone();state[:,0]=x;state[:,1]=0.;state[:,2]=z
                state[:,3:]=torch.tensor([torch.cos(torch.tensor(roll/2)),torch.sin(torch.tensor(roll/2)),0.,0.],device=env.device)
                robot.write_root_link_pose_to_sim(state)
                robot.write_root_link_velocity_to_sim(torch.zeros((1,6),device=env.device))
                env.sim.forward();env.sim.sense()
            def step(n):
                # Match BaseViewer._execute_step: inference_mode would create sensor
                # buffers that cannot later be reset by an ordinary GUI callback.
                with torch.no_grad():
                    for _ in range(n):
                        _,_,done,_=vec.step(torch.zeros((1,6),device=env.device))
                        if manual:assert not done.any()
            pose(1.,1.2)
            if not manual:
                step(3);assert resets,'The original flight-reset bug was not reproduced'
                result['original_flight_resets']=len(resets)
                continue
            env.episode_length_buf[:]=env.max_episode_length+1
            step(1)
            assert robot.data.root_link_pos_w[0,2]>.9,'Airborne state teleported to spawn'
            step(89)
            assert env.episode_length_buf.item()>env.max_episode_length
            pose(1.,.45,roll=1.57);step(20)
            pose(15.,.4);step(10)
            assert robot.data.root_link_pos_w[0,0]>14.
            assert not resets,'Playback reset without a user request'
            assert env.termination_manager.active_terms==['nan']
            server=viser.ViserServer(host='127.0.0.1',port=8099)
            viewer=RoughPlayViewer(vec,lambda obs:torch.zeros((1,6),device=env.device),viser_server=server)
            viewer.setup()
            # This is the existing Reset Environment button's handler.
            viewer.reset_environment()
            assert len(resets)==1 and env.episode_length_buf.item()==0
            assert abs(robot.data.root_link_pos_w[0,0].item())<1e-5
            assert abs(task.local_height(env).item()-.24)<.002
            step(2);assert len(resets)==1
            result.update(manual_reset_count=len(resets),continued_steps=120,
                original_training_terms=sorted(original_terms),nan_check_retained=True,viewer_reset_passed=True)
        finally:
            if viewer is not None:viewer.close()
            if server is not None:server.stop()
            vec.close()
    unchanged=task.make_cfg(1)
    assert unchanged.auto_reset and set(unchanged.terminations)==original_terms
    path=Path('artifacts/infantry_audit/manual_play_reset.json');path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2));print('MANUAL_PLAY_RESET_PASS',json.dumps(result),flush=True)


if __name__=='__main__':main()
