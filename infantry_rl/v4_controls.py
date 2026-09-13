"""Direct Viser controls; no keyboard focus or second port required."""
import time
from infantry_rl.mjlab_cmdvel_v4_task import LOW_HEIGHT, HIGH_HEIGHT, SPIN_SPEED


def configure_web_controls(command, keyboard, fast=False):
    def create_gui(name, server, get_env_idx, **kwargs):
        with server.gui.add_folder('小车驾驶'):
            server.gui.add_markdown('**网页控件直接控制**。先播放仿真，再点击方向按钮或拖动滑块。转弯后点击“停止转向”，全部停止点击“停止运动”。')
            source = server.gui.add_dropdown('控制方式', options=['网页控件', '键盘页面（8081）'], initial_value='网页控件')
            if fast:
                server.gui.add_markdown('高速版：正速度为已纠正的前进方向，最高 1.3 m/s。升高仅用于静止；行驶或自转自动切回低站姿。')
            vx = server.gui.add_slider('前后速度 (m/s)', min=-.5 if fast else -.3, max=1.3 if fast else .5, step=.01, initial_value=0.)
            wz = server.gui.add_slider('转向角速度 (rad/s)', min=-1.5, max=1.5, step=.05, initial_value=0.)
            high = server.gui.add_checkbox('撑腿升高：23.5 → 28 cm', initial_value=False)
            spin = server.gui.add_checkbox('小陀螺：原地 3 rad/s', initial_value=False)
            for label, x, z in [('前进',1.3 if fast else .3,0.), ('后退',-.5 if fast else -.2,0.), ('左转',None,1.2), ('右转',None,-1.2), ('停止转向',None,0.), ('停止运动',0.,0.)]:
                button = server.gui.add_button(label)
                @button.on_click
                def drive(_, x=x, z=z):
                    spin.value = False
                    if x is not None: vx.value = x
                    wz.value = z
            if fast:
                @high.on_update
                def raise_static(_):
                    if high.value:
                        vx.value=wz.value=0.
                        spin.value=False
                @spin.on_update
                def spin_low(_):
                    if spin.value:high.value=False
                def drive_low(_):
                    if abs(vx.value)>.01 or abs(wz.value)>.01:high.value=False
                vx.on_update(drive_low)
                wz.on_update(drive_low)
            reset = server.gui.add_button('全部归零 / 恢复低站姿')
            def clear():
                vx.value = wz.value = 0.
                spin.value = high.value = False
            @reset.on_click
            def reset_all(_): clear()
            @source.on_update
            def source_changed(_): clear()
            @server.on_client_disconnect
            def disconnected(_): clear()
            server.gui.add_markdown('按钮/滑块设定持续指令；断开查看器会自动归零。**撑腿升高不是跳跃**，此策略尚未训练腾空与落地。部分高站姿及旋转场景仍未通过验收。')
            feedback = server.gui.add_text('实际速度 / 角速度 / 高度', initial_value='等待仿真运行', disabled=True)
        last_update = 0.
        def read():
            nonlocal last_update
            if not server.get_clients(): return 0.,0.,LOW_HEIGHT
            if time.monotonic()-last_update > .2:
                d=command.robot.data
                idx=get_env_idx()
                actual=d.root_link_lin_vel_b[idx,0].item()*(-1. if fast else 1.)
                yaw=d.root_link_ang_vel_b[idx,2].item()
                height=(d.root_link_pos_w[idx,2]-command._env.scene.env_origins[idx,2]).item()
                feedback.value=f'{actual:.2f} m/s | {yaw:.2f} rad/s | {height*100:.1f} cm'
                last_update=time.monotonic()
            if source.value == '键盘页面（8081）': return keyboard.read()
            if fast:
                from infantry_rl.fast_control import resolve_command
                return resolve_command(vx.value,wz.value,high.value,spin.value)
            x,z = (0.,SPIN_SPEED) if spin.value else (vx.value,wz.value)
            return x,z,HIGH_HEIGHT if high.value else LOW_HEIGHT
        command.manual_source=read
    # Before viewer GUI is initialized, keep the robot stationary.
    command.manual_source=lambda:(0.,0.,LOW_HEIGHT)
    command.create_gui=create_gui
