"""Evaluate or visualize actual mjlab checkpoints with matching normalization."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from infantry_rl.mjlab_task import make_cfg, runner_cfg, MODEL, tilt_angle, velocity_error

# Height of the cmd_vel debug arrows above the robot base (robot frame, before
# the viz scale) in the viewer. mjlab's default (0.2) hides them inside the
# chassis; raise it so the mesh does not occlude the arrows.
ARROW_Z_OFFSET = 1.0


# ---- Viewer appearance -------------------------------------------------------
# Keep this in the play entry point, rather than in infantry.xml: the visual
# palette must not change the model hash recorded by existing checkpoints.
_VIEWER_BODY_COLORS = {
    "base_Link_del": (0.035, 0.045, 0.055, 1.0),    # carbon-black chassis
    "rf0_Link": (0.20, 0.44, 0.66, 1.0),
    "rf1_Link": (0.20, 0.44, 0.66, 1.0),
    "lf0_Link": (0.20, 0.44, 0.66, 1.0),
    "lf1_Link": (0.20, 0.44, 0.66, 1.0),
    "r20_Link": (0.95, 0.36, 0.06, 1.0),
    "r21_Link": (0.95, 0.36, 0.06, 1.0),
    "r22_Link": (0.95, 0.36, 0.06, 1.0),
    "r23_Link": (0.95, 0.36, 0.06, 1.0),
    "l20_Link": (0.95, 0.36, 0.06, 1.0),
    "l21_Link": (0.95, 0.36, 0.06, 1.0),
    "l22_Link": (0.95, 0.36, 0.06, 1.0),
    "l23_Link": (0.95, 0.36, 0.06, 1.0),
    "r_wheel_Link": (0.04, 0.05, 0.07, 1.0),
    "l_wheel_Link": (0.04, 0.05, 0.07, 1.0),
}


def _apply_viewer_palette(env):
    """Color the compiled MuJoCo model for playback only.

    Viser converts the flat MuJoCo ground into its own grid, so its ground
    colors are set in ``_install_viewer_palette`` below.  Robot colors are
    applied to the compiled model before Viser builds its scene.
    """
    model = env.sim.mj_model
    for geom_id in range(model.ngeom):
        body_name = model.body(model.geom_bodyid[geom_id]).name.rsplit("/", 1)[-1]
        color = _VIEWER_BODY_COLORS.get(body_name)
        if color is not None:
            model.geom_rgba[geom_id] = color


# ---- Chinese localization for the viser viewer GUI ---------------------------
# mjlab / mjviser build their panels by passing English labels to
# ``server.gui.add_*``. We wrap those methods at runtime so the web UI shows
# Chinese without editing any installed package (survives conda rebuilds).
# Technical identifiers (termination/reward term names, joint/geom/site names,
# checkpoint file names) are left unchanged on purpose.
_ZH_LABELS = {
    "Controls": "控制",
    "Visualization": "可视化",
    "Groups": "分组",
    "Checkpoints": "检查点",
    "Info": "信息",
    "Simulation": "仿真控制",
    "Commands": "指令",
    "Scene": "场景",
    "Camera Feeds": "相机画面",
    "Play": "播放",
    "Pause": "暂停",
    "Step": "单步",
    "Reset Environment": "重置环境",
    "Reset": "重置",
    "Speed": "速度",
    "Slower": "减速",
    "1x": "1x",
    "Faster": "加速",
    "Sync": "同步",
    "Use Latest": "使用最新",
    "Checkpoint": "检查点",
    "Twist": "速度指令",
    "Debug Viz": "调试可视化",
    "Enabled": "启用",
    "All envs": "所有环境",
    "Plots": "曲线图",
    "Select terms": "选择指标",
    "Select": "选择",
    "All": "全部",
    "None": "无",
    "Filter": "筛选",
    "Environment": "环境",
    "Hide others": "隐藏其他",
    "Camera": "相机",
    "Track camera": "跟踪相机",
    "FOV (\u00b0)": "视场角 (\u00b0)",
    "Frames": "坐标系",
    "Body": "本体",
    "Geom": "几何体",
    "Site": "位置点",
    "Frame length": "坐标系长度",
    "Frame width": "坐标系宽度",
    "Contact points": "接触点",
    "Contact forces": "接触力",
    "Width": "宽度",
    "Height": "高度",
    "Force scale": "力缩放",
    "Split normal/friction": "分离法向/摩擦",
    "Inertia": "惯量",
    "Shape": "形状",
    "Box": "长方体",
    "Ellipsoid": "椭球",
    "Color": "颜色",
    "Opacity": "不透明度",
    "Joints": "关节",
    "Actuators": "执行器",
    "Auto-connect": "自动关联",
    "Hide meshes": "隐藏网格",
    "Convex hull": "凸包",
    "Hide": "隐藏",
    "Geoms": "几何体",
    "Sites": "位置点",
    "Scale": "缩放",
    "Rewards": "奖励",
    "Metrics": "指标",
    "Length": "长度",
    "Tendons": "肌腱",
    "COM": "质心",
    "Constraints": "约束",
    "Global scale": "整体缩放",
    "Depth Scale": "深度缩放",
    "Frustum": "视锥",
    "Steps": "步数",
}

_ZH_TEXT = (
    ("Showing terms for environment #", "正在显示环境 "),
)

_ZH_HTML = (
    ("<strong>Status:</strong>", "<strong>状态：</strong>"),
    ("<strong>Steps:</strong>", "<strong>步数：</strong>"),
    ("<strong>Speed:</strong>", "<strong>速度：</strong>"),
    ("<strong>Target RT:</strong>", "<strong>目标实时：</strong>"),
    ("<strong>Actual RT:</strong>", "<strong>实际实时：</strong>"),
    ("<strong>Error:</strong>", "<strong>错误：</strong>"),
    ("<strong>Source:</strong>", "<strong>来源：</strong>"),
    ("<strong>Run:</strong>", "<strong>运行：</strong>"),
    ("Waiting for data", "等待数据"),
    ("状态：</strong> Paused", "状态：</strong> 已暂停"),
    ("状态：</strong> Running", "状态：</strong> 运行中"),
    ("[CAPPED]", "[受限]"),
)


def _zh(text):
    return _ZH_LABELS.get(text, text)


def _zh_text(text):
    for old, new in _ZH_TEXT:
        text = text.replace(old, new)
    return text


def _zh_html(text):
    if not text:
        return text
    for old, new in _ZH_HTML:
        text = text.replace(old, new)
    return text




_ZH_HINTS = (
    ("Select environment", "选择环境"),
    ("Show only the selected environment.", "仅显示所选环境。"),
    ("Keep tracked body centered.", "保持被跟踪物体居中。"),
    ("Vertical FOV in degrees.", "垂直视场角（度）。"),
    ("Show debug arrows and ghost meshes.", "显示调试箭头与幽灵网格。"),
    ("Show debug visualization for all environments.", "为所有环境显示调试可视化。"),
    ("Term name must contain this string", "名称需包含该字符串"),
    ("Color: ", "颜色："),
    ("Show/hide geometry in group ", "显示/隐藏几何体组 "),
    ("Show/hide sites in group ", "显示/隐藏位置点组 "),
)


def _zh_hint(text):
    if not text:
        return text
    for old, new in _ZH_HINTS:
        text = text.replace(old, new)
    return text


def _zh_cam_label(label):
    for suffix, zh in (("_segmentation", "_分割"), ("_depth", "_深度"), ("_rgb", "_彩色")):
        if label.endswith(suffix):
            return label[: -len(suffix)] + zh
    return label


def _zh_options(options):
    """Translate option display labels but keep the underlying *value* English
    so mjlab/mjviser event handlers (which compare ``event.target.value``
    against English strings) keep working."""
    out = []
    for option in options:
        if isinstance(option, str):
            out.append((_zh(option), option))
        else:
            label, value = option
            out.append((_zh(label), value))
    return out


class _ZhHandle:
    """Proxy over a viser GUI handle that forwards every attribute and
    translates dynamic ``.label`` / ``.content`` writes (Play/Pause button and
    the Info status panel)."""

    __slots__ = ("_handle",)

    def __init__(self, handle):
        object.__setattr__(self, "_handle", handle)

    def __getattr__(self, name):
        return getattr(self._handle, name)

    def __setattr__(self, name, value):
        if name == "label":
            self._handle.label = _zh(value)
        elif name == "content":
            self._handle.content = _zh_text(_zh_html(value))
        elif name == "options":
            self._handle.options = _zh_options(value)
        else:
            setattr(self._handle, name, value)

    def __enter__(self):
        self._handle.__enter__()
        return self

    def __exit__(self, *exc):
        return self._handle.__exit__(*exc)


class _ZhTabGroup:
    __slots__ = ("_handle",)

    def __init__(self, handle):
        object.__setattr__(self, "_handle", handle)

    def add_tab(self, label, *args, **kwargs):
        return _ZhHandle(self._handle.add_tab(_zh(label), *args, **kwargs))

    def __getattr__(self, name):
        return getattr(self._handle, name)


class _ZhOptionHandle(_ZhHandle):
    """Viser accepts strings only; map Chinese UI values back for callbacks."""

    __slots__ = ("_options",)

    def __init__(self, handle, options):
        super().__init__(handle)
        object.__setattr__(self, "_options", _zh_options(options))

    def __getattr__(self, name):
        if name == "value":
            value = self._handle.value
            return dict(self._options).get(value, value)
        if name == "options":
            return tuple(value for _, value in self._options)
        if name in ("on_click", "on_update"):
            def register(callback):
                from dataclasses import replace
                def translated(event):
                    return callback(replace(event, target=self))
                getattr(self._handle, name)(translated)
                return callback
            return register
        return super().__getattr__(name)

    def __setattr__(self, name, value):
        if name == "value":
            self._handle.value = {v: k for k, v in self._options}.get(value, value)
        elif name == "options":
            object.__setattr__(self, "_options", _zh_options(value))
            self._handle.options = [label for label, _ in self._options]
        else:
            super().__setattr__(name, value)


def _install_zh_gui(server):
    """Wrap ``server.gui.add_*`` so labels render in Chinese at runtime, with
    no edits to installed packages (mjlab / mjviser / viser)."""
    gui = server.gui

    def _wrap_label(method_name):
        original = getattr(gui, method_name, None)
        if original is None:
            return

        def wrapped(label, *args, **kwargs):
            if "hint" in kwargs:
                kwargs["hint"] = _zh_hint(kwargs["hint"])
            return _ZhHandle(original(_zh(label), *args, **kwargs))

        setattr(gui, method_name, wrapped)

    for method in ("add_folder", "add_button", "add_checkbox", "add_slider",
                   "add_text", "add_rgb"):
        _wrap_label(method)

    original_dropdown = getattr(gui, "add_dropdown", None)
    if original_dropdown is not None:
        def dropdown(label, options=None, *args, **kwargs):
            if options is None:
                options = kwargs.pop("options", [])
            if "hint" in kwargs:
                kwargs["hint"] = _zh_hint(kwargs["hint"])
            pairs = _zh_options(options)
            if "initial_value" in kwargs:
                kwargs["initial_value"] = {v: k for k, v in pairs}.get(kwargs["initial_value"], kwargs["initial_value"])
            return _ZhOptionHandle(original_dropdown(_zh(label), [k for k, _ in pairs], *args, **kwargs), options)
        gui.add_dropdown = dropdown

    original_group = getattr(gui, "add_button_group", None)
    if original_group is not None:
        def button_group(label, options=None, *args, **kwargs):
            if options is None:
                options = kwargs.pop("options", [])
            if "hint" in kwargs:
                kwargs["hint"] = _zh_hint(kwargs["hint"])
            return _ZhOptionHandle(original_group(_zh(label), [k for k, _ in _zh_options(options)], *args, **kwargs), options)
        gui.add_button_group = button_group

    original_markdown = getattr(gui, "add_markdown", None)
    if original_markdown is not None:
        gui.add_markdown = lambda text, *a, **k: _ZhHandle(original_markdown(_zh_text(text), *a, **k))

    original_html = getattr(gui, "add_html", None)
    if original_html is not None:
        gui.add_html = lambda html, *a, **k: _ZhHandle(original_html(_zh_html(html), *a, **k))

    original_tab_group = getattr(gui, "add_tab_group", None)
    if original_tab_group is not None:
        gui.add_tab_group = lambda *a, **k: _ZhTabGroup(original_tab_group(*a, **k))

    original_image = getattr(gui, "add_image", None)
    if original_image is not None:
        def image(*args, **kwargs):
            label = kwargs.get("label")
            if isinstance(label, str):
                kwargs["label"] = _zh_cam_label(label)
            return _ZhHandle(original_image(*args, **kwargs))
        gui.add_image = image

    original_uplot = getattr(gui, "add_uplot", None)
    if original_uplot is not None:
        def uplot(*args, **kwargs):
            series = kwargs.get("series")
            if series is not None:
                kwargs["series"] = tuple(
                    dict(s, label=_zh(s.get("label", "")))
                    if isinstance(s, dict) else s
                    for s in series
                )
            title = kwargs.get("title")
            if isinstance(title, str):
                kwargs["title"] = _zh(title)
            return _ZhHandle(original_uplot(*args, **kwargs))
        gui.add_uplot = uplot


def configure_command_gui(command, task):
    """Keep fixed command axes out of the upstream variable-range sliders."""
    def create_gui(name, server, get_env_idx, on_change=None, request_action=None):
        from types import SimpleNamespace
        with server.gui.add_folder(name.capitalize()):
            if task == "balance":
                server.gui.add_markdown("**平衡策略** —— 速度指令固定为零。")
                return
            server.gui.add_markdown("机体坐标系速度指令：**前进速度 linear.x（米/秒）** 前后移动，"
                                    "**转向角速度 angular.z（弧度/秒）** 转向，正值左转。")
            enabled = server.gui.add_checkbox("手动速度指令", initial_value=True)
            sliders = []
            for attr, disp in (("lin_vel_x", "前进速度 (m/s)"),
                               ("lin_vel_y", "横向速度 (m/s)"),
                               ("ang_vel_z", "转向角速度 (rad/s)")):
                low, high = getattr(command.cfg.ranges, attr)
                if low == high:
                    server.gui.add_markdown(f"{disp}: **{low:g}**（固定）")
                    sliders.append(SimpleNamespace(value=low))
                else:
                    sliders.append(server.gui.add_slider(disp, min=low, max=high,
                                   step=.01, initial_value=max(low, min(0., high))))
            zero = server.gui.add_button("停止 / 归零指令")
            @zero.on_click
            def reset_command(_):
                for slider in sliders:
                    slider.value = 0.
            command._joystick_enabled = enabled
            command._joystick_sliders = sliders
            command._joystick_get_env_idx = get_env_idx
    command.create_gui = create_gui


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path)
    ap.add_argument("--viewer", choices=["none", "viser", "native"], default="viser")
    ap.add_argument("--envs", type=int, default=16)
    ap.add_argument("--video", type=Path)
    args = ap.parse_args()
    torch.set_num_threads(4)
    meta = json.loads((args.run / "run.json").read_text(encoding="utf-8"))
    from infantry_rl.model_selection import select_model
    model_path = select_model(meta.get('model_variant','original'))
    if 'model_selection_sha256' in meta and hashlib.sha256(Path(__file__).with_name('model_selection.py').read_bytes()).hexdigest() != meta['model_selection_sha256']:
        raise ValueError('Model selection code changed since training')
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != meta["model_sha256"]:
        raise ValueError("Model changed since training; use the matching saved source")
    if hashlib.sha256(Path(__file__).with_name("mjlab_task.py").read_bytes()).hexdigest() != meta["task_sha256"]:
        raise ValueError("Task changed since training; use the matching saved source")
    checkpoint = args.checkpoint
    if checkpoint is None:
        choices = [(int(re.fullmatch(r"model_(\d+)\.pt", p.name)[1]), p)
                   for p in args.run.glob("model_*.pt") if re.fullmatch(r"model_(\d+)\.pt", p.name)]
        if not choices:
            raise FileNotFoundError("No checkpoint has been saved yet")
        checkpoint = max(choices)[1]
    task_factory = make_cfg
    if meta['task'] == 'cmd_vel':
        from infantry_rl.mjlab_cmdvel_task import make_cfg as task_factory
        if hashlib.sha256(Path(__file__).with_name('mjlab_cmdvel_task.py').read_bytes()).hexdigest() != meta['cmdvel_sha256']:
            raise ValueError('Command task changed since training')
    elif meta['task'] == 'cmd_vel_v3':
        from infantry_rl.mjlab_cmdvel_v3_task import make_cfg as task_factory
        if hashlib.sha256(Path(__file__).with_name('mjlab_cmdvel_task.py').read_bytes()).hexdigest() != meta['cmdvel_sha256']:
            raise ValueError('Command v2 dependency changed since training')
        if hashlib.sha256(Path(__file__).with_name('mjlab_cmdvel_v3_task.py').read_bytes()).hexdigest() != meta['cmdvel_v3_sha256']:
            raise ValueError('Command v3 task changed since training')
    elif meta['task'] in ('cmd_vel_v4', 'cmd_vel_v4_fast'):
        from infantry_rl.mjlab_cmdvel_v4_task import make_cfg as task_factory
        for key, filename in [('cmdvel_sha256','mjlab_cmdvel_task.py'), ('cmdvel_v3_sha256','mjlab_cmdvel_v3_task.py'), ('cmdvel_v4_sha256','mjlab_cmdvel_v4_task.py')]:
            if hashlib.sha256(Path(__file__).with_name(filename).read_bytes()).hexdigest() != meta[key]:
                raise ValueError('Command v4 dependency changed since training: '+filename)
        if meta['task'] == 'cmd_vel_v4_fast':
            from infantry_rl.mjlab_cmdvel_v4_fast_task import make_cfg as task_factory
            if hashlib.sha256(Path(__file__).with_name('mjlab_cmdvel_v4_fast_task.py').read_bytes()).hexdigest()!=meta['cmdvel_v4_fast_sha256']:
                raise ValueError('Fast task changed since training')
    cfg = task_factory(args.envs if args.viewer == "none" else 1, meta["task"], play=args.viewer != "none")
    from infantry_rl.stop_refinement import apply_from_metadata
    cfg=apply_from_metadata(cfg,meta)
    try:
        cfg.commands["twist"].viz.z_offset = ARROW_Z_OFFSET
    except (KeyError, AttributeError, TypeError):
        pass
    cfg.seed = 2000
    env = ManagerBasedRlEnv(cfg, device="cuda:0", render_mode="rgb_array" if args.video else None)
    if args.viewer != "none":
        _apply_viewer_palette(env)
    vec = RslRlVecEnvWrapper(env, clip_actions=1.)
    runner = MjlabOnPolicyRunner(vec, asdict(runner_cfg()), device="cuda:0")
    runner.load(str(checkpoint), load_cfg={"actor": True}, map_location="cuda:0")
    policy = runner.get_inference_policy(device="cuda:0")
    print(f"Loaded checkpoint: {checkpoint.resolve()}", flush=True)
    keyboard_server = None
    try:
        if args.viewer == "none":
            obs, _ = vec.reset()
            active = torch.ones(env.num_envs, dtype=torch.bool, device="cuda:0")
            durations = torch.zeros(env.num_envs, device="cuda:0")
            success = torch.zeros_like(active)
            tilt_sum = torch.zeros_like(durations)
            velocity_sum = torch.zeros_like(durations)
            reasons = [""] * env.num_envs
            frames = []
            with torch.inference_mode():
                for i in range(env.max_episode_length):
                    tilt_sum += active * torch.rad2deg(tilt_angle(env))
                    velocity_sum += active * velocity_error(env)
                    durations += active * env.step_dt
                    obs, _, dones, _ = vec.step(policy(obs))
                    ending = active & dones.bool()
                    success |= ending & env.reset_time_outs & ~env.reset_terminated
                    for name in env.termination_manager.active_terms:
                        mask = ending & env.termination_manager.get_term(name)
                        for j in mask.nonzero().flatten().tolist():
                            if name != "time_out" or not reasons[j]:
                                reasons[j] = name
                    if args.video and i % 2 == 0:
                        frames.append(env.render())
                    active &= ~ending
                    if not active.any():
                        break
            report = dict(checkpoint=str(checkpoint), seed=2000, episodes=env.num_envs,
                          survival_rate=float(success.float().mean()), mean_duration=float(durations.mean()),
                          durations=durations.tolist(), reasons=reasons,
                          mean_tilt_deg=(tilt_sum / (durations / env.step_dt).clamp_min(1)).tolist(),
                          velocity_mae=(velocity_sum / (durations / env.step_dt).clamp_min(1)).tolist())
            (args.run / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            print("EVALUATION " + json.dumps(report), flush=True)
            if args.video:
                import imageio.v2 as imageio
                args.video.parent.mkdir(parents=True, exist_ok=True)
                imageio.mimsave(args.video, frames, fps=25, macro_block_size=8)
        else:
            from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
            if args.viewer == "native":
                NativeMujocoViewer(vec, policy).run()
            else:
                import viser
                if meta['task'] in ('cmd_vel_v4','cmd_vel_v4_fast'):
                    from infantry_rl.v4_keyboard import start_keyboard_server
                    from infantry_rl.v4_controls import configure_web_controls
                    fast=meta['task']=='cmd_vel_v4_fast'
                    from infantry_rl.fast_control import map_legacy_keyboard
                    keyboard, keyboard_server = start_keyboard_server(command_mapper=map_legacy_keyboard if fast else None)
                    command = env.command_manager.get_term('twist')
                    configure_web_controls(command, keyboard, fast=fast)
                    print('KEYBOARD http://127.0.0.1:8081 (forward both 8080 and 8081)', flush=True)
                else:
                    configure_command_gui(env.command_manager.get_term("twist"), meta["task"])
                server = viser.ViserServer(host="127.0.0.1", port=8080)
                _install_zh_gui(server)
                ViserPlayViewer(vec, policy, viser_server=server).run()
    finally:
        if keyboard_server is not None:
            keyboard_server.shutdown()
            keyboard_server.server_close()
        vec.close()


if __name__ == "__main__":
    main()
