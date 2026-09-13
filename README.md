# MuJoCo 轮腿机器人强化学习

这个项目记录了一台闭链轮腿小车的训练过程：先学站稳和驾驶，再学起立、跳跃、坑洼路面移动，最后尝试行进中收腿跳跃和气弹簧助力。

目前建议先用 V5 保速跳跃策略。气弹簧版有些跳跃表现更好，但静止和制动退步了，还需要继续改。

## 先跑起来

需要 Linux、NVIDIA GPU 和 Python 3.12。依赖版本见 [安装说明](docs/QUICKSTART.md)。

```bash
git clone https://github.com/xiaoqi371317/mujoco_rl_wheel_leg.git
cd mujoco_rl_wheel_leg
conda create -n mujoco_rl python=3.12 -y
conda activate mujoco_rl
python -m pip install -r infantry_rl/requirements.txt
python tools/validate_release.py
CUDA_VISIBLE_DEVICES=0 python -m infantry_rl.mjlab_jumpcarry_play --run pretrained/v5_motion --checkpoint pretrained/v5_motion/model_3250.pt
```

在服务器上运行时，在自己电脑另开终端转发端口：

```bash
ssh -N -L 8080:127.0.0.1:8080 YOUR_USER@YOUR_SERVER
```

打开 `http://127.0.0.1:8080`，用网页滑块控制速度和转向，勾选 F 撑高、R 自转，点击按钮跳跃。这里的 F/R 是控件标签，不是终端快捷键。摔倒后点击重置环境。

## 这些权重怎么练出来的

训练顺序是：V4 驾驶 → 1.3 m/s 与快速升降 → V5 起立和原地跳 → 轻坑洼 → 更崎岖地面 → 2 m/s 与行进动作 → 保速跳跃。

公开的 `v5_motion/model_3250.pt` 取自最后的保速跳跃阶段，历史任务名是 `v5_jumpcarry`。气弹簧的两个检查点在它的基础上继续训练。

- [训练经过](docs/TRAINING_JOURNEY.md)：每阶段改了什么、用了哪个检查点，哪些尝试失败了。
- [从头训练](docs/FROM_SCRATCH.md)：按阶段执行的命令。早期中间权重未打包，需要自己训练。
- [继续训练](docs/TRAINING.md)：从现有权重恢复，以及如何评估。
- [实验结果](docs/EXPERIMENTS.md)：包括站姿分叉、慢慢转向和技能退化的问题。

## 试一下气弹簧版

```bash
CUDA_VISIBLE_DEVICES=0 python -m infantry_rl.mjlab_gas_landing_play --run pretrained/gas_landing --checkpoint pretrained/gas_landing/model_3748.pt --force 100 --tuck
```

`--force 100` 是每侧 100 N。训练采用假设的 200–300 N，尚未按实物标定。加 `--terrain v4` 可以试原台阶场地，但目前没有通过台阶越障验收。

## 文件放在哪里

`infantry_rl/` 放训练、回放和评估代码；`infantry_rl/robot/xmls/` 放模型；`infantry_V4/meshes/` 放网格和原场地；`pretrained/` 放三个检查点；`reports/` 放评估结果。

旧版本的独立入口已清理。`mjlab_cmdvel_task.py`、`mjlab_cmdvel_v3_task.py` 仍是 V4 以后的共用实现；部分 V6 task 文件也仍被气弹簧任务调用，因此保留。命令从本文和训练指南找即可，不需要按文件名逐个试。

[模型与动作接口](docs/MODEL_CARD.md) · [贡献说明](CONTRIBUTING.md) · [素材来源与许可](THIRD_PARTY_NOTICES.md)
