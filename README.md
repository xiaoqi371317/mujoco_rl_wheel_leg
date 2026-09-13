# MuJoCo 轮腿机器人强化学习

A reproducible wheel-legged robot simulation project with PPO training, browser playback, and documented failure cases.

本项目包含左右对称修正的闭链轮腿模型、速度跟踪、撑高、起立、行进跳跃、崎岖地形，以及气弹簧辅助跳跃实验。提供三个小型检查点，可直接回放，也可恢复优化器继续训练。

**当前定位：仿真实验项目。** 建议先体验 V5 行进策略。最新气弹簧策略改善了部分行进跳跃，但静止稳定性仍未通过完整验收；它不是全面优于 V5 的成品。100 N 是测试工况，最新模型在 200–300 N 下续训，尚未按实物标定。

## 第一次来，先选你的目标

| 我想做什么 | 从哪里开始 | 是否需要训练 |
|---|---|---|
| 先看小车跑起来 | 下方“快速开始”，加载 V5 基线 | 不需要 |
| 看懂预训练是怎么得到的 | [逐阶段成长路线](docs/TRAINING_JOURNEY.md)，含来源关系图与真实训练记录 | 不需要 |
| 自己从随机权重训练 | [从零操作指南](docs/FROM_SCRATCH.md)，按阶段训练、评估、选择检查点 | 需要，耗时较长 |
| 在已有能力上继续改 | [续训与评估](docs/TRAINING.md)，先做两轮冒烟验证 | 只训练后续阶段 |

预训练主线：**对称模型 → V4 驾驶 → 1.3 m/s → 起立/原地跳 → 轻坑洼 → 更崎岖 → 2 m/s 与行进组合 → 保速跳跃 → 气弹簧适应 → 收腿/落地优化。**

公开 `v5_motion/model_3250.pt` 来自历史 **jumpcarry** 阶段；它已经历多阶段训练。后两个气弹簧检查点从它继续得到。早期中间权重未打包，复现完整前段需要实际训练；阶段命令与迁移方式见上述指南。

## 快速开始

在 Linux NVIDIA GPU 环境，从仓库根目录执行（经过验证的依赖见 [安装与排错](docs/QUICKSTART.md)）：

```bash
git clone https://github.com/xiaoqi371317/mujoco_rl_wheel_leg.git
cd mujoco_rl_wheel_leg
conda create -n mujoco_rl python=3.12 -y
conda activate mujoco_rl
python -m pip install -r infantry_rl/requirements.txt
python tools/validate_release.py
CUDA_VISIBLE_DEVICES=0 python -m infantry_rl.mjlab_jumpcarry_play \
  --run pretrained/v5_motion --checkpoint pretrained/v5_motion/model_3250.pt
```

远程服务器运行时，在**自己的电脑**另开终端建立隧道，然后打开浏览器 `http://127.0.0.1:8080`：

```bash
ssh -N -L 8080:127.0.0.1:8080 YOUR_USER@YOUR_SERVER
```

在网页面板调整前进速度、转向，勾选撑高 F、自转 R，点击“跳跃一次”。F/R 是控件标记，请使用网页控件；SSH 终端按键不会自动传给浏览器。腾空、摔倒和超时不自动复位，使用查看器的重置按钮。

## 测试 100 N 气弹簧策略

```bash
CUDA_VISIBLE_DEVICES=0 python -m infantry_rl.mjlab_gas_landing_play \
  --run pretrained/gas_landing --checkpoint pretrained/gas_landing/model_3748.pt \
  --force 100 --tuck
```

`--force` 表示**每侧**恒定拉力 N。使用 `--terrain v4` 查看原台阶场地，或 `--terrain v4-steps` 查看台阶模式；默认网页可切换平地和坑洼地形。具备行进跳跃不代表已学会跨过真实台阶。

## 文档导航

- [预训练成长路线与真实来源](docs/TRAINING_JOURNEY.md)
- [从随机策略开始的分阶段操作](docs/FROM_SCRATCH.md)
- [安装、回放与常见问题](docs/QUICKSTART.md)
- [训练与评估：从检查点继续](docs/TRAINING.md)
- [模型、动作与检查点说明](docs/MODEL_CARD.md)
- [实验结果与经验教训](docs/EXPERIMENTS.md)
- [贡献方式](CONTRIBUTING.md) / [素材与第三方来源](THIRD_PARTY_NOTICES.md)

代码在 `infantry_rl/`，模型在 `infantry_rl/robot/xmls/`，网格在 `infantry_V4/meshes/`，脱敏后的实验报告在 `reports/`。保留部分早期模块是因为后续任务通过继承复用它们；推荐入口以本文为准。

本发布包不包含服务器凭据、私有运维资料、完整训练日志或旧仓库历史。许可范围见 [LICENSE](LICENSE)；模型网格和预训练权重的授权状态单独说明。
