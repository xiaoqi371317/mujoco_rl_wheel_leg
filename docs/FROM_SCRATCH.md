# 从随机策略走到预训练：分阶段操作

本页给出**当前发布代码**的重训路线，不要求先下载旧服务器的中间检查点。轮数参考历史预算，并非保证收敛的超参数。历史 1000/2250/3000/3250 等候选编号不能直接当作你自己的最优编号；每阶段评估后选择，再传给下一阶段。

完整历史关系见 [预训练成长路线](TRAINING_JOURNEY.md)。本次发布只包含末端三个权重，不包含早期中间权重；从零走这条路线需要实际训练出它们。已有 GPU 环境下的三段短测记录见 [lineage_validation.json](../reports/lineage_validation.json)，它只验证加载/迁移接口，不证明两轮能学会动作。

## 0. 准备并读懂一个训练输出

先完成根 README 安装，在仓库根目录、同一个 Bash 终端中执行：

```bash
conda activate mujoco_rl
export CUDA_VISIBLE_DEVICES=0  # 改成自己可用的 GPU
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
python tools/validate_release.py
```

使用发布的 fix XML 和姿态库即可，不需要先重新生成模型。改几何、惯量或姿态库后，旧检查点兼容性不再当然成立。

每个输出目录包含 `run.json`（配置、来源及哈希）、`model_*.pt`（网络与优化器）、训练日志，以及完成后的 `status.json`。**目录名是你给实验起的名字，检查点数字是迭代标签，二者都不是成绩。** 输出目录必须新建，不要覆盖已有实验。

首次只跑下述每阶段 `--envs 64 --iterations 2` 检查是否有报错；正式训练恢复表中预算。短测产生的模型只能验证下一阶段迁移能执行，不能替代正式训练的教师。

## 1. V4：从零学驾驶、转向、撑高、自转

```bash
python -m infantry_rl.mjlab_train --task cmd_vel_v4 --model fix \
  --envs 512 --iterations 2400 --out artifacts/tutorial/v4
```

这里没有 `--resume` 或 `--warm-start`，因此才是从随机权重开始。历史选的是 2399；你应回放并用 `mjlab_v4_audit --help` 查看验收方式，再选择自己的检查点。下面变量示例假设你选 2399：

```bash
V4=artifacts/tutorial/v4/model_2399.pt
python -m infantry_rl.mjlab_play --run artifacts/tutorial/v4 --viewer viser
```

验收：两轮稳定接触、方向正确、零命令能停住、不同高度和自转不倒。退出回放后再开始下一阶段。

## 2. fast：在已有驾驶上学 1.3 m/s 与更快升降

```bash
python -m infantry_rl.mjlab_train --task cmd_vel_v4_fast --model fix \
  --warm-start "$V4" --envs 512 --iterations 3000 --out artifacts/tutorial/fast
FAST=artifacts/tutorial/fast/model_2999.pt
python -m infantry_rl.mjlab_fast_audit --run artifacts/tutorial/fast --checkpoint "$FAST"
```

48 维结构兼容，热启动保留网络与归一化，重建 Adam。不要把这一步写成完整断点续训。验收 1.3 m/s、升降响应和停车，自转无需追求更快。

## 3. V5：加入起立与原地跳，48 → 56 维

```bash
python -m infantry_rl.mjlab_v5_train --source "$FAST" \
  --envs 1024 --iterations 1200 --out artifacts/tutorial/v5
V5=artifacts/tutorial/v5/model_1000.pt
python -m infantry_rl.mjlab_v5_audit --run artifacts/tutorial/v5 --checkpoint "$V5"
```

1000 是历史选择的示例，可能早于最终轮；请按自己评估的候选更改变量。新增观测包含阶段、阶段时间、目标离地高度。先起立后跳跃；必须检查真实双轮净空、腾空时间和落地站稳，不只看车体高度。

## 4. rough → rugged：逐步适应凹凸地面

```bash
python -m infantry_rl.mjlab_rough_train --source "$V5" \
  --envs 1024 --iterations 800 --out artifacts/tutorial/rough
ROUGH=artifacts/tutorial/rough/model_1799.pt
python -m infantry_rl.mjlab_rough_audit --run artifacts/tutorial/rough --checkpoint "$ROUGH"

python -m infantry_rl.mjlab_rugged_train --source "$V5" --resume "$ROUGH" \
  --envs 1024 --iterations 600 --out artifacts/tutorial/rugged
RUGGED=artifacts/tutorial/rugged/model_2250.pt
python -m infantry_rl.mjlab_rugged_audit --run artifacts/tutorial/rugged --checkpoint "$RUGGED"
```

这里给出一次执行 800 轮的简化 rough 操作；历史曾中途恢复，详见成长路线。**rugged 的 source 仍是 V5 教师，resume 才是 rough 学生状态。** 1799 的例子依赖 V5 从 1000 开始；改了源编号就要相应修改路径。

检查高难度路面上的存活和速度跟踪，也检查平地原技能。出生区是平坦的，必须驶离后才真正测试坑洼能力。

## 5. motion：行驶时撑高与跳跃，指令到 2 m/s

```bash
python -m infantry_rl.mjlab_motion_train --source "$RUGGED" --teacher "$ROUGH" \
  --envs 1024 --iterations 900 --out artifacts/tutorial/motion
MOTION=artifacts/tutorial/motion/model_3000.pt
python -m infantry_rl.mjlab_motion_audit --run artifacts/tutorial/motion --checkpoint "$MOTION"
```

保留既有策略和 Adam，旧 rough 教师约束旧技能区间。此阶段允许 F 与前进组合，跳跃不再清空速度。验收起跳前已建立速度、撑高延迟、空中速度与落地跟踪。

## 6. jumpcarry：专门保持起跳前速度

```bash
python -m infantry_rl.mjlab_jumpcarry_train --source "$MOTION" --teacher "$MOTION" \
  --envs 1024 --iterations 400 --out artifacts/tutorial/jumpcarry
CARRY=artifacts/tutorial/jumpcarry/model_3250.pt
python -m infantry_rl.mjlab_jumpcarry_audit --run artifacts/tutorial/jumpcarry --checkpoint "$CARRY"
```

奖励目标是起跳瞬间的实际水平速度，各阶段单独统计损失。历史这里选出的 3250 就是公开 `pretrained/v5_motion/model_3250.pt` 的来源。你的重训结果即使同名，权重与质量也不一定相同。

## 7. 可选：气弹簧适应 → 收腿与落地切换

这两步仍是实验分支。若目标是获得已有的移动基线，可在第 6 步停止。

```bash
python -m infantry_rl.mjlab_gas_train --source "$CARRY" --teacher "$CARRY" \
  --stage stance --envs 1024 --iterations 200 --out artifacts/tutorial/gas_adapt
ADAPT=artifacts/tutorial/gas_adapt/model_3449.pt
python -m infantry_rl.mjlab_gas_landing_train --source "$ADAPT" --teacher "$ADAPT" \
  --stage tuck --envs 1024 --iterations 300 --out artifacts/tutorial/gas_landing
```

假设每侧 200–300 N，未标定实物。使用 [训练评估指南](TRAINING.md) 中动态和长时站姿测试。切勿因为 1.5 m/s 跳跃改善就忽略站立退化。

## 执行到一半时如何决定下一步？

先检查 `status.json` 是否完成，再看评估各项，而非只看平均奖励。若出现分叉、偏航漂移或落地失败，保留当前候选与之前基线，单独分析模型/奖励/课程；不要为凑满版本链而继续增加难度。

从零训练会比使用公开权重耗时很多。想修改后段跳跃，直接设 `CARRY=pretrained/v5_motion/model_3250.pt` 从第 7 步进入，或按 [续训指南](TRAINING.md) 做有限实验即可；不必重复前六阶段。
