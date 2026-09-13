# 续训与验收

本页适合已有检查点的续训。想知道权重如何获得，先读 [预训练成长路线](TRAINING_JOURNEY.md)；想从随机策略开始，使用 [从零操作指南](FROM_SCRATCH.md)。

以下命令在仓库根目录、已激活环境中运行。先查看 GPU 占用，再选择自己可用的卡；示例的 0 只是占位选择。建议设定 `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4`，遵守所在服务器配额。

## 最小续训验证

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
python -m infantry_rl.mjlab_jumpcarry_train \
  --source pretrained/v5_motion/model_3250.pt \
  --teacher pretrained/v5_motion/model_3250.pt \
  --out artifacts/smoke_v5 --envs 64 --iterations 2 --smoke
```

输出目录必须是新目录。`--iterations` 是**追加轮数**；上述从 3250 恢复两轮，最终文件名按训练器规则为 `model_3251.pt`。`--smoke` 提前推进课程阶段，只适合冒烟验证，正式实验去掉它。

训练器恢复 actor、critic 和 Adam 状态，降低学习率并调整探索标准差与归一化计数，保留归一化统计矩；因此是带明确调整的续训，不是完全不变的断点恢复。固定 teacher 提供旧动作约束，仍不能保证旧技能不遗忘。

## 气弹簧落地实验续训

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
python -m infantry_rl.mjlab_gas_landing_train \
  --source pretrained/gas_adapt/model_3449.pt \
  --teacher pretrained/gas_adapt/model_3449.pt \
  --out artifacts/gas_landing_trial --envs 1024 --iterations 300
```

这是复现本轮续训路径的入口，结果可能因硬件、随机性而不同。若要在最新状态继续，可把 source 改为 `pretrained/gas_landing/model_3748.pt`，但先保留验收基线。当前奖励方案有站立退化，不建议单纯延长轮数当作修复。

## 评估

先用 `python -m infantry_rl.mjlab_gas_landing_audit --help` 查看参数，再执行：

```bash
CUDA_VISIBLE_DEVICES=0 python -m infantry_rl.mjlab_gas_landing_audit \
  --run pretrained/gas_landing --checkpoint pretrained/gas_landing/model_3748.pt \
  --out artifacts/eval/gas_dynamic.json
CUDA_VISIBLE_DEVICES=0 python -m infantry_rl.mjlab_gas_landing_stance_audit \
  --checkpoint pretrained/gas_landing/model_3748.pt --force 200 \
  --out artifacts/eval/stance_200.json
```

动态评估按多种速度、动作和气弹簧力组合统计。一个 case 要所有重复通过才算通过；存活率与完整成功率分开。静止评估有低姿、高姿、低姿制动、高姿制动，每类 4 次，运行 30 秒，前 5 秒不计稳态误差。

验收必须同时看：速度误差、起跳前实测速度、各阶段最低保速比例、轮子净空、落地存活、腿部对称、静止航向漂移。总奖励升高不能替代这些指标；不要将不同评估器的 42 项与 32 项通过率直接相比。

训练自动保存 `run.json`、源码哈希、模型及阶段结果。发布实验前移除绝对路径及检查点里的私有元数据，并重新生成公开校验和。请勿提交完整工作目录或账号配置。
