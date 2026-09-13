# 安装与回放

验证平台为 Linux、Python 3.12、NVIDIA CUDA GPU；本次发布验证记录见 `reports/release_validation.json`。Windows 原生训练、CPU 训练和其他依赖版本未作为本次发布验收目标。

锁定环境：PyTorch 2.9.1+cu130，mjlab 1.3.0，MuJoCo 3.10.0，mujoco-warp 3.8.1，warp-lang 1.12.0，rsl-rl-lib 5.0.1，NumPy 2.4.1，SciPy 1.18.0，onnxruntime 1.24.4，Viser 1.1.0。CUDA wheel 需要兼容的 NVIDIA 驱动；先确认 `nvidia-smi`，不要在共享服务器擅自更新驱动。

从根目录安装 `python -m pip install -r infantry_rl/requirements.txt`，也可使用 `conda env create -f infantry_rl/environment.yml`。首次启动 Warp 会编译内核，耗时比后续启动长。用 `python tools/validate_release.py` 验证代码语法、发布文件 SHA-256 和运行时源码匹配。

回放命令见根 README。终端会打印实际的 `PLAYBACK CHECKPOINT` / `MOTION_CHECKPOINT`；请核对路径。所有示例明确指定检查点，避免旧选择文件退回其他策略。模型与其 `run.json` 必须保存在一起。

远程回放用 SSH 转发 8080，网页打开本机 127.0.0.1:8080。非默认 SSH 端口加 `-p PORT`。Viser 只绑定服务器回环地址，不需要把服务暴露公网。多个回放共享固定 8080 会冲突，应先退出自己先前的回放进程。

## 常见问题

- **只有车，没有地面**：使用与任务对应的 `mjlab_jumpcarry_play` 或 `mjlab_gas_landing_play`，它们带有地形显示逻辑。不要用通用旧查看器加载崎岖地形策略。
- **按键无效**：使用网页速度滑块、撑高复选框和跳跃按钮。点击“停止运动”清空控制；浏览器断连也会清空命令。
- **前后方向反了**：面板正速度对应模型坐标 -X。不要只对某个电机改符号，否则会破坏动作约定。
- **源码哈希失败**：检查下载完整性和是否改动模型/任务代码。仓库禁止 Git 自动转换换行，以保留训练时字节哈希。不要直接删除检查；实验修改后应记录新的训练配置。
- **找不到网格**：保持目录结构，XML 通过相对路径引用 `infantry_V4/meshes`；不要单独移动 XML。
- **GPU 内存不足**：先用 64 个环境验证，再增至 1024。`CUDA_VISIBLE_DEVICES` 设置物理可用卡号；程序内部 `cuda:0` 指该可见卡。
- **摔倒后不复位**：这是交互回放的设计，点击重置环境。训练和批量评估仍保留终止条件。

仅加载可信来源的 PyTorch 检查点；加载器使用完整 pickle 兼容模式。本仓库提供权重校验和。
