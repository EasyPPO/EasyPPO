# Qwen3.5-27B 同步 PPO：FrontierSmith 运行说明

入口：[`run_qwen3_5_27b_sync_frontiersmith.py`](examples/easyppo/run_qwen3_5_27b_sync_frontiersmith.py)。以下按环境配置、judge 部署、模型准备、正式训练的顺序执行。训练集群使用 `/shared` 共享目录，judge 部署在独立 CPU 机器上；judge 不需要挂载训练集群的共享盘。示例 head 地址为 `10.0.0.10`，judge 地址为 `10.0.0.20:8081`，都必须替换为实际机器的可达地址。

## 1. 环境配置

默认资源为 4 个节点、每节点 8 张 B200：8 张用于 actor/critic 训练，24 张用于 SGLang 推理。训练使用 CPU offload，需要为主机内存和任务容器预留容量。

在训练集群初次获取代码：

```bash
git clone --branch miles https://github.com/EasyPPO/EasyPPO.git /shared/EasyPPO
cd /shared/EasyPPO
git submodule update --init third_party/compaction
mkdir -p /shared/models /shared/data /shared/runs
```

GPU 训练环境使用 Miles NVIDIA 镜像，安装方法见 [installation.md](docs/getting-started/installation.md)，构建定义见 [docker/Dockerfile](docker/Dockerfile)。所有节点使用同一镜像 digest。

仓库 NVIDIA Dockerfile 已包含以下加速组件，不能只靠安装项目的 Python extras 来补齐：

| 组件 | Dockerfile 中的配置 |
| --- | --- |
| SGLang / Megatron-LM | Miles 对应的分支和补丁；不要直接换成任意 PyPI 版本 |
| FlashAttention 2 | 安装对应 CUDA / PyTorch / 架构的 `flash_attn` wheel |
| FlashAttention 3 | 安装 `flash_attn_3` wheel；该实现限定 Hopper，不等于所有 GPU 都使用 FA3 |
| FLA | `flash-linear-attention==0.5.2`，包含 Dockerfile 指定的补丁 |
| 其他内核 | Transformer Engine、causal-conv1d、mamba-ssm 等，按镜像构建定义安装 |

在每个 Miles worker 的环境内安装本仓库（Python 3.12+）：

```bash
cd /shared/EasyPPO
python -m pip install -e '.[agentic,terminal]'
python -m pip install 'transformers==5.12.1'
python -c "import miles, agentic_critic, transformers; print(miles.__file__); print(agentic_critic.__file__); print(transformers.__version__)"
docker version
docker compose version
```

导入路径应指向 `/shared/EasyPPO`。terminal extra 固定 Harbor 0.24.0 和 mini-swe-agent 1.17.1。不要额外安装 Harbor 的 `tinker` 依赖集合，它的 Transformers 版本要求不同。

部署时需满足：

- `/shared` 在训练集群所有 worker 和任务 Docker daemon 宿主机上具有相同绝对路径与内容。
- 容器内的 Miles worker 要挂载 `/shared` 和 `/var/run/docker.sock`，且安装 Docker CLI、Compose 插件；bind mount 的源路径由 Docker 宿主机解析。
- 独立 judge 机器安装 Docker Engine、Compose 插件和 rsync，允许运行 privileged 容器；不需要安装 Miles、CUDA、模型或加入 Ray。部署所需文件按第 2 节复制到该机器本地磁盘。
- 训练节点有足够的主机内存承载 offload，并按集群要求配置网络和 RDMA 设备。
- 默认 Megatron 路径是 `/root/Megatron-LM`；安装位置不同则给模型准备和训练命令都增加 `--megatron-path`。

## 2. judge 部署

先准备一台独立 CPU 机器，取得它从任务容器网络可达的 IP 或 DNS 名称。下面假设实际分配的地址为 `10.0.0.20`，监听端口为 `8081`。`--judge-url` 只设置客户端请求地址，不会创建机器、配置网络或远程部署服务。跨网络部署需要先打通路由或 VPN，并允许任务容器流量访问该端口。

**在训练集群的 head 上**，使用已安装的项目环境准备题目数据和 judge 配置：

```bash
cd /shared/EasyPPO
CUDA_VISIBLE_DEVICES='' python scripts/make_miles_frontiersmith_data.py \
  --source-dir /shared/data/FrontierSmith \
  --frontier-cs-source-dir /shared/data/Frontier-CS-eval \
  --tasks-dir /shared/data/harbor \
  --out /shared/data \
  --judge-url http://10.0.0.20:8081 \
  --judge-timeout 3600
```

该命令下载固定版本的题目源码，生成：

```text
/shared/data/frontiersmith.train.jsonl   # 190 道训练题
/shared/data/fcs.eval.jsonl              # 179 道评测题
/shared/data/harbor/                    # Harbor 任务目录
/shared/data/frontier-judge.compose.yaml
```

judge 同时加载这 369 道题，使用固定版本 Frontier-CS judge 以及各自固定版本的 checker/testdata。

**仍在 head 上**，把两份题目源码、Compose 文件和 judge Dockerfile 复制到独立 judge 机器。以下 SSH 账号需有目标目录的写权限，两端安装 rsync：

```bash
export JUDGE_SSH=ubuntu@10.0.0.20
ssh "$JUDGE_SSH" 'mkdir -p /shared/data /shared/EasyPPO/docker'
rsync -a /shared/data/FrontierSmith /shared/data/Frontier-CS-eval \
  "$JUDGE_SSH:/shared/data/"
rsync -a /shared/data/frontier-judge.compose.yaml "$JUDGE_SSH:/shared/data/"
rsync -a /shared/EasyPPO/docker/Dockerfile.fcs-judge-cpp \
  "$JUDGE_SSH:/shared/EasyPPO/docker/"
```

judge 上的 `/shared` 在这里是本地目录，不必是共享文件系统。当前生成的 Compose 使用绝对路径，因此复制后需保持这些路径一致；若改用其他目录，必须同时修改 Compose 的 `build.context`、`build.dockerfile` 和各题目的 `volumes[].source`。不用复制模型、checkpoint 或 Harbor 任务目录到 judge。

**SSH 登录独立 judge 机器后**，在该机器的 Docker daemon 上启动服务：

```bash
export FRONTIERCS_JUDGE_BIND_IP=0.0.0.0
export FRONTIER_JUDGE_PORT=8081
export JUDGE_WORKERS=8 GJ_PARALLELISM=8
docker compose -p easyppo-frontier-judge \
  -f /shared/data/frontier-judge.compose.yaml up -d --build --wait
curl --fail http://127.0.0.1:8081/problems
```

`FRONTIERCS_JUDGE_BIND_IP=0.0.0.0` 表示在 judge 本机所有接口发布端口；也可以改为 judge 本机实际网卡的 IP，不能填写另一台机器的 IP。任务配置中的 `--judge-url http://10.0.0.20:8081` 则是客户端访问地址，不能填 `0.0.0.0`。两者用途不同；存在 NAT 或代理时，访问地址和本机监听地址也可能不同。

所有 trial 向这一个服务提交请求，共享队列、worker pool 和 checker 缓存。原生分数按 100 归一化为 reward；任务结束不会关闭 judge。HTTP API 没有认证，需通过私网及防火墙限制访问，确保每个任务容器都能连接配置的 URL。judge 本机的 curl 成功只代表服务已启动，不代表训练侧网络已经可达。

**地址写入生成的任务配置。** 改地址或超时必须重跑数据准备；只在 `--action train` 命令中改 `--frontier-judge-url` 不会更新已有任务。默认地址 `http://host.docker.internal:8081` 只适合 judge 与任务 Docker daemon 在同一宿主机的情况，多机应显式使用 judge 私网地址。

`--judge-timeout 3600` 包含排队时间；Harbor verifier 的总超时额外增加 90 秒。判题拥堵时需结合 CPU 容量调整 worker 数、生成并发和超时。

## 3. 模型准备

回到训练集群，在一个 8 卡节点的 Miles GPU 环境中执行模型下载和 checkpoint 转换：

```bash
cd /shared/EasyPPO
python examples/easyppo/run_qwen3_5_27b_sync_frontiersmith.py \
  --action prepare \
  --frontier-judge-url http://10.0.0.20:8081 \
  --frontier-judge-timeout 3600 \
  --model-dir /shared/models \
  --data-dir /shared/data \
  --output-dir /shared/runs
```

此命令会下载固定 revision 的 `Qwen/Qwen3.5-27B`，再次准备 Frontier 数据，并转换 Megatron checkpoint。应产生 `/shared/models/Qwen3.5-27B` 和 `/shared/models/Qwen3.5-27B_torch_dist`。这里的 judge 地址必须与第 2 节一致。

## 4. 正式训练

启动脚本会自动启动 Ray head，再按 hostfile 通过 SSH 启动其余 worker，无需手动执行 `ray start`。在 head 上准备 hostfile，第一列为节点 IP，包括 head；下面是默认同步训练的 4 个节点：

```bash
cat > /shared/ray-hostfile <<'EOF'
10.0.0.10 slots=8
10.0.0.11 slots=8
10.0.0.12 slots=8
10.0.0.13 slots=8
EOF
```

hostfile 只包含本次训练的 GPU 节点，不包含独立 judge。`MASTER_ADDR` 必须是其他节点可达的 head IP，并与 hostfile 中的 head 行一致；若该行使用主机名，则另设 `MLP_WORKER_0_HOST` 为该名称，以便跳过 head。

head 需要能免密 `ssh root@<worker IP>`，且远端 SSH shell 中的 `ray` 来自已配置好的 Miles 环境。若训练环境在容器里，SSH 必须进入该容器环境，而不是宿主机的另一套 Python。各 worker 需能访问相同代码、模型和数据路径。按站点网络配置 `NCCL_SOCKET_IFNAME` / `GLOO_SOCKET_IFNAME`。

自动启动会清理这些节点上的旧 Miles/SGLang/Ray 进程，hostfile 应只列出本次独占使用的节点。若平台已经管理好 Ray 集群，设置 `MILES_SCRIPT_EXTERNAL_RAY=1`，脚本会跳过 head 启动和 SSH worker 加入；`--no-join-ray-workers` 则只关闭 SSH 加入。

从 head 提交正式训练：

```bash
cd /shared/EasyPPO
export MASTER_ADDR=10.0.0.10
python examples/easyppo/run_qwen3_5_27b_sync_frontiersmith.py \
  --action train --ray-hostfile /shared/ray-hostfile \
  --run-id sync-fs-full-detach-001 \
  --critic-context-mode full_detach \
  --model-dir /shared/models --data-dir /shared/data --output-dir /shared/runs \
  --sglang-server-concurrency 1
```

命令中的 `--sglang-server-concurrency 1` 对应默认 6 个 engine 下最多 6 个并发 trial，可按任务宿主机和 judge 的容量提高；脚本原始默认值是 128。需要 `actor-summary` 时更换 context mode 和 run ID，不要混用同一套 checkpoint。

正式默认配置会在初始阶段评测，此后每 10 轮及最后一轮评测和保存。每次完整评测为 179 题 × 5 次，即 895 个 trial；首次评测耗时不代表训练卡死。`--eval-max-tasks` 大于零表示固定抽样子集，此时不能把结果当成完整 Frontier-CS 评测。设置 `WANDB_API_KEY` 可启用启动脚本的 W&B 配置。

默认训练 200 轮，每轮 32 组 × 16 个 episode；前 30 轮仅更新 critic，之后每轮 critic 更新 4 次、actor 更新 1 次。此入口使用同步 clipped PPO，不开启 `--score-centering`。

输出目录：

```text
/shared/runs/<run-id>/
  checkpoints/          # actor
  checkpoints_critic/   # critic
  trials/               # Harbor 轨迹与 verifier 结果
```

恢复时使用相同启动脚本、数据集、context mode、run ID、训练拓扑和原有训练参数重新提交。脚本会从该 run 的 actor/critic checkpoint 目录加载。新实验使用新 run ID；不要把一次全新的训练意外指向旧 checkpoint。
