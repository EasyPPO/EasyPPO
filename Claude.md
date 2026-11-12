# Qwen3.5-27B 同步 PPO：FrontierSmith 启动要求

入口：[`run_qwen3_5_27b_sync_frontiersmith.py`](examples/easyppo/run_qwen3_5_27b_sync_frontiersmith.py)。按环境、judge、模型准备、训练四步执行。先查看实际集群的镜像、调度器、存储和服务地址，沿用已有配置；下面的命令是参数模板，路径和部署方式由执行者按环境调整。

## 1. 环境

使用兼容本分支的 Miles GPU 镜像。镜像已有的依赖无需重复安装；构建细节见 [installation.md](docs/getting-started/installation.md)。确保运行环境使用本分支代码；只有镜像尚未包含本项目及 terminal 依赖时，才安装：

```bash
git clone --branch miles https://github.com/dyxg/EasyPPO.git EasyPPO
cd EasyPPO
python -m pip install -e '.[agentic,terminal]'
```

当前实验不需要初始化 `third_party/compaction`。这里的 actor 文本摘要由 terminal 实现，外部 KV compaction 包只用于单独的 KV 压缩实验。

默认申请 4 节点 × 8 张 B200。同一批 32 张卡依次执行 rollout、critic 训练和 actor 训练，脚本已开启 colocate 和 offload；训练 TP4/DP8，推理 8 个 TP4 engine。按实际分配设置 `--num-nodes`、`--num-gpus-per-node`，并为 offload 和 Harbor 任务安排足够的 CPU/内存。

执行前确定以下路径和服务，本文用环境变量表示：

- `MODEL_DIR`、`DATA_DIR`、`RUN_DIR`：模型、准备后的任务数据、训练输出。由平台挂载或分发到所需进程，不限定共享盘、目录名或部署用户；当前任务的 Docker bind mount 源路径须在对应 daemon 主机上可访问。
- `JUDGE_URL`：Harbor 任务容器能访问的 judge 服务 URL。
- 集群的 Ray 作业提交地址和训练通信配置：优先使用平台提供的配置，不自行改写网络接口或重建集群。

Harbor 当前使用 Docker 环境，需要任务执行进程能使用相应 Docker daemon；具体接入由平台决定，不要求固定挂载 `/var/run/docker.sock`。Megatron 不在脚本默认位置时，通过 `--megatron-path` 指定实际位置。

## 2. judge 部署

仓库 `Frontier-CS/algorithmic/` 已内嵌 judge 和合并题库：200 道 FrontierSmith 用于训练，180 道 FCS 用于评测。两边已公布的 10 道重叠题保留并记录在清单中。

所有 Harbor trial 共用一个 judge 服务。可以复用兼容的已有服务，也可以在独立 CPU 机器或平台分配的服务环境部署。官方 Compose 启动方式如下，在拥有本分支代码的 judge 环境中执行：

```bash
cd Frontier-CS/algorithmic
export FRONTIERCS_JUDGE_BIND_IP=0.0.0.0
docker compose up -d --build
curl --fail http://127.0.0.1:8081/problems
```

Compose 需要 privileged 容器，挂载本地 `problems/`、`data/`、`submissions/`。若平台使用其他服务编排方式，按这份 Compose 的镜像、挂载和权限要求部署。judge 不需要训练集群的模型、共享盘或 Ray；并发容量按 CPU/内存调整 `JUDGE_WORKERS` 和 `GJ_PARALLELISM`。

`FRONTIERCS_JUDGE_BIND_IP` 是服务端监听地址；`JUDGE_URL` 使用任务容器可达的服务 DNS 或地址，不能填 `0.0.0.0`。API 无认证，应使用平台内部网络或访问控制。

## 3. 模型准备

在集群分配的 GPU 作业环境中，从仓库根目录执行。路径变量须已按实际环境设置：

```bash
python examples/easyppo/run_qwen3_5_27b_sync_frontiersmith.py \
  --action prepare \
  --frontier-judge-url "$JUDGE_URL" \
  --model-dir "$MODEL_DIR" --data-dir "$DATA_DIR" --output-dir "$RUN_DIR"
```

此步骤准备固定 revision 的 `Qwen/Qwen3.5-27B`、Harbor 任务和 JSONL，并转换 Megatron checkpoint。模型产物为 `$MODEL_DIR/Qwen3.5-27B` 和 `$MODEL_DIR/Qwen3.5-27B_torch_dist`；已有兼容产物可直接复用。转换需要 GPU。judge URL 和超时写入任务配置，变更后需重新准备数据。

## 4. 正式训练

优先通过集群现有的作业入口运行。平台已提供 Ray 集群时设置：

```bash
export MILES_SCRIPT_EXTERNAL_RAY=1
# RAY_ADDRESS 使用平台提供的 Ray Jobs HTTP 地址；在 head 本地运行可沿用默认地址。
# MASTER_ADDR 使用训练节点可达的 head 地址，沿用平台设置。
```

此模式不启动或停止 Ray，也不通过 SSH 加入节点。当前 Ray backend 仍会清理提交环境中的旧训练进程，应在本实验的作业容器/进程环境内运行。代码和准备后的模型、任务数据需在执行端可用。

仅在平台没有管理 Ray、且分配的环境支持 SSH 时，使用脚本已有的自动启动方式：不设置 `MILES_SCRIPT_EXTERNAL_RAY`，提供实际 `MASTER_ADDR` 和 `--ray-hostfile`。这个可选路径使用 root SSH；不是训练本身的要求。其他平台后端按 Miles 对应部署配置接入。

```bash
python examples/easyppo/run_qwen3_5_27b_sync_frontiersmith.py \
  --action train \
  --run-id sync-fs-full-detach-001 \
  --critic-context-mode full_detach \
  --model-dir "$MODEL_DIR" --data-dir "$DATA_DIR" --output-dir "$RUN_DIR"
```

节点数按实际资源覆盖。Harbor 并发按任务宿主机和 judge 容量设置 `--sglang-server-concurrency`；默认每 engine 为 128，设置为 1 时默认 8 个 engine 最多运行 8 个 trial。

默认训练 200 轮，每轮 32 组 × 16 个 episode；前 30 轮只训练 critic，之后每轮 critic 更新 4 次、actor 更新 1 次。初始阶段、每 10 轮和最后一轮进行评测；完整评测为 180 题 × 5 次。需要 `actor-summary` 时更换 context mode 和 run ID。

输出在 `$RUN_DIR/<run-id>/` 的 `checkpoints/`、`checkpoints_critic/` 和 `trials/`。恢复时保持 run ID、context mode、拓扑及训练参数一致；新实验使用新的 run ID。
