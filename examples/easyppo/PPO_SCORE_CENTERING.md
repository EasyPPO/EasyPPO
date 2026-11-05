# Miles 官方 Score Centering + critic GAE

默认 launcher 直接调用仓库已有的 `--loss-type score_centering`，并使用
`--advantage-estimator ppo` 保留独立 critic、GAE 和两种 critic memory。
异步 importance correction 默认打开官方 TIS：`--score-centering-is tis`，上限为 2。
没有自定义 PPO-SC estimator、旧候选概率重算、三分布校正或双 tail 模型。

## 直接使用的官方开关

| 用途 | Miles 参数 | 当前 recipe |
|---|---|---|
| critic 和 GAE | `--advantage-estimator ppo` | 开启，32 prompts ×16条轨迹 |
| 官方 SC loss | `--loss-type score_centering` | 开启 |
| 实际 sampler 概率 | `--use-rollout-logprobs` | 开启，同时跳过独立 actor scoring |
| 候选记录 | `--rollout-top-logprobs-num 128` | 在真实生成调用中记录 |
| SC importance weighting | `--score-centering-is tis` | 官方 TIS，上限 2 |
| TIS 上限 | `--score-centering-tis-clip 2` | 不叠加通用 `--use-tis` |
| 推理 LM head 精度 | `--sglang-enable-fp32-lm-head` | 显式开启，与新版官方 SC recipe 一致 |
| 概率计算 chunk | `--log-probs-chunk-size 128` | 沿用官方分块实现 |
| loss 重算 | `--recompute-loss-function` | 开启，减少反向保存量 |

[Miles 官方 SC 文档](https://github.com/radixark/miles/blob/main/docs/examples/infra-features/score-centering.md)
把 SC 定义为独立 REINFORCE-style actor loss，PPO clipping 参数不生效。
本配置仍使用 PPO 的 critic/GAE 训练流程；**actor 使用官方 SC 目标，不是 clipped PPO + SC**。
critic 仍接收 `value_clip` 和 `value_loss_coef`；回归系数实际缩放loss，
每轮四个critic mini-batches，共用更新前的整批V_old；首批等价MSE，后续批可能触发clipping。
参数及运行条件见 [主说明](ASYNC_PPO.md)。

## 本 fork 的最小接入

原有 `score_centering_loss.py` 和 `score_centering/` 数学实现保持原样。
它们接受逐 token 的 detached advantages，不依赖组内 reward 或 group size。
本 fork 仅将 `validate_score_centering_args` 的 estimator 白名单从 GRPO 扩为 GRPO/PPO，
使已有 critic GAE 可以进入同一个官方 SC loss；没有放开其它未验证的 estimator。
因此“SC + critic/GAE 的组合接入”是本 fork 的变更，不能称为官方已经跑过的训练 recipe。

已有 PPO 配置入口仍会开启 critic、要求 Megatron、拒绝 reward-level KL；
这些兼容约束和 critic-first、pre-update values 的顺序都保留。
PPO advantage 不进行 GRPO 的组内 reward 归一化，本 recipe 也不做 advantage whitening。
不传 `--skip-actor-forward-only`：`--use-rollout-logprobs` 已让 actor 跳过该 scoring，
同时照常接收 critic values 并计算 GAE。

## 旧策略分母与 importance sampling

Miles 已有以下选择，不需要重新实现：

- 普通 `policy_loss` + `--keep-old-actor`：保留生成版本的旧权重，在 trainer 上重算 PPO 分母。
- 普通 `policy_loss` + `--use-rollout-logprobs`：用生成时记录的 sampler log-prob 作 PPO 分母。
- 普通 `policy_loss` + `--use-tis`：调用已有的 train/sampler mismatch correction。
- 官方 `score_centering` + `--score-centering-is none|tis|mis`：使用其自带的权重和中心化逻辑。

默认 SC 使用当前 trainer `p` 和实际 sampler `q`，不需要旧 trainer 候选分布或 old actor 副本。
`tis` 为 `min(p/q, C)`，`mis` 在 `[low, high]` 内保留 `p/q`，其它位置权重为零；
`none` 的权重为 1，关闭的是 IS weighting，SC 仍然开启。
这些权重和对应的 centering 系数由原有实现一起计算并 detach。
通用 `--use-tis` 与 SC 的 IS 开关是不同入口，官方 validator 禁止叠加。

关闭 SC 时，launcher 恢复原始 `policy_loss + --keep-old-actor` 基线，clip low/high为 `.2/.28`。
它不请求候选，不开启 SC 专用 FP32 LM head 或 SC loss 重算。

## 使用与参数

```bash
# 默认：官方 SC + TIS(C=2)，critic/GAE 保留。
MILES_SCRIPT_EXTERNAL_RAY=1 python examples/easyppo/run_qwen3_5_27b_async_ppo.py \
  --action train --dataset frontiersmith --critic-context-mode full_detach \
  --model-dir /shared/models --data-dir /shared/data --output-dir /shared/runs

# 关闭 importance weighting，保留官方 SC：
# 在上述命令上加 --score-centering-is none

# 官方 MIS 对照：
# 加 --score-centering-is mis --score-centering-mis-low 0.5 --score-centering-mis-high 5

# 原始 clipped PPO 对照：
# 加 --no-score-centering
```

| Launcher 参数 | 默认 | 作用 |
|---|---|---|
| `score-centering` | true | 使用官方 SC；关闭恢复原 clipped PPO |
| `score-centering-candidates` | 128 | 每 response 位置保存的候选数 |
| `score-centering-is` | tis | 官方 none/tis/mis 权重选择 |
| `score-centering-tis-clip` | 2 | TIS 权重上限，仅 tis 模式使用 |
| `score-centering-mis-low / score-centering-mis-high` | .5 / 5 | MIS 区间，仅 mis 模式使用 |
| `score-centering-chunk-size` | 128 | 官方 candidate log-prob 的 token chunk |
| `score-centering-recompute-loss` | true | 官方 loss checkpointing |

两种 memory 和两个数据配置均可使用。terminal 训练采样仍固定 temperature 1、top-p 1、top-k -1；
评测也使用 `1/1/-1`，固定16题、每题1条，且不请求候选。
旧基线没有自动启用 FP32 LM head，因此本 launcher 使用已有 SGLang flag 显式开启；
无需回迁新版 Miles 的自动开关逻辑。该精度选项可能增加推理计算和显存成本。

## 候选链路、成本与验证边界

terminal 的 Completion/Trace 保存真实生成调用的 int32 IDs 与 FP32 log-probs；
summary 和 action 均记录，工具和模板间隔填 `-1/-inf`，EOS token-in 与 compaction 保持逐行对齐。
Harbor 将候选无损保存到 `easyppo-topk.npz`，JSON 记录引用与 shape。
critic 专用转换关闭 actor 候选和 support replay，避免对 value regression 错误施加候选要求。

官方 SC 使用 top-K 单 tail 近似，假设尾部 `q` 与 `p` 成比例；任意真实 tail 下不保证无偏。
K128 的两个候选数组约为每 response 位置 1 KiB，包括工具间隔。
512 条轨迹各有 32k response 位置时，仅这两个数组约 16 GiB，尚未计 transport 和日志开销。
若每条episode生成达到120000，则512条的生成位置候选数组已约58.6GiB；
工具间隔会继续增大response区域，实际大小按response positions统计，不能只按单次32k窗口估算。
现在不再保存旧 trainer 的候选概率和候选 ID 快照。

观察官方指标 `sc_correction`、`sc_train_head_mass`、`sc_rollout_head_mass`、`sc_tail_ratio`、
`sc_importance_weight`、`train_rollout_kl`，同时观察 reward、critic explained variance 和 grad norm。
验证使用官方既有 SC 数学/分布式 tests，并补 terminal→conversion→critic values→GAE→官方 loss/backward 的集成测试。
CPU eager 检查不能证明训练收敛或真实 GPU 吞吐；原生 Megatron/Transformer Engine、真实 offload、
Docker judge 和 64/32×B200 端到端训练仍需目标集群验证。

验证结果和运行限制见 [主文档](ASYNC_PPO.md#验证范围)。
