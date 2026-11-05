# Qwen3.5-27B：async actor-critic，官方 SC/TIS

两组训练/评测配置共用一个 Python launcher：

| `--dataset` | 训练 | 评测 |
|---|---|---|
| `frontiersmith` | FrontierSmith 190题；200题中排除10个显式 parity aliases | 固定版本 Frontier-CS algorithmic 179题 |
| `tmax` | TMAX-15K，执行数据审计后保留合格任务 | 官方 TB2.1 的 89 题 |

每组分别运行 `--critic-context-mode full_detach` 和 `actor-summary`，总共四个实验。
launcher默认开启Miles已有的score-centering loss及其官方TIS权重（上限2），保留critic/GAE；
`--no-score-centering` 切换到clipped-PPO对照。
官方SC不使用PPO clipping。开关、概率链路和兼容接入见 [PPO_SCORE_CENTERING.md](PPO_SCORE_CENTERING.md)。
每次运行只有一个独立 critic；这里的“两种 memory”是两种对照设置，不是两个 critic 同时给 actor 算 advantage。
数据来源、数量、明确的训练/评测重叠过滤和 Docker 路径见 [ASYNC_DATA.md](ASYNC_DATA.md)。
默认值及其含义见下方参数表，实际传给Miles的命令见 [execute快照](../../tests/snapshots/launch_scripts/py/examples/easyppo/run_qwen3_5_27b_async_ppo.py/execute.txt)。

这是基于 [T1 v1](https://arxiv.org/html/2609.11042v1) 的 **27B dense 改编**，不是原文 122B MoE 结果的复现。
T1原文设置（本配置仅采用其中一部分）是：actor/critic LR `1e-6/1.5e-5`，Adam `(0.9,0.98)`，WD `0.1`，constant LR，policy/value clip `0.2`，gamma/lambda `1`，无 KL、无 reward/advantage normalization，TMAX batch `512`、oversample `560`，一步 async，10 步存盘。
默认SC actor目标及其TIS权重是本实验相对T1的改动；policy clip仅在关闭SC的基线生效。
本配置选择critic LR `5e-6`、Adam beta2 `.999`、WD `.01`、四步critic和inverse-std；这些值与T1的整套optimizer超参不同。

## 启动与四组实验

先使用本仓库的 Miles GPU 环境（Megatron、SGLang、Transformer Engine、CUDA kernels），
`transformers==5.12.1`。在所有 worker 安装本 checkout：

```bash
git submodule update --init third_party/compaction
python -m pip install -e '.[agentic,terminal]'
```

默认八台 8×B200 节点先加入同一个 Ray cluster，在 head 上执行下面命令。
`MILES_SCRIPT_EXTERNAL_RAY=1` 表示使用已经组好的集群。
模型、任务、FrontierSmith judge 所挂载的完整源码、输出目录都要位于所有相关主机可见的共享路径。

```bash
python examples/easyppo/run_qwen3_5_27b_async_ppo.py \
  --action prepare --dataset frontiersmith \
  --model-dir /shared/models --data-dir /shared/data --output-dir /shared/runs

python examples/easyppo/run_qwen3_5_27b_async_ppo.py \
  --action prepare --dataset tmax \
  --model-dir /shared/models --data-dir /shared/data --output-dir /shared/runs

MILES_SCRIPT_EXTERNAL_RAY=1 python examples/easyppo/run_qwen3_5_27b_async_ppo.py \
  --action train --dataset frontiersmith --critic-context-mode full_detach \
  --run-id fs-full-detach --model-dir /shared/models --data-dir /shared/data --output-dir /shared/runs
```

另外三组替换 `dataset/context-mode/run-id`：
`frontiersmith/actor-summary/fs-actor-summary`、`tmax/full_detach/tmax-full-detach`、
`tmax/actor-summary/tmax-actor-summary`。同一比较使用相同初始 checkpoint、随机 seed 和训练预算。
四组按顺序运行会复用默认64张卡；同时运行四组需要四份资源。
32卡候选可加 `--num-nodes 4 --trainer-num-nodes 1`，得到8 trainer +24 rollout。

先做小 batch 的端到端检查：在相同启动命令上加
`--rollout-batch-size 4 --over-sampling-batch-size 5 --samples-per-prompt 2 --critic-steps-per-rollout 1 --num-rollout 3 --num-critic-only-steps 1`。
它仍申请默认64张卡，用于确认跨节点通信、换入换出、容器、critic/actor update 和发布屏障。
此小预算检查不能验证长轨迹峰值；随后应使用实际长任务和生产 batch 测显存、主机内存和耗时。

## Async、版本与优化顺序

入口是已有 `train_async.py`，不是 `fully_async` producer：

1. 初始 actor 权重发布给 SGLang。
2. 当前 batch 收齐后，预取下一 batch；它和本次训练重叠。
3. critic 先用 **更新前** 的参数计算全部 `V_old`，固定 targets，再更新 critic。
4. 把 `V_old` 切回对应 actor segment；actor 用这些值计算 advantage 并更新。
5. 收齐下一 batch，取消并等待多发的尾部 Harbor trials 完成清理，再发布新权重。

一次发布间隔对应一个 outer rollout step。首批和 critic-only 阶段 actor 不变，不存在强行制造的版本差。
默认SC直接开启 `--use-rollout-logprobs`，使用实际生成调用的概率，并由官方SC TIS计算当前trainer与sampler的比值。
无需保存旧actor，也不额外重算旧候选概率。关闭SC的基线才使用 `--keep-old-actor`，
保留生成该batch的旧权重，在Megatron端重算PPO denominator。
terminal abort hook 覆盖环境命令/验证期间的 trial，并等待 teardown，避免发布屏障挂在尾部任务上。

该模型是 dense，不启用 R3。将来换 MoE 需要贯通逐层 expert capture、stitch 和旧/新 actor replay；
当前 terminal wrapper 会拒绝不支持的 replay side channels。

## Critic memory 与第二段去重

actor 和 critic 参数、value head、optimizer 和 memory 都独立。
每次 critic 读取和更新都用当时的 critic 权重重新构建 memory；不继承 actor KV，也不跨 optimizer update 复用旧 KV。

`actor-summary`：每段读取 actor 当时真正看到的输入，即 system/task + actor summary + 此段对话；
每段清空 KV、linear recurrent state 和 convolution state。上下文相同不表示共享网络。

`full_detach`：第一段正常读取；后续读取保留完整历史的 attention KV、linear recurrent state 和 convolution state。
每个 segment 边界将三类状态 detach；后段 loss 可以使用历史，但梯度不能回传到历史计算图。
段内梯度仍训练全部 critic backbone 和 value head。读取 value 的位置是目标 token 的 **前一个位置**，防止看到待预测 action。

`critic_episode()` 拼接actor segments时，第二段会重新出现system、原任务和summary。
独立保存的 `critic_segments` 用于去重：在明确的 actor compaction 边界，以简短 continuation prompt 替代这份重复 resume prompt。
summary 的原始生成仍在前一段中，summary 副本不再进入 full-detach 历史。
保留最近 turns 的 compaction 也会先保留尚未记录的新 observation。
无法证明来源的非 prefix/template break 对 full-detach 明确报错，避免猜测性去重；actor-summary 仍按 actor 真实输入工作。

actor 的 tokens、loss mask、原始 sampler logprobs 都不因 critic 去重而改变。
每段完整 response 区域（含 mask=0 的 tool/glue）有精确 span；critic values 按 span 切回 actor。
summary 生成 token 的 loss mask=1，后续输入里的 summary 副本 mask=0，工具输出/模板 glue mask=0。
生成侧对已证明的EOS边界实施token-in：下次canonical prompt匹配上一完整assistant render时，
用实际已采样的token前缀替换其重编码副本，并保留EOS后的换行、observation和generation glue。
这既保留训练tokens，也让后续采样真正条件化在相同原始token上；显式compaction清空该anchor。
没有EOS或边界不可证明时不猜测修复；未实现论文所有bounded prefix/text-space repair，不能声称完整 T1 TITO。

## Reward、GAE、loss

默认 `--reward-mode native`：TMAX/TB 使用任务 verifier 的 reward；FrontierSmith/FCS 保留原始 checker score 的尺度。
不把优化题的连续得分假装成“通过 assertion 数量”。基础设施/HTTP/verifier 错误不写假零奖励。

有可靠 per-assertion report 时可用 `--reward-mode test-count`：
读取 `--pass-count-key passing_assertions`，除以固定 `--reward-divisor 20`，不 clamp。
计数缺失/不可解析时仅允许真正的 binary outcome fallback，并把原因写入样本 metadata。
已有 TMAX tasks 不保证提供此字段；脚本没有声称替每个 verifier 实现了逐 assertion 解析。
原文按其数据分布选择的 20，也不是任意 corpus 的最优 reward scale。
数据 metadata 的 `reward_mode/reward_key` 可以覆盖 launch default；divisor 固定为整次运行的尺度。

gamma=lambda=1，reward KL=0：每个有 loss 的 actor token 的 return 是该 episode 的 terminal reward，advantage 是 `R-V_old`。
这使分段计算与本配置一致；没有在一条连接的 GAE 链上重复发多次终奖。
超长/超时轨迹仍验证当前产物，标记 truncated；开启 actor-only truncation filter：截断轨迹仍训练critic，actor对应loss mask清零。
critic-only阶段允许全批截断；actor开始训练后全批截断会明确报错，避免空actor更新。

默认actor直接使用Miles的 `score_centering_loss.py` 和已有SC estimator，优势仍来自critic GAE。
只扩展原先GRPO-only的兼容校验，允许PPO优势；不自写actor estimator。
官方SC目标不使用PPO clip，`--no-score-centering` 才使用标准clipped PPO。
两者都不开dual clip、entropy、KL或advantage whitening。SC的IS仅使用其官方 `score-centering-is` 开关。
critic 使用 `value_loss_coef * max((V-R)^2, (clip(V,V_old±value_clip)-R)^2)`。
actor整批一步；critic按完整prompt group拆成 **4个mini-batches，每批8组×16=128 episodes**，每批一步。
整批 `V_old` 在第一次更新前冻结，四步后仍把该快照返回actor；每步重新构建critic memory。
第一批尚未更新时value clip等价MSE，后续批可以相对同一份 `V_old` 触发clipping；它仍不是参数更新幅度硬上限。
启用上游已有inverse-std helper：每组16条terminal returns的population std，floor .075，
按完整rollout的有效prompt-group原始权重均值归一化；segment按原episode去重，权重仅乘loss分子。
每个mini-batch用其全局未加权有效token数作分母，不在各mini-batch重新估计或归一化权重。
EV 从全局 token moments 计算，不平均各 rank 的局部 EV。

## B200 资源候选与具体分配

| 角色 | 卡数/并行 | 行为 |
|---|---|---|
| Megatron actor | 16 卡，TP4 × PP1 × CP1 × DP4 | 分布式 optimizer、逐层重算 |
| 独立 HF critic | 同一批 16 卡 | 与 actor 轮流 CPU offload；不是额外申请 16 卡 |
| SGLang rollout | 48 卡，12 个 TP4 engine | 生成期间不与 trainer 共卡 |

HF critic 主要使用 BF16 参数/梯度，各 rank 拥有完整 forward 网络。
`A_log` 在HF checkpoint加载阶段就保留FP32，并在CPU/GPU换入换出时保持dtype；
这避免先round到BF16再转回FP32而永久损失循环层参数精度。
critic 的本地梯度累计使用 BF16；Megatron 的 `--accumulate-allreduce-grads-in-fp32` 只作用于 actor。
FP32 master weights 和 Adam moments 分片放在 CPU；梯度以最多 64 MiB 的 FP32 chunk 求和，
全局 norm clipping 后由 owner 更新，再 broadcast BF16 参数。
actor TP siblings 读到同一个 episode shard，因此这里四个 HF replica 重复 critic forward。
梯度和 token-count 的全局求和都包含这份重复，数学归一化抵消，但 **吞吐代价保留**。
这是缓存兼容的直接实现，尚未实现 tensor-sharded HF critic 或通信/计算重叠。

以 27B 粗估：每个 critic GPU 参数约54GB、梯度约54GB，另加 KV、激活和临时张量；
FP32 CPU master+两份 moments 总约324GB，分在16个 worker；8个 CPU-offloaded critic 副本每节点约432GB。
加actor backup、optimizer、数据和进程开销，训练节点保守预留约1.5–2TB主机内存；实际值需测。
默认SC不需要old-actor/rollout-actor队列；关闭SC的PPO基线会启用这两份backup。
`--max-tokens-per-gpu` 是调度上限，不是显存保证。
当前 scheduler 让同 episode 的 actor segments 在一个 microbatch 中；
其总 token 数及 critic 全历史都必须通过262144预算检查，超出时明确报错，不偷偷截历史或丢 segment。
262144为120000生成tokens之外的工具输出和重复actor context预留空间。
actor同episode的segments会进入同一个packed forward，不能把262144当作纯CPU检查；
full_detach历史KV也随episode增长。它没有通过B200峰值验收，32k actor窗口不能保证critic只保留32k。

默认64卡为16 trainer +48 rollout；32卡候选为8+24。
1:3分区参考 [AReaL §6.1](https://proceedings.neurips.cc/paper_files/paper/2025/file/33c00862bfa29ac72ecf630a41e19352-Paper-Conference.pdf#page=7)，
并保留默认训练DP4；这是待测候选，不能从不同模型/算法的论文确定本配置最优比例。
两者均未完成B200容量/吞吐验收。CPU 验证不等价于目标集群实测。本机没有 GPU，未宣称容量、Flash Attention kernel 或吞吐已获验证。

模型结构由已有 `scripts/models/qwen3.5-27B.py` 导入：64层，hidden 5120，FFN 17408，
24个query heads、4个KV heads、head dimension 256，词表248320；RMSNorm epsilon `1e-6`，
RoPE base `1e7`、rotary fraction `.25`。使用gated attention、SwiGLU、独立embedding/output weights，关闭linear bias。
FP32 softmax/梯度累计用于Megatron数值稳定；sequence parallel在TP组内分摊激活；
uniform逐层完整重算用额外计算换激活内存，distributed optimizer在DP ranks间分摊optimizer状态。
actor attention/hidden dropout均为0，避免额外的train/inference差异。

## Launcher 参数逐项说明

`0` 的 batch/oversample 表示按 dataset 自动选默认；显式值覆盖自动值。

| 参数 | 默认 | 含义 |
|---|---|---|
| `action` | all | prepare / train / 二者依次执行 |
| `dataset` | frontiersmith | 同时选择训练和评测 corpus |
| `critic-context-mode` | full_detach | 两种独立 memory 实验 |
| `num-nodes / num-gpus-per-node` | 8 / 8 | 默认64卡，支持更改分区 |
| `trainer-num-nodes / rollout-num-gpus` | 2 / 0 | 0自动使用剩余节点，默认16 train +48 rollout |
| `tensor-model-parallel-size` | 4 | actor TP；PP/CP固定1 |
| `rollout-gpus-per-engine` | 4 | SGLang TP；48卡形成12 engines |
| `sglang-mem-fraction-static` | 0.7 | inference静态权重/KV预算比例 |
| `sglang-server-concurrency` | 128 | 每engine的generate-loop并发预算；默认12 engines理论上限1536，仍受实际发起量和sandbox容量限制 |
| `model-dir / data-dir / output-dir` | /root/models / /root/datasets / /root/shared_data | 可改为共享目录 |
| `model-name / hf-repo / hf-revision` | Qwen3.5-27B / Qwen/Qwen3.5-27B / 固定SHA | 下载目录、来源、版本；更换repo时同时换revision |
| `megatron-path` | /root/Megatron-LM | 训练后端代码路径 |
| `prompt-data / eval-data` | 自动路径 | 覆盖所选corpus的JSONL |
| `tasks-dir / frontiersmith-source-dir` | data-dir/harbor / data-dir/FrontierSmith | Harbor任务根目录和judge挂载源码 |
| `frontier-cs-source-dir` | data-dir/Frontier-CS-eval | 独立固定版本的Frontier-CS评测源码 |
| `trials-dir` | output-dir/run-id/trials | Harbor日志、轨迹与verifier结果 |
| `run-id` | 自动生成 | 独立实验目录；恢复时沿用原值 |
| `critic-load` | 空 | 从初始HF backbone起步；否则载入native memory-critic checkpoint |
| `critic-weights-only` | false | 搭配critic-load和新run-id做warm-start；清空optimizer/RNG/step而非恢复旧训练 |
| `num-rollout` | 120 | 包括critic-only的outer steps预算，**不是epochs** |
| `num-critic-only-steps` | 30 | 前30个outer rounds冻结actor；首actor更新为外轮31 |
| `rollout-batch-size / samples-per-prompt` | 32 / 16 | 接受32个完整prompt groups，共512 episodes |
| `over-sampling-batch-size` | 35 | 首发/补采35 groups，共560 episodes；完整组入批，尾部取消 |
| `critic-steps-per-rollout / critic-return-std-floor` | 4 / .075 | 每步8 groups×16=128 episodes；开启官方inverse-std weighting |
| `temperature / top-p / top-k` | 1 / 1 / -1 | 训练sampling；-1关闭top-k，此组是改编选择 |
| `actor-lr / critic-lr` | 1e-6 / 5e-6 | 两个网络独立学习率 |
| `adam-beta1 / adam-beta2 / adam-eps` | .9 / .999 / 1e-8 | Adam动量衰减和数值稳定项 |
| `weight-decay / clip-grad` | .01 / 1 | AdamW权重衰减、全局梯度范数上限 |
| `policy-clip / policy-clip-high` | .2 / .28 | 仅关闭SC的PPO基线生效 |
| `value-clip / value-loss-coef` | .2 / .5 | 四步critic使用冻结的整批V_old；回归系数缩放loss |
| `score-centering / score-centering-is` | true / tis | 官方SC loss及其none/tis/mis权重选择 |
| `score-centering-candidates` | 128 | 真实生成调用中保存的top-K数 |
| `score-centering-tis-clip` | 2 | 官方TIS权重上限 |
| `score-centering-mis-low / score-centering-mis-high` | .5 / 5 | 选择MIS时的权重区间 |
| `score-centering-chunk-size / score-centering-recompute-loss` | 128 / true | 官方分块概率计算及loss重算 |
| `max-context-tokens / compact-threshold` | 32768 / 24576 | actor窗口上限、请求摘要阈值；为生成预留空间时可能提前触发 |
| `max-new-tokens / summary-tokens` | 8192 / 2048 | 单次action/摘要的采样token上限 |
| `max-total-tokens` | 120000 | episode累计生成量，含summary；不含环境输出 |
| `max-steps` | 300 | action循环上限；不是wall-clock timeout |
| `command-timeout / max-output-chars` | 120 / 16000 | 单bash命令秒数、环境输出截断字符数 |
| `max-tokens-per-gpu` | 262144 | episode训练调度上限；超长episode报错 |
| `reward-mode / reward-key` | native / reward | 奖励接口与Harbor字段 |
| `pass-count-key / reward-divisor` | passing_assertions / 20 | 可选test-count字段与固定尺度 |
| `eval-interval / save-interval` | 10 / 10 | outer round周期+final；eval另有initial，显式关闭epoch附加触发 |
| `eval-temperature / eval-top-p / eval-top-k` | 1 / 1 / -1 | 每题一条，不请求SC候选 |
| `eval-max-tasks / eval-subset-seed` | 16 / 42 | 固定随机子集；0显式全量，缓存原数据不变 |
| `seed` | 1234 | 训练随机状态；data source shuffle另用未覆盖的rollout-seed默认42 |
| `chat-template-path` | 内置interleaved模板 | 工具turn间保留thinking；不再次apply-chat-template |

继承的 `U.ExecuteTrainConfig` 提供 Ray/Kubernetes backend、namespace、runtime env、部署与日志设置。
一般只需要共享路径和 `MILES_SCRIPT_EXTERNAL_RAY=1`；通用选项通过 `--help` 查看。
该 launcher 不接受一条不受校验的 extra-args 覆盖串，资源/算法旋钮显式声明在 dataclass 中。
默认120外轮：critic480次optimizer updates，actor90次；每10轮及final保存，共12点。
评测initial和10..120，共13点，每点固定16题×1=16 trials，总208 trials/实验。
完整TB89题或FCS179题只在显式 `--eval-max-tasks 0` 时使用，没有自动全量final。
任务子集固定不表示temperature1生成可确定；quick分数不当作完整benchmark成绩。

## Warmup、checkpoint 与论文差异

初始模型是固定版本 Qwen3.5-27B，不是原文的专门 SFT checkpoint。
critic 从其 text backbone 加独立随机 scalar head起步。默认30轮 critic-only 共120次critic更新，
**不等价于论文的完整 warm-start**；冷启动应先单独训练 critic，并根据 EV/value loss 验证校准情况。
原文 warmup 的叙述有歧义，没有完整公开到可直接复制的独立训练命令；不补造其缺失参数。

纯critic运行可设 `--num-rollout N --num-critic-only-steps N`，然后用保存的critic启动正式实验。
例如 warmup 的 run-id 为 `fs-warmup`，正式实验使用新run-id，并加
`--critic-load /shared/runs/fs-warmup/checkpoints_critic --critic-weights-only`。
恢复同一实验则沿用run-id，显式指定对应 `--critic-load .../checkpoints_critic`，不加weights-only。
只恢复actor而不指定critic会因两者step不一致而拒绝启动。
actor Megatron checkpoints 与 `checkpoints_critic/memory_critic_*` 的 native HF checkpoints 分开。
critic 保存 BF16 backbone/head、每 rank FP32 master/Adam state、随机状态和模式/拓扑信息；
该格式不兼容旧独立critic_engine的输出，也不能直接传Megatron critic checkpoint。
同run恢复保持同world size、context mode和actor TP；权重warm-start与恢复optimizer/RNG应明确区分。
本实现保留optimizer以便真正恢复，区别于论文仅存权重的存盘策略。

本配置的32k窗口、压缩阈值、summary预算、累计生成120000、每步生成量、max300 actions、梯度裁剪、entropy=0、value系数和120外轮均是工程选择。
T1没有明确给出累计episode生成token上限；84k context不能替代累计生成预算。
[VHD-Play Table 9](https://arxiv.org/html/2609.27321v1#A6)明确使用120000累计生成上限；
它提供同量级agentic RL先例，但属于工具机制GRPO，不能当作terminal训练的统一标准。
论文的84k窗口、完整TITO边界修复、122B MoE/R3、自有T1数据质量筛选和assertion解析没有被冒充已经复现。
FS数据集很小，每个prompt抽16条轨迹；跨shuffle epoch和outer rounds仍会反复遇到同一个problem。
Harbor task自身另有agent/verifier wall-clock timeout；`max-steps`和`command-timeout`不会覆盖它们。
并发预算覆盖整个自定义episode loop；调整它会影响Docker资源消耗、尾部耗时及“最先完成”数据分布。

## 验证范围

相关CPU回归为 **1400 passed、122 skipped**。覆盖tiny Qwen3.5两种memory的数值/梯度、
跨段detach、value位置与span映射、分片CPU Adam参考一致性、native checkpoint恢复、
compaction去重、Harbor取消清理、奖励与数据边界，以及官方SC/GAE概率链路。
512条episode的conversion→pack→process调度与小模型critic训练分别验证。
长轨迹测试以解码桩驱动真实TerminalAgent累计生成120000 tokens并多次摘要，
验证actor段长≤32768、mask/value spans完整和262144打包预算；
它不等于真实模型在该长度的GPU forward或显存验收。

全仓launcher检查为 **551 passed、72 failed、10 errors**；82项失败/错误与
干净上游 `eb67505` 完全一致，包括旧快照缺失/过期、旧AMD launcher API和环境依赖。
本launcher的prepare/execute快照4项通过，文档生成一致性、Ruff和语法检查通过。
数值测试使用CPU eager模式，`OMP_NUM_THREADS=1 TORCH_COMPILE_DISABLE=1`。

原生Megatron/Transformer Engine、真实GPU↔CPU offload、Docker judge及B200端到端训练
仍需目标集群验证；未宣称长序列容量、kernel或吞吐已经通过。
实际数据准备生成190/179与14,601/89行输入。

## Score Centering 的官方支持边界

官方Miles已提供 `--loss-type score_centering`、`--score-centering-is none|tis|mis`，
以及普通PPO的 `--keep-old-actor`、`--use-rollout-logprobs` 和 `--use-tis`。
本recipe直接使用这些已有入口；不新增actor estimator或importance-correction公式。
原基线的SC启动校验只接受GRPO，本fork将其扩为GRPO/PPO，复用critic产生的GAE。
这是组合接入的兼容扩展；官方SC数学保持不变，完整训练结果仍需集群实测。
与PPO clip的区别、IS开关和单tail近似见 [PPO_SCORE_CENTERING.md](PPO_SCORE_CENTERING.md)。
