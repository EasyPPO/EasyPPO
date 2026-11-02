# AM 官方对齐修复与其余实现审查

日期：2026-11-03。算法依赖固定在 `adamzweiger/compaction` 的
`2020f6d37940fe013e7e7f1ebf409e5d107c5670`。

## 本次实际修复

| 原实现 | 处理结果 | 依据 |
|---|---|---|
| 每 128 query 自定义 SDPA 循环及 chunk checkpoint | 删除；native critic 通过 Megatron/TE 读取，HF 参考使用普通 additive-mask backend 调用 | 官方模型在 cache/beta 接入后复用 attention interface，没有该循环 |
| 无历史首段仍走通用 memory reader | 整段一次普通 causal attention，仍保留边界拟合 | 没有旧 memory 时可见性与原始 causal 完全一致 |
| value fit 向原始保留 V 正则化的 ridge | AM 主路径删除；使用官方 C2 least squares，ridge_lambda=0 | 官方 HighestAttnKeys demo 与算法组件 |
| 替换官方 selector 为本地 chunked 方法 | AM 主路径不再使用；官方方法不被 monkeypatch | 原样调用固定版本算法方法 |
| 强制将全部 fitting Q/K/V 转 FP32 | 删除；保留官方 matmul/softmax/solver 精度处理 | 官方自身的 precision policy |
| 每个边界强制 seed=0 并 fork RNG | 删除；使用正常 RNG 与框架 checkpoint RNG 恢复 | 无官方依据必须每边界重复同一随机流 |
| 压缩 slot 数使用 round | 改为 floor，至少一个 | 官方 demo 的 target_size 计算 |
| recipe 固定 Flash-only | AM 模式选择 auto，启动时拒绝 Flash-only | 一般 post-scale beta 不能由 Flash-only 表达 |
| HF 参考注册捕获 model 的全局 attention closure | 改为有作用域的 instance mapping，异常退出也恢复 | 原实现会让全局 registry 持有最后一个模型及 memory |
| 新旧算法共享结果标记 | 使用 problem_am_resume_v2 | 防止把旧自定义 ridge/reader 的输出当成本次结果 |

`official_am.py` 使用官方 demo 的明确配置：RMS scoring、NNLS 两次 refinement、
权重边界 0.05/20、C2 lsq、无 ridge、lstsq solver。未改官方源码，未手写新 attention kernel。
旧 `mode=kv` 下的 `am_ridge` 与分块方法仍为显式消融保留，不参与主 AM 路径。

## 官方算法与必须保留的任务适配

这不是直接运行官方 Qwen3 推理 demo。官方没有当前 Qwen3.5 GDN/Miles critic 的完整接入，
以下内容是任务适配，不应被称为官方原生实验定义：

- `[问题] + [AM memory] + [summary、继续作答指令和模板]` 的读取结构。
- 问题 prefix 独立计算；不保留另一份完整生成 summary KV。
- 原 actor segment 边界、token、mask、reward 和 value 对齐保持不变。
- 线性注意力每段重置卷积与递归状态，与用户确认的 actor-summary 语义一致。
- 使用本段末尾 summary completion token 位置的 critic Q，而不是 actor Q。
- 历史 AM memory detached；每个新前向按当前权重重建。

这些约定保留，因为是已确认的任务需求或其必要实现。它们没有论文层面的效果保证。

## 唯一必须显式解释的 bias 衔接

已有 memory 的 source beta 非零时，官方 key selector 构造的设计矩阵是
`exp(QK/sqrt(d) + source_beta)` 的选中列，因此 NNLS 输出的 log weight 是增量。
存入后续读取、并交给官方 C2 拟合的 beta 必须是：

```text
absolute_beta = incremental_beta + source_beta[selected]
```

这里调用官方未替换的 `_select_keys_highest_attention` 与 `_compute_C2_with_method`，
在两步之间进行这项转换。它不是声称官方 public wrapper 在带历史 beta 时已经做了转换。
我们没有盲目复制其 public wrapper，否则后续联合 attention 的历史 mass 会出错。

独立验证：三个完全相同的 key 带 mass 2、3、5，压成一个后必须带 mass 10。
若保留 mass=5 的 key，NNLS 增量 multiplier 为 2；只保存 log(2) 是错误的，
必须保存 log(5)+log(2)=log(10)。测试同时验证了与一个新 token 联合归一化后的输出，
没有用“比较两个相同实现”替代数学检查。

无历史 source bias 时，新路径与官方 public `compute_compacted_cache` 逐 head 对照，
覆盖 FP32/BF16，并核对 RNG 消耗；测试禁止调用旧 patched selector 与 V0 ridge。

## native attention 接入具体做法

外层继续使用原 THD segment packing，以保留 GDN 重置边界。进入 full-attention core 后，
每个独立读取对采用原生 TE BSHD 接口：

- 无历史：整段普通 causal。
- 有历史的问题 prefix：独立 causal。
- 有历史的 suffix：Q 为 suffix，K/V 为 `[问题][AM memory][suffix]`；使用
  `causal_bottom_right`，通过 `attention_bias` 加 per-key beta。

bias 使用真实连续的 `[1,Hq,1,Lk]` 张量，通过 singleton query 维度广播，
不使用零 stride 的 `[1,Hq,Lq,Lk]` expand 视图。TE fused C++ 只传 data_ptr/shape，
其 cuDNN descriptor 根据 shape 生成连续 stride；旧 expand 视图在此边界不安全，已修正。
具体命中 fused 还是 unfused、workspace 和长序列显存开销，
仍须真实 GPU 验证，不能依据接口存在就承诺性能。

HF frozen reference 采用官方同类 additive-mask + backend 调用，可能创建较大 dense mask，
不作为 HF AM 训练器或高性能长序列实现宣传。

## 还存在的设计选择与未验证风险

| 项目 | 结论与处理 |
|---|---|
| summary Q 包含哪些 token | 精确原始 summary completion span，可能包含 thinking/EOS；resume 正文可能经 answer_text 去掉 thinking。保留原始 offset，明确这是任务 probe 选择，不称为官方默认 |
| summary Q 的 token 位置 | 是该 token 位置的 Q，不是 value 读取用的前一位置；未偷偷改成下一段未来 Q |
| 跨段 RoPE 的统一偏移 | 保留现有 logical offset，保持当前段内部相对位置；跨历史的距离选择属于适配，CPU 相对位置测试不证明 9B 效果最优 |
| problem prefix 的 LCP | 使用原 token ID 的公共前缀，BPE 边界保守保留在 suffix；不是重新生成或按文本查找问题。应在真实 tokenizer/数据上继续核查 |
| 逐 head 拟合 | 官方 HighestAttnKeys 默认就是顺序路径；不是偏离官方的独立证据。是否优化由 profiler 决定，不自行接一个不兼容 batch solver |
| 官方 dense 拟合显存 | 恢复原始算法可能比旧 chunked selector 使用更多临时显存；尚无长 summary 的生产峰值验证，不静默换 solver/probe |
| checkpoint 重放拟合 | 仍可能发生，no_grad 不阻止重放；之前 CPU 已复现。未引入新的持久缓存来掩盖重算，待真实 Megatron 计数 |
| 全梯度与 detached-history 对比 | AM 仍 detached；归因时要有合适梯度控制，不能仅凭 full vs AM 得出算法质量结论 |
| 官方 LS 的异常回退 | 保留上游行为；上游可能在非有限/失败时自行回退正则求解，不另外添加静默 V0 ridge |
| cuDNN 诊断结论 | 已确认的是历史 HF SDPA 某条路径；不能直接推定 native TE fused 错误或正确。HF 参考保留已审计的禁用策略，TE 必须独立验证 |
| 上下文长度与拓扑 | 当前明确限制 TP=PP=CP=1、整 episode batches；不能声称支持任意并行和任意长度 |

## 验证边界

CPU 测试验证官方无 bias 数值对齐、历史 bias mass、上下文/指令保留、actor fallback、
原生包装器 BSHD 与 bottom-right/bias 入参及梯度、checkpoint 数值一致、HF registry 异常恢复。
包装器 CPU 测试使用 stand-in，不等于真实 Megatron/TE 测试。

新增 `tests/agentic_critic/test_native_am_gpu.py` 在有 CUDA 的独立进程中使用真实
Megatron/TE、16:4 GQA、head_dim=256、BF16 做 forward/backward 对照。当前 CPU 环境会跳过，
不能把跳过算通过；也不能用小形状 smoke test 替代 9B 长序列、DP、checkpoint 和收敛实验。

实验目标仍是比较 critic 上下文方案；恢复官方 C2/精度/RNG 会改变数值，必须使用新目录重跑，
不能复用旧 AM 预测或将其作为同一算法训练曲线续接。
