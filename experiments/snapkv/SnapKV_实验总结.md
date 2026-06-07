# SnapKV 实验总结(Qwen3-30B-A3B-Instruct)

> 在干净的 chat 流水线(chat template + EOS、非思考 Instruct 模型、无截断、greedy、
> `alias_match` ACC)上,在 LongBench-v1 多跳 QA(2wikimqa / hotpotqa / musique)上
> 系统评测 **SnapKV(基于注意力重要性的 KV-Cache 压缩)**。
>
> 这与 cache-**splice**(cached vs fresh 复用保真度)是**不同的维度**:SnapKV 问的是
> "重要性裁剪能把 KV-Cache 压到多小才开始掉点"。

---

## 1. 两个 runner / 两种 Observation Window

| 文件 | 内容 | Observation Window | 能否预计算 |
|---|---|---|---|
| `scripts/32_snapkv_coverage.py` | **SnapKV vs Fresh** | **问题**(query-aware) | ❌(query 时才有问题) |
| `scripts/33_rag_snapkv.py` | **RAG 原生可预计算 SnapKV** | **anchor=检索 top-1(oracle)chunk**,完整保留并用作打分窗口 | ✅(与问题无关) |

实现要点(两者通用):
- `kvpress` 与 transformers 5.7 不兼容 → **自研忠实 SnapKV**(per-kv-head Top-K + 1D
  max-pool kernel 7,处理 GQA q=32/kv=4)。
- **不做 RoPE shift**:SnapKV 保留 post-RoPE key 在原始位置,相对距离精确(已用
  keep-all identity gate 逐字节验证)。
- **性能关键**:prefill 用 SDPA(eager 会在长上下文上物化 O(L²) 注意力矩阵);只有
  **短的观察窗口 forward**(q_len 很小)临时切 eager 拿 `output_attentions`,显存极小。

---

## 2. 实验一:SnapKV vs Fresh(obs=问题,全量 LongBench n=600)

结果文件 `results/snapkv_cov.s*.json`(n=600);`results/snapkv_lim100/` 是更早的 n=300。
**acc_fresh = 0.669**

| keep 比例 | keep frac | acc_snap | Δ vs Fresh |
|---|---|---|---|
| 5%  | 0.053 | 0.580 | −0.089 |
| 10% | 0.103 | 0.602 | −0.067 |
| 20% | 0.203 | 0.625 | −0.043 |
| 30% | 0.302 | 0.637 | −0.032 |
| 50% | 0.502 | 0.654 | −0.015 |

**单调、平滑衰减**:50% 保留 ≈ Fresh,20–30% 内差 ~3–4 点,只有激进的 5–10% 才掉
~7–9 点。这些上下文信息密集,低预算时**答案本身可能被裁掉**,所以会掉点。
(n=300 与 n=600 形状一致,结论在全量规模下成立。)

---

## 3. 实验二:RAG 可预计算 SnapKV —— anchor-obs(B) vs question-obs(A)(n=501)

**动机**:真实 RAG 里 cache 必须在 query 之前**离线算好**,所以 obs **不能是问题**。
设计:anchor = 检索 top-1 = **oracle chunk(含答案那块)**,移到 context 末尾、**完整保留**
并用它的 query 给其余 chunk 打分压缩(query-independent → 可预计算);query 时再拼问题解码。
每条记录在**同一布局、压缩同一区域、anchor 都完整保留**下跑三臂:
**Fresh(全留)/ B(anchor-obs,可部署)/ A(question-obs,上界,不可预计算)**,
`A−B` = **"可预计算"的代价**。结果文件 `results/ragsnap_cov.s*.json`。**acc_fresh = 0.735**

| keep 比例 | keep frac | acc_B(anchor,可部署) | ΔB | acc_A(question,上界) | ΔA | gap A−B |
|---|---|---|---|---|---|---|
| 5%  | 0.080 | 0.719 | −0.016 | 0.760 | +0.026 | +0.042 |
| 10% | 0.128 | 0.745 | +0.010 | 0.766 | +0.032 | +0.022 |
| 20% | 0.225 | 0.750 | +0.016 | 0.750 | +0.016 | +0.000 |
| 30% | 0.322 | 0.749 | +0.014 | 0.747 | +0.012 | −0.002 |
| 50% | 0.516 | 0.762 | +0.028 | 0.747 | +0.012 | −0.016 |

**核心结论**:用检索 top-1 chunk 当观察窗口(与问题无关、可预计算),在 **keep ≥ 20% 时
与不可预计算的 query-aware 版本几乎无差(gap ≈ 0)**,只有极端 5% 才有 ~4 点代价。两者都
≥ Fresh,因为 oracle anchor 保证答案在 cache 里;query-aware 窗口只在很紧的预算下更好地
保留**多跳支撑上下文**(gap 集中在 musique)。压缩有时还略**超过** Fresh —— 去掉了干扰
上下文(SnapKV 已知效应)。

**部署结论**:RAG 预计算时,用**检索 top-1 chunk 作为 SnapKV 观察窗口**对每个检索集**离线
压缩**,保留 ~20%+ → 近乎无损,压缩时**不需要问题**。

---

## 4. 与 cache-splice / 实验之间的可比性(重要)

- **实验二保护了答案**(anchor=答案块,恒留),所以曲线比实验一平很多(实验一答案可能被裁)。
  **两次 SnapKV 的绝对数字不可直接比**(记录集不同 + anchor 保护),它们回答不同问题。
- **SnapKV 与 cache-splice coverage 是不同维度**(后者是"复用 vs 重算"的保真度,cached ≈
  fresh):不要把两者的 Δ 放进同一张表。三者是 KV-Cache 效率的**互补坐标轴**。

---

## 5. 复现

见 `launchers/`(16-way 分片:`launch_snap_range.sh` / `launch_rag_range.sh`,参数
`MAXNEW LIMIT START NUM`);单卡 sanity:`snap_sanity.sh` / `rag_sanity.sh`(含 identity
gate);汇总:`snap_progress.sh` / `rag_progress.sh`。
模型:`Qwen3-30B-A3B-Instruct-2507`;ratios 0.05/0.1/0.2/0.3/0.5;kernel 7;max_new 1024。
