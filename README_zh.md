# 让数据决定

**通过 Off-Policy Distillation 持续预训练中的监督分析、能力权衡与自适应目标路由**

Jiangan Yuan*，Zhixuan Li*，Han Xu
（* 表示共同第一作者）

[📄 论文（PDF）](./Let_the_Data_Decide.pdf) · [arXiv](#) · [引用](#citation)

---

# TL;DR

我们研究了 **Top-`k` 截断（top-`k`-truncated）、温度缩放（temperature-scaled）的 Off-Policy Distillation**，并将其视为一个**结构化监督设计（structured supervision design）问题**，而不是一个全局超参数选择问题。

通过在 **token** 层面对 **LM** 与 **KD** 的梯度差异进行分解，并提出能够量化**教师模型与数据一致性（teacher–data alignment）**的诊断指标，我们发现：

* `k` 决定了**覆盖率（coverage）—尖锐性（sharpness）之间的权衡**；
* `τ` 控制的是**支持集（support）内部的概率分配**，这一作用与支持集成员本身相互独立（orthogonal）；
* **领域级目标路由（Domain-level objective routing）**（数学/代码数据使用 LM，一般领域数据使用 KD）始终优于：

  * 单一训练目标（single-objective）；
  * 更细粒度的 token 级路由（token-level routing）。

原因在于，决定路由效果的并不是**路由粒度（granularity）**，而是**路由信号本身的质量**（即 teacher–data alignment）。

因此，我们将持续预训练中的 Off-Policy Distillation 重新定义为一种**数据条件（data-conditional）的监督设计问题**。

---

# 关键贡献（Key Contributions）

1. **能力差异是一种结构性现象，而非随机噪声。**

   在训练数据、模型初始化以及所有超参数均保持一致的条件下，LM 与 KD 会学习出**系统性不同（systematically distinct）**的能力画像（capability profiles）。

   * LM 在高难度推理、数学以及 Pass@K 指标上表现更优，例如：

     * MATH-500 Pass@16：+29.86
     * AIME 2025 Pass@128：+23.33
   * KD 在常识推理、事实检索、阅读理解以及结构化程序生成方面表现更优。

   二者不存在绝对优劣，而是各有所长。

2. **提出了 LM 与 KD 的梯度级分解（gradient-level decomposition）。**

   我们证明，两种训练目标实际上对应两类互补的监督信号：

   * **对真实观测 token（observed token）的直接强化（direct observed-token reinforcement）**
   * **教师模型支持的替代监督（teacher-supported alternative supervision）**

   两种监督信号之间的相对权重完全由以下两个因素决定：

   * 真实 token 是否位于教师模型 Top-`k` 支持集中；
   * 教师模型如何在支持集内部进行概率分配。

3. **提出了一套诊断指标（diagnostic metrics）。**

   包括：

   * Coverage@k
   * OTRank
   * OTMass
   * CondOTMass
   * Top1Prob
   * Entropy
   * RawTopKMass

   这些指标能够在**每一个训练位置（training position）**定量衡量监督信号，从而将过去只能定性理解的 LM↔KD 光谱（spectrum）转化为可量化分析的权衡关系。

4. **通过受控实验（controlled sweeps）分离了 KD 的两个自由度（degrees of freedom）。**

   实验表明：

   * `k` 改变的是**覆盖率（coverage）**，代价是降低监督的尖锐程度（sharpness）；
   * `τ` 则不会改变支持集成员，而是在**固定支持集内部重新分配概率质量（mass）**。

   不同任务对应不同的最优配置。

5. **领域级目标路由取得最佳效果。**

   将：

   * 数学/代码数据交给 LM；
   * 通用领域数据交给 KD；

   可以同时实现：

   * 恢复 LM 在推理、数学和 Pass@K 上的优势；
   * 保留 KD 在知识和常识任务上的优势；
   * **在 MBPP、BBH、AIME 2024、AIME 2025 等多个 benchmark 上超过两种单一目标训练方法。**

6. **Token 级路由不如领域级路由；在 token 级信号中，OTMass 优于 Entropy。**

   OTMass 同时依赖：

   * 当前观测 token；
   * 教师模型预测分布；

   因而能够直接刻画 **teacher–data alignment（一致性）**。

   相比之下，Entropy 只反映教师模型预测分布自身的集中程度，它无法区分：

   * 教师模型**与数据一致**；
   * 教师模型**虽然十分自信，但预测错误**。

---

# 论文整体逻辑（The Logical Loop）

整篇论文按照一个闭环展开：

> **现象（Phenomenon） → 机制（Mechanism） → 测量（Measurement） → 控制（Control） → 应用（Application） → 闭环（Closure）**

| 步骤                     | 章节           | 输出                                                                                                              |
| ---------------------- | ------------ | --------------------------------------------------------------------------------------------------------------- |
| **1. 现象（Phenomenon）**  | §5           | LM 与 KD 学习出了系统性不同且互不支配（non-dominant）的能力画像。                                                                      |
| **2. 机制（Mechanism）**   | §6.1         | 梯度分解揭示了两类互补监督信号：真实 token 强化（observed-token reinforcement）与教师支持的替代监督（teacher-supported alternative supervision）。 |
| **3. 测量（Measurement）** | §6.2         | 提出的诊断指标使监督信号之间的平衡能够在每一个 token 上进行量化分析。                                                                          |
| **4. 控制（Control）**     | §6.3、§6.4    | `k` 与 `τ` 通过不同机制调节监督信号之间的权衡，因此不同任务具有不同的最优配置。                                                                    |
| **5. 应用（Application）** | §7.1、§7.2    | 领域级路由将这种监督权衡转化为可利用的设计自由度；按领域统计的诊断指标进一步解释了其成功原因。                                                                 |
| **6. 闭环（Closure）**     | §7.3、§7.4、§8 | OTMass 优于 Entropy，因为它直接衡量了 Eq.(11) 中导致梯度差异的核心项。路由效果并不取决于粒度，而取决于路由信号是否与底层梯度差异保持一致。                               |

最终，这一闭环支撑了论文的核心观点：

> **持续预训练中的 Off-Policy Distillation 本质上是一种结构化（structured）、数据条件（data-conditional）的监督设计问题（supervision design）**。

这也正是论文标题 **"Let the Data Decide"（让数据决定）** 所表达的核心思想。

下面是第二部分的翻译。我保留了**所有 Markdown 格式**，包括公式、代码块、表格、图片的 Markdown 写法（图片链接保持原样，不翻译）。

---

# 方法（Method）

## Off-policy KD 目标函数（Off-policy KD Objective）

给定教师模型 `f_T` 和学生模型 `f_S`，标准的插值训练目标（interpolated objective）定义为：

```text
L = (1 - α) · L_LM + α · L_KD
L_KD = (1/n) Σ_r KL( p̃_T^{k,τ}(· | x_<r) ‖ p_S(· | x_<r) )
```

其中：

* `p̃_T^{k,τ}` 表示教师模型经过 **Top-`k` 截断（top-`k` truncation）**、**温度缩放（temperature scaling）**以及**重新归一化（renormalization）**后的概率分布。
* `α` 用于控制语言模型目标（LM）与知识蒸馏目标（KD）的权重。

因此，整个训练目标实际上是在：

> **真实数据监督（LM）** 与 **教师模型监督（KD）**

之间进行插值。

---

## 梯度差异（Gradient Gap，公式 11）

论文推导得到 LM 与 KD 对学生模型 logits 的梯度差异为：

```text
Δg_i  =  ∂L_KD/∂z_i^S  -  ∂L_LM/∂z_i^S  =  1[i = x] - p̃_T^{k,τ}_i
```

一个十分重要的结论是：

> **学生模型当前的预测分布完全被抵消（cancel）了。**

因此，LM 与 KD 的梯度差异**仅仅取决于**：

* 数据标签（one-hot target）
* 教师模型经过 Top-`k` 截断后的目标分布

两者之间的差异。

换句话说，

LM 与 KD 的区别并不是来自学生模型当前学得怎么样，而是来自：

> **教师模型是否支持当前真实 token，以及支持程度有多大。**

---

### 两种情况

| 情况        | 对真实 token `x` 的影响                              | 对教师支持的其他 token（alternatives）的影响                                        |
| --------- | ---------------------------------------------- | ---------------------------------------------------------------------- |
| `x ∈ K^T` | KD 对真实 token 的增强作用弱于 LM（因为概率质量被分配给了其他候选 token） | 不再像 LM 那样统一压制其他 token，而是进行非抑制式更新（non-suppressive update），有时甚至会提升它们的概率。 |
| `x ∉ K^T` | 梯度方向发生反转：LM 会提高 `z_x`，而 KD 会主动压低 `z_x`。        | 与上面相同，教师支持的候选 token 会获得强化。                                             |

论文认为，这正对应着两种互补的监督机制：

* **LM：**

  * 全力强化真实观测 token（observed token reinforcement）。

* **KD：**

  * 除了关注真实 token，还会学习教师认为合理的其他候选 token（teacher-supported alternatives）。

因此，两者天然存在能力上的权衡。

---

## 诊断指标（Diagnostic Metrics）

为了能够直接观察监督信号，论文提出了一组诊断指标。

| 指标                        | 定义                                | 不变性（Invariances）                        |
| ------------------------- | --------------------------------- | --------------------------------------- |
| **Coverage@k**            | `1[x_r ∈ K_r^T]`                  | 仅依赖于 `k`；与 `τ` 无关。                      |
| **OTRank**                | 真实 token `x_r` 在教师模型 logits 中的排名。 | 与 `τ` 无关。                               |
| **OTMass**                | `p̃_T^{k,τ}_{r, x_r}`             | 同时受 `k` 和 `τ` 影响；若 `x_r ∉ K_r^T`，则值为 0。 |
| **CondOTMass**            | 在 `x_r ∈ K_r^T` 条件下计算的 OTMass。    | 用于刻画支持集内部（within-support）的概率分配。         |
| **Top1Prob**              | `p̃_T^{k,τ}_{r,(1)}`              | 教师模型 Top-1 token 的概率。                   |
| **Entropy / NormEntropy** | `H(p̃_T^{k,τ})` / `H / log k`     | 衡量教师目标分布的平滑程度（softness）。                |
| **RawTopKMass**           | 截断前 Top-`k` token 所占的概率质量。        | 用于衡量 Top-`k` 截断丢弃了多少概率质量。               |

---

这些指标可以进一步理解如下。

### Coverage@k

表示：

> 当前真实 token 是否位于教师模型 Top-`k` 集合内。

因此它回答的是：

> **教师模型是否认可（cover）这个真实 token？**

Coverage 越高，说明教师模型越经常认为真实 token 是合理候选。

---

### OTRank

表示：

> 真实 token 在教师模型排序中的名次。

例如：

```
Teacher:

1. "is"
2. "was"
3. "has"

Ground Truth:

"was"
```

那么：

```
OTRank = 2
```

如果真实 token 排名很靠后，则说明：

教师模型与数据之间存在较大的分歧。

---

### OTMass

表示：

> 教师模型究竟给真实 token 分配了多少概率。

例如：

```
Teacher：

is   0.60
was  0.30
has  0.10
```

真实 token 为：

```
was
```

那么：

```
OTMass = 0.30
```

它比 OTRank 更细致。

因为：

```
Rank=2
```

可能对应：

```
0.49
```

也可能对应：

```
0.01
```

OTMass 能够直接刻画：

> **教师模型究竟有多相信真实 token。**

---

### CondOTMass

由于很多 token 根本不在 Top-`k` 中，

论文进一步计算：

```
仅统计：

x ∈ Top-k
```

情况下的 OTMass。

这样可以将两个因素解耦：

第一步：

```
Coverage
```

决定：

> 是否进入支持集。

第二步：

```
CondOTMass
```

决定：

> 在支持集内部获得多少概率。

因此，

Coverage 与 CondOTMass 分别对应：

* 是否被教师认可；
* 被认可之后认可到什么程度。

---

### Top1Prob

表示：

教师模型最可能 token 的概率。

例如：

```
Top1

0.98
```

说明：

教师预测极其自信。

如果：

```
0.25
```

则说明：

教师分布比较平缓。

---

### Entropy

Entropy 衡量：

> **教师目标分布是否集中。**

Entropy 越小：

```
Teacher

0.99
0.01
```

监督越尖锐（sharp）。

Entropy 越大：

```
0.30
0.28
0.24
0.18
```

说明：

教师认为多个 token 都是合理答案。

---

### RawTopKMass

它统计的是：

Top-`k` 截断之前，

Top-`k` token 一共占据多少概率。

例如：

```
Top256

总质量 = 96%
```

那么：

```
RawTopKMass = 0.96
```

意味着：

只有

```
4%
```

概率被截断掉。

因此它衡量的是：

> **Top-`k` 截断到底损失了多少教师信息。**

---

# 实验设置（Experimental Setup）

## 剪枝–蒸馏 Scaling Ladder（Pruning–Distillation Scaling Ladder）

为了能够在保留大模型行为先验（behavioral priors）的同时开展可控实验，论文基于 **DeepSeek V3 Base** 构建了一条多阶段的**剪枝–蒸馏（pruning–distillation）**模型链（scaling ladder）。

| 阶段                      | 父模型   | 总参数 / 激活参数 | 蒸馏 Token 数 |
| ----------------------- | ----- | ---------- | ---------- |
| `M_0`（DeepSeek V3 Base） | —     | 671B / 37B | —          |
| `M_1`                   | `M_0` | 128B / 19B | 1T         |
| `M_2`                   | `M_1` | 59B / 10B  | 200B       |
| `M_3`（学生模型）             | `M_2` | 24B / 4.5B | 200B       |

其中：

**Teacher：**

DeepSeek V3.1 Base（内部评测性能优于 V3 Base）。

**Student：**

`M_3`。

所有 RQ1/RQ2 实验均使用 **100B token** 进行训练，数据组成如下：

* 通用领域（general）：33%
* 数学（math）：40%
* 代码（code）：27%

---

## 评测（Evaluation）

论文采用如下评测体系：

* **13 个 Benchmark：**

  * PIQA
  * HellaSwag
  * MMLU
  * TriviaQA
  * RACE
  * DROP
  * MMLU-Pro
  * BBH
  * C-Eval
  * CMMLU
  * MBPP
  * GSM8K
  * MATH

* **Pass@K 评测：**

  在以下数据集上测试：

  * MATH-500
  * AIME 2024
  * AIME 2025
  * HumanEval

  其中：

  ```
  K ∈ {1,16,64,128}
  ```

用于评估模型在多次采样条件下的推理与代码生成能力。

下面是第三部分的翻译。我继续**保留原 Markdown 结构**，其中图片的 Markdown 写法保持原样，不做修改。

---

# 关键结果（Key Results）

## LM 与 KD —— 能力画像呈现系统性分化

| Benchmark          |        LM |        KD |            差距 |
| ------------------ | --------: | --------: | ------------: |
| MMLU-Pro           | **43.98** |     39.74 |  **LM +4.24** |
| MATH (Minerva)     | **43.38** |     40.64 |  **LM +2.74** |
| MATH-500 Pass@16   | **60.75** |     30.89 | **LM +29.86** |
| AIME 2025 Pass@128 | **23.33** |      0.00 | **LM +23.33** |
| HumanEval Pass@1   | **29.63** |     22.79 |  **LM +6.84** |
| PIQA               |     79.16 | **80.63** |  **KD +1.47** |
| DROP               |     51.59 | **53.37** |  **KD +1.78** |
| MBPP               |     53.00 | **54.80** |  **KD +1.80** |

实验结果表明，在**完全相同的训练数据、初始化方式和训练配置**下：

* **LM** 更擅长：

  * 高难度推理（reasoning）
  * 数学（math）
  * Pass@K 等需要多样化采样的生成任务
  * 代码生成（HumanEval）

* **KD** 更擅长：

  * 常识推理（commonsense）
  * 阅读理解（DROP）
  * 程序生成（MBPP）
  * 知识密集型任务

因此，两种训练目标形成了**系统性的能力分化（capability divergence）**，而不是某一方全面优于另一方。

---

## Top-`k` —— Coverage 与 Sharpness 的权衡

![Figure 1: Diagnostic statistics for top-k truncated teacher distributions at τ = 1.](figures/topk_diagnostic_metrics_1.png)

*图 1：在 `τ = 1` 时，Top-`k` 截断教师分布的诊断统计结果。*

| `k` | Coverage@k | CondOTMass | Top1Prob | NormEntropy |
| --: | ---------: | ---------: | -------: | ----------: |
|   1 |     0.7157 |     1.0000 |   1.0000 |           — |
|   4 |     0.8640 |     0.7677 |   0.7854 |      0.3822 |
|  16 |     0.9349 |     0.6895 |   0.7396 |      0.3017 |
|  64 |     0.9704 |     0.6577 |   0.7256 |      0.2442 |
| 256 |     0.9876 |     0.6439 |   0.7206 |      0.2014 |

此外：

* OTRank 中位数约为 **1**
* 90% 分位数为 **7.77**
* 99% 分位数为 **256.7**

说明：

> 大多数真实 token 本身就已经位于教师模型 **Top-4 或 Top-16** 内。

因此：

增大 `k` 的主要作用并不是让更多真实 token 被覆盖，而是：

> **不断加入概率很小的尾部 token（tail tokens），从而稀释（dilute）监督信号。**

换句话说：

随着 `k` 增大：

* Coverage 持续提高；
* CondOTMass 持续下降；
* Top1Prob 持续下降；
* Entropy 持续下降（归一化后）。

因此：

> **`k` 本质上控制的是 Coverage 与 Sharpness 之间的权衡。**

---

## Temperature —— 仅改变支持集内部的概率分配

![Figure 2: Diagnostic statistics for top-256 truncated teacher distributions under different distillation temperatures.](figures/temperature_diagnostic_metrics_1.png)

*图 2：固定 Top-256 时，不同蒸馏温度下教师分布的诊断统计。*

固定：

```text
k = 256
```

改变温度 `τ` 后得到：

|  `τ` | OTMass | CondOTMass | Top1Prob | Entropy |
| ---: | -----: | ---------: | -------: | ------: |
|   →0 | 0.7157 |     1.0000 |   1.0000 |       — |
| 0.25 | 0.7110 |     0.7200 |   0.9280 |  0.1909 |
|  0.5 | 0.6959 |     0.7046 |   0.8586 |  0.4345 |
|  1.0 | 0.6359 |     0.6439 |   0.7206 |  1.1166 |

实验发现：

无论如何调整 `τ`，

下面两个指标几乎完全保持不变：

* Coverage@k
* OTRank

因为：

> **温度不会改变 Top-`k` 集合的成员，只会重新分配这些成员之间的概率。**

随着温度升高：

* Top1Prob 下降；
* OTMass 下降；
* Entropy 上升；

即：

教师模型逐渐将概率质量从 Top-1 token 转移到更多低排名 token。

因此：

> **`τ` 仅控制支持集（support）内部的概率分配，而不会改变支持集本身。**

这也是论文强调：

> **`k` 与 `τ` 是两个相互独立（orthogonal）的控制自由度。**

---

## 领域级路由优于两种单一训练目标

| Benchmark          |    LM |    KD | **Domain Routing** |
| ------------------ | ----: | ----: | -----------------: |
| MMLU-Pro           | 43.98 | 39.74 |          **43.72** |
| MATH (Minerva)     | 43.38 | 40.64 |          **43.58** |
| BBH                | 69.90 | 69.33 |          **70.74** |
| MBPP               | 53.00 | 54.80 |          **58.40** |
| MATH-500 Pass@128  | 84.40 | 63.00 |          **83.20** |
| AIME 2024 Pass@128 | 23.33 | 10.00 |          **33.33** |
| AIME 2025 Pass@128 | 23.33 |  0.00 |          **23.33** |

这里的 **Domain Routing** 指：

* 数学、代码数据使用 LM；
* 通用数据使用 KD。

实验结果显示：

这种策略能够：

* 保留 LM 在数学和推理任务上的优势；
* 保留 KD 在知识和常识任务上的优势；

甚至：

> **在多个 Benchmark 上超过两种单一训练目标。**

例如：

MBPP：

```text
LM：53.00
KD：54.80
Domain：58.40
```

明显超过两种基线。

---

## 为什么领域路由有效——按领域划分的诊断指标分析

![Figure 3: Domain-stratified diagnostic metrics under top-256 distillation with τ = 1.](figures/domain_routing_diagnostic_metrics_1.png)

*图 3：Top-256、`τ = 1` 条件下，不同领域数据的诊断指标。*

| Domain  | Coverage@k | CondOTMass | Entropy |
| ------- | ---------: | ---------: | ------: |
| general |     0.9797 |     0.4958 |  1.6553 |
| math    |     0.9888 |     0.6556 |  1.0579 |
| code    |     0.9945 |     0.8258 |  0.4932 |

可以发现：

数学和代码数据具有：

* 更高的 Coverage；
* 更高的 CondOTMass；
* 更低的 Entropy。

这意味着：

教师模型预测与真实 token 高度一致。

因此：

LM 对真实 token 的完全强化能够获得更多有效监督。

相反：

通用数据具有：

* 更低的 CondOTMass；
* 更高的 Entropy。

说明：

教师模型认为存在大量合理候选。

因此：

KD 能够利用这些替代 token（alternative tokens）提供额外监督，从而优于 LM。

论文据此认为：

> **领域之间 teacher–data alignment 的差异，是领域级路由能够成功的根本原因。**

---

## Token 级路由——更细粒度并不意味着更好的效果

| Benchmark          |        LM |    KD | OTMass (T=0.37) | Entropy (T=0.5) | **Domain** |
| ------------------ | --------: | ----: | --------------: | --------------: | ---------: |
| MMLU-Pro           |     43.98 | 39.74 |           41.42 |           43.86 |  **43.72** |
| MATH (Minerva)     |     43.38 | 40.64 |           41.98 |           41.94 |  **43.58** |
| MATH-500 Pass@128  | **84.40** | 63.00 |           74.00 |           69.40 |      83.20 |
| AIME 2024 Pass@128 |     23.33 | 10.00 |            3.33 |           13.33 |  **33.33** |
| RACE               | **46.89** | 45.74 |           45.65 |        39.33 ⚠️ |      44.98 |

论文进一步尝试：

根据每一个 token 的诊断指标决定：

* 使用 LM；
* 或使用 KD。

然而结果发现：

> **Token 级路由反而不如简单的领域级路由。**

与此同时，

如果必须进行 token 级路由，

那么：

> **OTMass 明显优于 Entropy。**

原因在于：

OTMass 直接对应论文 Eq.(11) 中梯度差异的核心项：

```text
1[x=x_r]-p̃_T
```

因此：

它能够真实反映：

> **教师模型是否支持当前真实 token。**

而 Entropy 只能衡量：

教师模型是否自信。

它无法区分：

* 教师非常自信且预测正确；
* 教师非常自信但预测错误。

因此：

Entropy 与真正决定 LM/KD 差异的机制并不一致。

论文最终总结：

> **路由效果并不由粒度（granularity）决定，而是由路由信号是否能够准确反映 teacher–data alignment 决定。**

---

# 这一工作的重新定义（What This Reframes）

本文认为：

持续预训练中的 Off-Policy Distillation：

**既不是**一个需要统一调节的全局训练配方（global recipe），

**也不是**

LM 与 KD 的二元选择问题。

相反，

它应被理解为一个：

> **结构化（structured）、数据条件（data-conditional）的监督设计问题（supervision design）**。

这一设计问题包含三个相互耦合的自由度：

1. **路由信号（Routing Signal）**

   即：

   根据什么数据属性决定采用哪一种训练目标。

2. **目标函数家族（Objective Family）**

   包括：

   ```
   (α, k, τ, divergence)
   ```

   它们共同决定监督信号位于：

   > **真实 token 强化（reinforcement） ↔ 教师替代监督（alternatives）**

   这一连续谱（spectrum）中的哪个位置。

3. **路由策略（Routing Rule）**

   包括：

   * 固定阈值（fixed threshold）；
   * 学习得到的策略（learned policy）；
   * 面向能力（capability-aware）的映射。

未来工作包括：

1. 进一步研究：

   > Packed Sequence 引入的上下文污染（context contamination）是否是 Token-level Routing 性能下降的真正原因。

2. 将路由从：

   ```
   LM ↔ KD
   ```

   扩展到：

   **KD 家族内部不同配置之间的路由。**

3. 学习一种真正基于 **teacher–data alignment** 的路由策略，而不是仅依赖教师模型的不确定性（teacher-side uncertainty）。

---

# 引用（Citation）

```bibtex
@article{yuan2026letdatadecide,
  title   = {Let the Data Decide: Supervision Analysis, Capability Trade-offs,
             and Adaptive Objective Routing in Continued Pre-training via
             Off-Policy Distillation},
  author  = {Yuan, Jiangan and Li, Zhixuan and Xu, Han},
  journal = {arXiv preprint},
  year    = {2026}
}
```
